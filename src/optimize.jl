#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 Optimizer object and algorithm
=========================================================#

mutable struct Optimizer <: MOI.AbstractOptimizer
    log_level::Int                          # Verbosity flag: 0 for quiet, higher for basic solve info
    timeout::Float64                        # Time limit for algorithm (in seconds)
    rel_gap::Float64                        # Relative optimality gap termination condition
    mip_solver_drives::Union{Nothing,Bool} # Let MIP solver manage convergence ("branch and cut")
    mip_solver::Union{Nothing,MOI.OptimizerWithAttributes}  # MIP solver constructor
    cont_solver::Union{Nothing,MOI.OptimizerWithAttributes} # Continuous NLP solver constructor

    mip_optimizer::Union{Nothing,MOI.ModelLike}        # MIP optimizer instantiated from `mip_solver`
    cont_optimizer::Union{Nothing,MOI.ModelLike}       # Continuous NLP optimizer instantiated from `cont_solver`
    infeasible_optimizer::Union{Nothing,MOI.ModelLike} # Continuous NLP optimizer instantiated from `cont_solver`, used for infeasible subproblems

    nlp_obj_var::Union{Nothing,MOI.VariableIndex}  # new MIP objective function if the original is nonlinear
    mip_variables::Vector{MOI.VariableIndex}        # Variable indices of `mip_optimizer`
    cont_variables::Vector{MOI.VariableIndex}       # Variable indices of `cont_optimizer`
    infeasible_variables::Vector{MOI.VariableIndex} # Variable indices of `infeasible_optimizer`

    # Slack variable indices for `infeasible_optimizer`
    nl_slack_variables::Union{Nothing,Vector{MOI.VariableIndex}} # for the nonlinear constraints
    quad_LT_slack::Union{Nothing,Vector{MOI.VariableIndex}}      # for the less than constraints
    quad_GT_slack::Union{Nothing,Vector{MOI.VariableIndex}}      # for the greater than constraints

    # Quadratic constraints for `infeasible_optimizer`
    quad_LT_infeasible_con::Union{
        Nothing,
        Vector{
            MOI.ConstraintIndex{
                MOI.ScalarQuadraticFunction{Float64},
                MOI.LessThan{Float64},
            },
        },
    } # `q - slack <= ub`
    quad_GT_infeasible_con::Union{
        Nothing,
        Vector{
            MOI.ConstraintIndex{
                MOI.ScalarQuadraticFunction{Float64},
                MOI.GreaterThan{Float64},
            },
        },
    } # `q + slack >= lb`
    infeasible_evaluator::_InfeasibleNLPEvaluator # NLP evaluator used for `infeasible_optimizer`
    int_indices::BitSet                          # Indices of discrete variables

    nlp_block::Union{Nothing,MOI.NLPBlockData}           # NLP block set to `Optimizer`
    objective::Union{Nothing,MOI.AbstractScalarFunction} # Objective function set to `Optimizer`
    quad_LT::Vector{
        Tuple{MOI.ScalarQuadraticFunction{Float64},MOI.LessThan{Float64}},
    }   # Cached quadratic less than constraints
    quad_GT::Vector{
        Tuple{MOI.ScalarQuadraticFunction{Float64},MOI.GreaterThan{Float64}},
    }   # Cached quadratic greater than constraints
    status::MOI.TerminationStatusCode # Termination status to be returned
    incumbent::Vector{Float64}    # Starting values set and then current best nonlinear feasible solution
    new_incumb::Bool              # `true` if a better nonlinear feasible solution was found
    mip_solution::Vector{Float64} # MIP solution cached for used to check redundancy of lazy constraint
    total_time::Float64           # Total solve time
    objective_value::Float64      # Objective value corresponding to `incumbent`
    objective_bound::Float64      # Best objective bound found by MIP
    objective_gap::Float64        # Objective gap between objective value and bound
    num_iters_or_callbacks::Int   # Either the number of iterations or the number of calls to the lazy constraint callback if `mip_solver_drives`

    function Optimizer()
        model = new()
        model.log_level = 1
        model.timeout = Inf
        model.rel_gap = 1e-5
        model.mip_solver_drives = nothing
        model.mip_solver = nothing
        model.cont_solver = nothing
        MOI.empty!(model)
        return model
    end
end

function _print_model_summary(model)
    obj_type = if !isnothing(model.nlp_block) && model.nlp_block.has_objective
        "nonlinear"
    elseif model.objective isa MOI.ScalarQuadraticFunction{Float64}
        "quadratic"
    else
        "linear"
    end
    ncont = length(model.cont_variables)
    nint = length(model.int_indices)
    nnlp = if isnothing(model.nlp_block)
        0
    else
        length(model.nlp_block.constraint_bounds)
    end
    nquad = length(model.quad_LT) + length(model.quad_GT)
    println(
        "\nMINLP has a $obj_type objective, $ncont continuous variables, " *
        "$nint integer variables, $nnlp nonlinear constraints, and " *
        "$nquad quadratic constraints",
    )

    alg = model.mip_solver_drives ? "MIP-solver-driven" : "iterative"
    println("\nPavito started, using $alg method...")
    return
end

function _clean_up_algorithm(model, start, mip_time, nlp_time)
    model.total_time = time() - start
    if model.log_level > 0
        println("\nPavito finished...\n")
        Printf.@printf("Status           %13s\n", model.status)
        Printf.@printf("Objective value  %13.5f\n", model.objective_value)
        Printf.@printf("Objective bound  %13.5f\n", model.objective_bound)
        Printf.@printf("Objective gap    %13.5f\n", model.objective_gap)
        if !model.mip_solver_drives
            Printf.@printf(
                "Iterations       %13d\n",
                model.num_iters_or_callbacks
            )
        else
            Printf.@printf(
                "Callbacks        %13d\n",
                model.num_iters_or_callbacks
            )
        end
        Printf.@printf("Total time       %13.5f sec\n", model.total_time)
        Printf.@printf("MIP total time   %13.5f sec\n", mip_time)
        Printf.@printf("NLP total time   %13.5f sec\n", nlp_time)
        println()
    end
    flush(stdout)
    return
end

abstract type _AbstractAlgortithm end

struct _MIPSolverDrivenAlgorithm end

function _run_algorithm(
    ::_MIPSolverDrivenAlgorithm,
    model,
    start,
    mip_time,
    nlp_time,
    jac_IJ,
    jac_V,
    grad_f,
    is_max,
)
    comp = is_max ? (>) : (<)
    cache_contsol = Dict{Vector{Float64},Vector{Float64}}()
    function lazy_callback(cb)
        model.num_iters_or_callbacks += 1
        model.mip_solution = MOI.get(
            model.mip_optimizer,
            MOI.CallbackVariablePrimal(cb),
            model.mip_variables,
        )
        round_mipsol = round.(model.mip_solution)
        if any(
            i -> abs(model.mip_solution[i] - round_mipsol[i]) > 1e-5,
            model.int_indices,
        )
            # The solution is integer-infeasible; see:
            # https://github.com/jump-dev/GLPK.jl/issues/146
            # https://github.com/jump-dev/MathOptInterface.jl/pull/1172
            @warn "Integer-infeasible solution in lazy callback"
            return
        end
        # If integer assignment has been seen before, use cached point
        if haskey(cache_contsol, round_mipsol)
            cont_solution = cache_contsol[round_mipsol]
        else  # Try to solve new subproblem, update incumbent if feasible.
            start_nlp = time()
            cont_solution = _solve_subproblem(model, comp)
            nlp_time += time() - start_nlp
            cache_contsol[round_mipsol] = cont_solution
        end
        # Add gradient cuts to MIP model from NLP solution
        if all(isfinite, cont_solution)
            _add_cuts(model, cont_solution, jac_IJ, jac_V, grad_f, is_max, cb)
        else
            @warn "no cuts could be added, Pavito should be terminated"
            # TODO terminate the solver once there is a
            # solver-independent callback for that in MOI
            return
        end
    end
    MOI.set(model.mip_optimizer, MOI.LazyConstraintCallback(), lazy_callback)
    function heuristic_callback(cb)
        # If have a new best feasible solution since last heuristic
        # solution added, set MIP solution to the new incumbent.
        if !model.new_incumb
            return
        elseif isnothing(model.nlp_obj_var)
            MOI.submit(
                model.mip_optimizer,
                MOI.HeuristicSolution(cb),
                model.mip_variables,
                model.incumbent,
            )
        else
            MOI.submit(
                model.mip_optimizer,
                MOI.HeuristicSolution(cb),
                vcat(model.mip_variables, model.nlp_obj_var),
                vcat(model.incumbent, model.objective_value),
            )
        end
        model.new_incumb = false
        return
    end
    MOI.set(model.mip_optimizer, MOI.HeuristicCallback(), heuristic_callback)
    if isfinite(model.timeout)
        MOI.set(
            model.mip_optimizer,
            MOI.TimeLimitSec(),
            max(1.0, model.timeout - (time() - start)),
        )
    end
    MOI.optimize!(model.mip_optimizer)
    mip_status = MOI.get(model.mip_optimizer, MOI.TerminationStatus())
    if mip_status == MOI.OPTIMAL
        model.status = MOI.LOCALLY_SOLVED
    elseif mip_status == MOI.ALMOST_OPTIMAL
        model.status = MOI.ALMOST_LOCALLY_SOLVED
    else
        model.status = mip_status
    end
    model.objective_bound = MOI.get(model.mip_optimizer, MOI.ObjectiveBound())
    _update_gap(model, is_max)
    return mip_time, nlp_time
end

struct _IterativeAlgorithm end

function _run_algorithm(
    ::_IterativeAlgorithm,
    model,
    start,
    mip_time,
    nlp_time,
    jac_IJ,
    jac_V,
    grad_f,
    is_max,
)
    comp = is_max ? (>) : (<)
    prev_mip_solution = fill(NaN, length(model.mip_variables))
    while time() - start < model.timeout
        model.num_iters_or_callbacks += 1
        # Set remaining time limit on MIP solver
        if isfinite(model.timeout)
            MOI.set(
                model.mip_optimizer,
                MOI.TimeLimitSec(),
                max(1.0, model.timeout - (time() - start)),
            )
        end
        # Solve MIP model
        start_mip = time()
        MOI.optimize!(model.mip_optimizer)
        mip_status = MOI.get(model.mip_optimizer, MOI.TerminationStatus())
        mip_time += time() - start_mip
        # Finish if MIP was infeasible or if problematic status
        if mip_status in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
            model.status = MOI.LOCALLY_INFEASIBLE
            break
        elseif (mip_status != MOI.OPTIMAL) && (mip_status != MOI.ALMOST_OPTIMAL)
            @warn "MIP solver status was $mip_status, terminating Pavito"
            model.status = mip_status
            break
        end
        model.mip_solution = MOI.get(
            model.mip_optimizer,
            MOI.VariablePrimal(),
            model.mip_variables,
        )
        # Update best bound from MIP bound
        mip_obj_bound = MOI.get(model.mip_optimizer, MOI.ObjectiveBound())
        if isfinite(mip_obj_bound) && comp(model.objective_bound, mip_obj_bound)
            model.objective_bound = mip_obj_bound
        end
        _update_gap(model, is_max)
        _print_gap(model, start)
        if _check_progress(model, prev_mip_solution)
            break
        end
        # Try to solve new subproblem, update incumbent if feasible
        start_nlp = time()
        cont_solution = _solve_subproblem(model, comp)
        nlp_time += time() - start_nlp
        _update_gap(model, is_max)
        if _check_progress(model, prev_mip_solution)
            break
        end
        # Add gradient cuts to MIP model from NLP solution
        if all(isfinite, cont_solution)
            _add_cuts(model, cont_solution, jac_IJ, jac_V, grad_f, is_max)
        else
            @warn "no cuts could be added, terminating Pavito"
            break
        end
        # TODO warmstart MIP from incumbent
        prev_mip_solution = model.mip_solution
        flush(stdout)
    end
    return mip_time, nlp_time
end

function MOI.optimize!(model::Optimizer)
    model.status = MOI.OPTIMIZE_NOT_CALLED
    fill!(model.incumbent, NaN)
    model.new_incumb = false
    model.total_time = 0.0
    model.objective_value = NaN
    model.objective_bound = NaN
    model.objective_gap = Inf
    model.num_iters_or_callbacks = 0
    if isempty(model.int_indices) && model.log_level >= 1
        @warn "No variables of type integer or binary; call the continuous " *
              "solver directly for pure continuous problems."
    end
    if (!isnothing(model.nlp_block) && model.nlp_block.has_objective) ||
       model.objective isa MOI.ScalarQuadraticFunction{Float64}
        if isnothing(model.nlp_obj_var)
            model.nlp_obj_var = MOI.add_variable(model.mip_optimizer)
            MOI.set(
                model.mip_optimizer,
                MOI.ObjectiveFunction{MOI.VariableIndex}(),
                model.nlp_obj_var,
            )
        end
    else
        if !isnothing(model.nlp_obj_var)
            MOI.delete(model.mip_optimizer, model.nlp_obj_var)
            model.nlp_obj_var = nothing
        end
    end
    start = time()
    nlp_time = 0.0
    mip_time = 0.0
    is_max =
        MOI.get(model.cont_optimizer, MOI.ObjectiveSense()) == MOI.MAX_SENSE
    model.objective_value = is_max ? -Inf : Inf
    model.objective_bound = -model.objective_value
    if model.log_level > 0
        _print_model_summary(model)
    end
    flush(stdout)
    if isnothing(model.nlp_block)
        jac_V = jac_IJ = nothing
    else
        MOI.initialize(
            model.nlp_block.evaluator,
            intersect(
                [:Grad, :Jac, :Hess],
                MOI.features_available(model.nlp_block.evaluator),
            ),
        )
        jac_IJ = MOI.jacobian_structure(model.nlp_block.evaluator)
        jac_V = zeros(length(jac_IJ))
    end
    grad_f = zeros(length(model.cont_variables))
    # Solve initial continuous relaxation NLP model.
    start_nlp = time()
    MOI.optimize!(model.cont_optimizer)
    nlp_time += time() - start_nlp
    ini_nlp_status = MOI.get(model.cont_optimizer, MOI.TerminationStatus())
    cont_solution, cont_obj = nothing, 0.0
    if ini_nlp_status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)
        cont_solution = MOI.get(
            model.cont_optimizer,
            MOI.VariablePrimal(),
            model.cont_variables,
        )
        cont_obj = MOI.get(model.cont_optimizer, MOI.ObjectiveValue())
        if isempty(model.int_indices)
            model.objective_value = cont_obj
            model.objective_bound = cont_obj
            _update_gap(model, is_max)
            model.mip_solution = Float64[]
            model.incumbent = cont_solution
        else
            _add_cuts(model, cont_solution, jac_IJ, jac_V, grad_f, is_max)
        end
    elseif ini_nlp_status == MOI.DUAL_INFEASIBLE
        # The integrality constraints may make the MINLP bounded so we continue
        @warn "initial NLP relaxation unbounded"
    elseif ini_nlp_status == MOI.NORM_LIMIT
        # Ipopt usually ends with `Diverging_Iterates` for unbounded problems,
        # this gets converted to `MOI.NORM_LIMIT`
        @warn "initial NLP relaxation terminated with status `NORM_LIMIT` " *
              "which usually means that the problem is unbounded"
    else
        if ini_nlp_status in
           (MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE, MOI.ALMOST_INFEASIBLE)
            # The original problem is infeasible too
            @warn "initial NLP relaxation infeasible"
            model.status = ini_nlp_status
        else
            @warn "NLP solver failure: initial NLP relaxation terminated with " *
                  "status $ini_nlp_status"
            model.status = ini_nlp_status
        end
        return
    end
    flush(stdout)

    # If there is no integrality, exit early! We've already solved the NLP.
    if isempty(model.int_indices)
        model.status = ini_nlp_status
        _clean_up_algorithm(model, start, mip_time, nlp_time)
        return
    end
    # Set a VariablePrimalStart if supported by the solver
    if !isnothing(cont_solution) &&
       MOI.supports(
           model.mip_optimizer,
           MOI.VariablePrimalStart(),
           MOI.VariableIndex,
       ) &&
       all(isfinite, cont_solution)
        MOI.set(
            model.mip_optimizer,
            MOI.VariablePrimalStart(),
            model.mip_variables,
            cont_solution,
        )

        if !isnothing(model.nlp_obj_var)
            MOI.set(
                model.mip_optimizer,
                MOI.VariablePrimalStart(),
                model.nlp_obj_var,
                cont_obj,
            )
        end
    end
    algorithm = if model.mip_solver_drives
        _MIPSolverDrivenAlgorithm()
    else
        _IterativeAlgorithm()
    end
    mip_time, nlp_time = _run_algorithm(
        algorithm,
        model,
        start,
        mip_time,
        nlp_time,
        jac_IJ,
        jac_V,
        grad_f,
        is_max,
    )
    flush(stdout)
    _clean_up_algorithm(model, start, mip_time, nlp_time)
    return
end

function _update_gap(model::Optimizer, is_max::Bool)
    # Update gap if best bound and best objective are finite
    if isfinite(model.objective_value) && isfinite(model.objective_bound)
        model.objective_gap =
            (model.objective_value - model.objective_bound) /
            (abs(model.objective_value) + 1e-5)
        if is_max
            model.objective_gap = -model.objective_gap
        end
    end
    return
end

function _check_progress(model::Optimizer, prev_mip_solution)
    # Finish if optimal or cycling integer solutions.
    int_ind = collect(model.int_indices)
    if model.objective_gap <= model.rel_gap
        model.status = MOI.LOCALLY_SOLVED
        return true
    elseif round.(prev_mip_solution[int_ind]) ==
           round.(model.mip_solution[int_ind])
        @warn "mixed-integer cycling detected, terminating Pavito"
        if isfinite(model.objective_gap)
            model.status = MOI.ALMOST_LOCALLY_SOLVED
        else
            model.status = MOI.OTHER_ERROR
        end
        return true
    end
    return false
end

function _fix_int_vars(
    optimizer::MOI.ModelLike,
    vars,
    mip_solution,
    int_indices,
)
    for i in int_indices
        vi = vars[i]
        idx = vi.value
        ci = MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}}(idx)
        MOI.is_valid(optimizer, ci) && MOI.delete(optimizer, ci)
        ci =
            MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}}(idx)
        MOI.is_valid(optimizer, ci) && MOI.delete(optimizer, ci)
        ci = MOI.ConstraintIndex{MOI.VariableIndex,MOI.EqualTo{Float64}}(idx)
        set = MOI.EqualTo(mip_solution[i])

        if MOI.is_valid(optimizer, ci)
            MOI.set(optimizer, MOI.ConstraintSet(), ci, set)
        else
            MOI.add_constraint(optimizer, vi, set)
        end
    end
    return
end

# solve NLP subproblem defined by integer assignment
function _solve_subproblem(model::Optimizer, comp::Function)
    _fix_int_vars(
        model.cont_optimizer,
        model.cont_variables,
        model.mip_solution,
        model.int_indices,
    )
    MOI.optimize!(model.cont_optimizer)
    if MOI.get(model.cont_optimizer, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        # Subproblem is feasible, check if solution is new incumbent
        nlp_objective_value =
            MOI.get(model.cont_optimizer, MOI.ObjectiveValue())
        nlp_solution = MOI.get(
            model.cont_optimizer,
            MOI.VariablePrimal(),
            model.cont_variables,
        )
        if comp(nlp_objective_value, model.objective_value)
            model.objective_value = nlp_objective_value
            copyto!(model.incumbent, nlp_solution)
            model.new_incumb = true
        end
        return nlp_solution
    end
    # Assume subproblem is infeasible, so solve infeasible recovery NLP
    # subproblem.
    if (
           !isnothing(model.nlp_block) &&
           !isempty(model.nlp_block.constraint_bounds) &&
           isnothing(model.nl_slack_variables)
       ) ||
       (!isempty(model.quad_LT) && isnothing(model.quad_LT_slack)) ||
       (!isempty(model.quad_GT) && isnothing(model.quad_GT_slack))
        if !isnothing(model.nl_slack_variables)
            obj = MOI.ScalarAffineFunction(
                MOI.ScalarAffineTerm.(1.0, model.nl_slack_variables),
                0.0,
            )
        else
            obj = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[], 0.0)
        end
        function _add_to_obj(vi::MOI.VariableIndex)
            return push!(obj.terms, MOI.ScalarAffineTerm(1.0, vi))
        end
        if !isnothing(model.nlp_block) &&
           !isempty(model.nlp_block.constraint_bounds)
            bounds = copy(model.nlp_block.constraint_bounds)
            model.infeasible_evaluator = _InfeasibleNLPEvaluator(
                model.nlp_block.evaluator,
                length(model.infeasible_variables),
                falses(length(model.nlp_block.constraint_bounds)),
            )
            model.nl_slack_variables = MOI.add_variables(
                _infeasible(model),
                length(model.nlp_block.constraint_bounds),
            )
            for i in eachindex(model.nlp_block.constraint_bounds)
                _add_to_obj(model.nl_slack_variables[i])
                push!(bounds, MOI.NLPBoundsPair(0.0, Inf))
                set = _bound_set(model, i)
                if set isa MOI.LessThan{Float64}
                    model.infeasible_evaluator.minus[i] = true
                end
            end
            MOI.set(
                _infeasible(model),
                MOI.NLPBlock(),
                MOI.NLPBlockData(bounds, model.infeasible_evaluator, false),
            )
        end
        # We need to add quadratic variables afterwards because
        # `_InfeasibleNLPEvaluator` assumes the original variables are directly
        # followed by the NL slacks.
        if !isempty(model.quad_LT)
            model.quad_LT_slack =
                MOI.add_variables(_infeasible(model), length(model.quad_LT))
            model.quad_LT_infeasible_con = map(eachindex(model.quad_LT)) do i
                (func, set) = model.quad_LT[i]
                new_func = MOI.Utilities.operate(
                    -,
                    Float64,
                    func,
                    model.quad_LT_slack[i],
                )
                return MOI.add_constraint(_infeasible(model), new_func, set)
            end
            for vi in model.quad_LT_slack
                _add_to_obj(vi)
            end
        end
        if !isempty(model.quad_GT)
            model.quad_GT_slack =
                MOI.add_variables(_infeasible(model), length(model.quad_GT))
            model.quad_GT_infeasible_con = map(eachindex(model.quad_GT)) do i
                (func, set) = model.quad_GT[i]
                new_func = MOI.Utilities.operate(
                    +,
                    Float64,
                    func,
                    model.quad_GT_slack[i],
                )
                return MOI.add_constraint(_infeasible(model), new_func, set)
            end
            for vi in model.quad_GT_slack
                _add_to_obj(vi)
            end
        end
        MOI.set(_infeasible(model), MOI.ObjectiveFunction{typeof(obj)}(), obj)
    end
    _fix_int_vars(
        model.infeasible_optimizer,
        model.infeasible_variables,
        model.mip_solution,
        model.int_indices,
    )
    MOI.set(
        _infeasible(model),
        MOI.VariablePrimalStart(),
        model.infeasible_variables,
        model.mip_solution,
    )
    if !isnothing(model.nlp_block) &&
       !isempty(model.nlp_block.constraint_bounds)
        fill!(model.infeasible_evaluator.minus, false)
        g = zeros(length(model.nlp_block.constraint_bounds))
        MOI.eval_constraint(model.nlp_block.evaluator, g, model.mip_solution)
        for i in eachindex(model.nlp_block.constraint_bounds)
            bounds = model.nlp_block.constraint_bounds[i]
            val = if model.infeasible_evaluator.minus[i]
                g[i] - bounds.upper
            else
                bounds.lower - g[i]
            end
            # Sign of the slack changes if the constraint direction changes.
            MOI.set(
                _infeasible(model),
                MOI.VariablePrimalStart(),
                model.nl_slack_variables[i],
                max(0.0, val),
            )
        end
    end
    for i in eachindex(model.quad_LT)
        val =
            _eval_func(model.mip_solution, model.quad_LT[i][1]) -
            model.quad_LT[i][2].upper
        MOI.set(
            _infeasible(model),
            MOI.VariablePrimalStart(),
            model.quad_LT_slack[i],
            max(0.0, val),
        )
    end
    for i in eachindex(model.quad_GT)
        val =
            model.quad_GT[i][2].lower -
            _eval_func(model.mip_solution, model.quad_GT[i][1])
        MOI.set(
            _infeasible(model),
            MOI.VariablePrimalStart(),
            model.quad_GT_slack[i],
            max(0.0, val),
        )
    end
    MOI.optimize!(model.infeasible_optimizer)
    status = MOI.get(model.infeasible_optimizer, MOI.PrimalStatus())
    if status != MOI.FEASIBLE_POINT
        @warn "Infeasible NLP problem terminated with primal status: $status"
    end
    return MOI.get(
        model.infeasible_optimizer,
        MOI.VariablePrimal(),
        model.infeasible_variables,
    )
end

# print objective gap information for iterative
function _print_gap(model::Optimizer, start)
    if model.log_level >= 1
        if model.num_iters_or_callbacks == 1 || model.log_level >= 2
            Printf.@printf(
                "\n%-5s | %-14s | %-14s | %-11s | %-11s\n",
                "Iter.",
                "Best feasible",
                "Best bound",
                "Rel. gap",
                "Time (s)",
            )
        end
        if model.objective_gap < 1000
            Printf.@printf(
                "%5d | %+14.6e | %+14.6e | %11.3e | %11.3e\n",
                model.num_iters_or_callbacks,
                model.objective_value,
                model.objective_bound,
                model.objective_gap,
                time() - start,
            )
        else
            obj_gap = isnan(model.objective_gap) ? "Inf" : ">1000"
            Printf.@printf(
                "%5d | %+14.6e | %+14.6e | %11s | %11.3e\n",
                model.num_iters_or_callbacks,
                model.objective_value,
                model.objective_bound,
                obj_gap,
                time() - start,
            )
        end
        flush(stdout)
        flush(stderr)
    end
    return
end

# utilities:

function _eval_func(values, func)
    return MOI.Utilities.eval_variables(vi -> values[vi.value], func)
end

# because Pavito only supports one bound on NLP constraints:
# TODO handle two bounds?
_has_upper(bound) = (bound != typemax(bound))
_has_lower(bound) = (bound != typemin(bound))

function _bound_set(model::Optimizer, i::Integer)
    bounds = model.nlp_block.constraint_bounds[i]
    return _bound_set(bounds.lower, bounds.upper)
end

function _bound_set(lb::T, ub::T) where {T}
    if _has_upper(ub)
        if _has_lower(lb)
            error(
                "An NLP constraint has lower bound $lb and upper bound $ub " *
                "but only one bound is supported.",
            )
        else
            return MOI.LessThan{Float64}(ub)
        end
    else
        if _has_lower(lb)
            return MOI.GreaterThan{Float64}(lb)
        else
            error("Pavito needs one bound per NLP constraint.")
        end
    end
end
