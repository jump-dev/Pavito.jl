#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

eval_func(values, func) = MOI.Utilities.eval_variables(vi -> values[vi.value], func)

function eval_objective(model::Optimizer, values)
    if model.nlp_block !== nothing && model.nlp_block.has_objective
        return MOI.eval_objective(model.nlp_block.evaluator, values)
    else
        return eval_func(values, model.objective)
    end
end

function eval_gradient(func::SQF, grad_f, values)
    fill!(grad_f, 0.0)
    for term in func.affine_terms
        grad_f[term.variable.value] += term.coefficient
    end
    for term in func.quadratic_terms
        grad_f[term.variable_1.value] += term.coefficient * values[term.variable_2.value]
        # If variables are the same, the coefficient is already multiplied by 2
        # 2 by definition of `SQF`.
        if term.variable_1 != term.variable_2
            grad_f[term.variable_2.value] += term.coefficient * values[term.variable_1.value]
        end
    end
end
function eval_objective_gradient(model::Optimizer, grad_f, values)
    if (model.nlp_block !== nothing && model.nlp_block.has_objective)
        MOI.eval_objective_gradient(model.nlp_block.evaluator, grad_f, values)
    else
        eval_gradient(model.objective, grad_f, values)
    end
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

    if isempty(model.int_indices)
        error("No variables of type integer or binary; call the continuous solver directly for pure continuous problems.")
    end

    if (model.nlp_block !== nothing && model.nlp_block.has_objective) || model.objective isa SQF
        if model.θ === nothing
            model.θ = MOI.add_variable(model.mip_optimizer)
            MOI.set(model.mip_optimizer, MOI.ObjectiveFunction{typeof(model.θ)}(), model.θ)
        end
    else
        if model.θ !== nothing
            MOI.delete(model.mip_optimizer, model.θ)
            model.θ = nothing
        end
    end

    start = time()
    nlp_time = 0.0
    mip_time = 0.0
    is_max = MOI.get(model.cont_optimizer, MOI.ObjectiveSense()) == MOI.MAX_SENSE
    comp = is_max ? (>) : (<)
    model.objective_value = is_max ? -Inf :  Inf
    model.objective_bound = is_max ?  Inf : -Inf

    if !model.silent && model.log_level > 0
        println("\nMINLP has a ", ((model.nlp_block !== nothing && model.nlp_block.has_objective) ? "nonlinear" : (model.objective isa SQF ? "quadratic" : "linear")), " objective, $(length(model.cont_variables)) variables ($(length(model.int_indices)) integer), $(model.nlp_block === nothing ? 0 : length(model.nlp_block.constraint_bounds)) nonlinear constraints, $(length(model.quad_less_than) + length(model.quad_greater_than)) quadratic constraints.")
        println("\nPavito started, using ", (model.mip_solver_drives ? "MIP-solver-driven" : "iterative"), " method...")
    end
    flush(stdout)

    if model.nlp_block === nothing
        jac_V = jac_IJ = nothing
    else
        MOI.initialize(model.nlp_block.evaluator, intersect([:Grad, :Jac, :Hess], MOI.features_available(model.nlp_block.evaluator)))
        jac_IJ = MOI.jacobian_structure(model.nlp_block.evaluator)
        jac_V = zeros(length(jac_IJ))
    end
    grad_f = zeros(length(model.cont_variables))

    # solve initial continuous relaxation NLP model
    start_nlp = time()
    MOI.optimize!(model.cont_optimizer)
    nlp_time += time() - start_nlp
    ini_nlp_status = MOI.get(model.cont_optimizer, MOI.TerminationStatus())
    cont_solution = nothing
    if ini_nlp_status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL]
        cont_solution = MOI.get(model.cont_optimizer, MOI.VariablePrimal(), model.cont_variables)
        cont_obj = MOI.get(model.cont_optimizer, MOI.ObjectiveValue())
        add_cuts(model, cont_solution, jac_IJ, jac_V, grad_f, is_max)
        # We cannot update `model.objective_value` or `model.incumbent` as this
        # it may not be feasible, it is only feasible for the relaxed problem.
    elseif ini_nlp_status == MOI.DUAL_INFEASIBLE
        # This may not mean that the problem is unbounded as the integrality constraints
        # may make it bounded so we continue.
        @warn "initial NLP relaxation unbounded"
    elseif ini_nlp_status == MOI.NORM_LIMIT
        # Ipopt usually ends with `Diverging_Iterates` for unbounded problems.
        # This gets converted to `MOI.NORM_LIMIT`.
        @warn "initial NLP relaxation terminated with status `NORM_LIMIT` which usually means that the problem is unbounded"
    else
        if ini_nlp_status in [MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE, MOI.ALMOST_INFEASIBLE]
            # If the relaxation is infeasible then the original problem is infeasible as well
            @warn "initial NLP relaxation infeasible"
            model.status = ini_nlp_status
        else
            @warn "NLP solver failure: initial NLP relaxation terminated with status $ini_nlp_status"
            model.status = ini_nlp_status
        end
        return
    end
    flush(stdout)

    if cont_solution !== nothing && MOI.supports(model.mip_optimizer, MOI.VariablePrimalStart(), MOI.VariableIndex) && all(isfinite, cont_solution)
        MOI.set(model.mip_optimizer, MOI.VariablePrimalStart(), model.mip_variables, cont_solution)

        if model.θ !== nothing
            MOI.set(model.mip_optimizer, MOI.VariablePrimalStart(), model.θ, cont_obj)
        end
    end

    # start main OA algorithm

    if model.mip_solver_drives
        # MIP-solver-driven method
        cache_contsol = Dict{Vector{Float64},Vector{Float64}}()

        function lazy_callback(cb)
            model.num_iters_or_callbacks += 1

            # if integer assignment has been seen before, use cached point
            model.mip_solution = MOI.get(model.mip_optimizer, MOI.CallbackVariablePrimal(cb), model.mip_variables)
            if any(i -> abs(model.mip_solution[i] - round(model.mip_solution[i])) > 1e-5, model.int_indices)
                # The solution is integer-infeasible, the `1e-5` was chosen for backward
                # compatibility as this filter used to be done by JuMP:
                # https://github.com/JuliaOpt/GLPKMathProgInterface.jl/blob/26f755770f8b70a81ade83cd1e6d85fa8c8254c9/src/GLPKInterfaceMIP.jl#L103
                # For more details, see
                # https://github.com/jump-dev/GLPK.jl/issues/146
                # and
                # https://github.com/jump-dev/MathOptInterface.jl/pull/1172
                @warn "Solver called a lazy callback with an integer-infeasible solution"
                return
            end
            round_mipsol = round.(model.mip_solution)
            if haskey(cache_contsol, round_mipsol)
                # retrieve existing solution
                cont_solution = cache_contsol[round_mipsol]
            else
                # try to solve new subproblem, update incumbent solution if feasible
                start_nlp = time()
                cont_solution = solve_subproblem(model, comp)
                nlp_time += time() - start_nlp
                cache_contsol[round_mipsol] = cont_solution
            end

            # add gradient cuts to MIP model from NLP solution
            if all(isfinite, cont_solution)
                add_cuts(model, cont_solution, jac_IJ, jac_V, grad_f, is_max, cb)
            else
                @warn "no cuts could be added, Pavito should be terminated"
                # TODO terminate the solver once there is a solver-independent callback for that in MOI
                return
            end
        end
        MOI.set(model.mip_optimizer, MOI.LazyConstraintCallback(), lazy_callback)

        function heuristic_callback(cb)
            # if have a new best feasible solution since last heuristic solution added, set MIP solution to the new best feasible solution
            if model.new_incumb
                if model.θ === nothing
                    MOI.submit(model.mip_optimizer, MOI.HeuristicSolution(cb), model.mip_variables, model.incumbent)
                else
                    MOI.submit(model.mip_optimizer, MOI.HeuristicSolution(cb), [model.mip_variables; model.θ], [model.incumbent; model.objective_value])
                end

                model.new_incumb = false
            end
        end

        MOI.set(model.mip_optimizer, MOI.HeuristicCallback(), heuristic_callback)

        if isfinite(model.timeout)
            MOI.set(model.mip_optimizer, MOI.TimeLimitSec(), max(1.0, model.timeout - (time() - start)))
        end
        MOI.optimize!(model.mip_optimizer)
        model.status = MOI.get(model.mip_optimizer, MOI.TerminationStatus())
        model.objective_bound = MOI.get(model.mip_optimizer, MOI.ObjectiveBound())
    else
        # iterative method
        prev_mip_solution = fill(NaN, length(model.mip_variables))
        while (time() - start) < model.timeout
            model.num_iters_or_callbacks += 1

            # set remaining time limit on MIP solver
            if isfinite(model.timeout)
                MOI.set(model.mip_optimizer, MOI.TimeLimitSec(), max(1.0, model.timeout - (time() - start)))
            end

            # solve MIP model
            start_mip = time()
            MOI.optimize!(model.mip_optimizer)
            mip_status = MOI.get(model.mip_optimizer, MOI.TerminationStatus())
            mip_time += time() - start_mip

            # finish if MIP was infeasible or if problematic status
            if (mip_status == MOI.INFEASIBLE) || (mip_status == MOI.INFEASIBLE_OR_UNBOUNDED)
                model.status = MOI.INFEASIBLE
                break
            elseif (mip_status != MOI.OPTIMAL) && (mip_status != MOI.ALMOST_OPTIMAL)
                @warn "MIP solver status was $mip_status, terminating Pavito"
                model.status = mip_status
                break
            end
            model.mip_solution = MOI.get(model.mip_optimizer, MOI.VariablePrimal(), model.mip_variables)

            # update best bound from MIP bound
            mip_obj_bound = MOI.get(model.mip_optimizer, MOI.ObjectiveBound())
            if isfinite(mip_obj_bound) && comp(model.objective_bound, mip_obj_bound)
                model.objective_bound = mip_obj_bound
            end

            update_gap(model, is_max)
            printgap(model, start)
            check_progress(model, prev_mip_solution) && break

            # try to solve new subproblem, update incumbent solution if feasible
            start_nlp = time()
            cont_solution = solve_subproblem(model, comp)
            nlp_time += time() - start_nlp

            update_gap(model, is_max)
            check_progress(model, prev_mip_solution) && break

            # add gradient cuts to MIP model from NLP solution
            if all(isfinite, cont_solution)
                add_cuts(model, cont_solution, jac_IJ, jac_V, grad_f, is_max)
            else
                @warn "no cuts could be added, terminating Pavito"
                break
            end

            # TODO warmstart MIP from upper bound as MPB's version

            prev_mip_solution = model.mip_solution
            flush(stdout)
        end
    end
    flush(stdout)

    # finish
    model.total_time = time() - start
    if !model.silent && model.log_level > 0
        println("\nPavito finished...\n")
        @printf "Status           %13s\n" model.status
        @printf "Objective value  %13.5f\n" model.objective_value
        @printf "Objective bound  %13.5f\n" model.objective_bound
        @printf "Objective gap    %13.5f\n" model.objective_gap
        if !model.mip_solver_drives
            @printf "Iterations       %13d\n" model.num_iters_or_callbacks
        else
            @printf "Callbacks        %13d\n" model.num_iters_or_callbacks
        end
        @printf "Total time       %13.5f sec\n" model.total_time
        @printf "MIP total time   %13.5f sec\n" mip_time
        @printf "NLP total time   %13.5f sec\n" nlp_time
        println()
    end
    flush(stdout)

    return nothing
end

function update_gap(model::Optimizer, is_max::Bool)
    # update gap if best bound and best objective are finite
    if isfinite(model.objective_value) && isfinite(model.objective_bound)
        model.objective_gap = (model.objective_value - model.objective_bound) / (abs(model.objective_value) + 1e-5)
        if is_max
            model.objective_gap = -model.objective_gap
        end
    end
end

function check_progress(model::Optimizer, prev_mip_solution)
    # finish if optimal or cycling integer solutions
    int_ind = collect(model.int_indices)
    if model.objective_gap <= model.rel_gap
        model.status = MOI.OPTIMAL
        return true
    elseif round.(prev_mip_solution[int_ind]) == round.(model.mip_solution[int_ind])
        @warn "mixed-integer cycling detected ($(round.(prev_mip_solution[int_ind])) == $(round.(model.mip_solution[int_ind]))), terminating Pavito"
        if isfinite(model.objective_gap)
            model.status = MOI.ALMOST_OPTIMAL
        else
            model.status = MOI.OTHER_ERROR
        end
        return true
    end
    return false
end

function fix_int_vars(optimizer::MOI.ModelLike, vars, mip_solution, int_indices)
    for i in int_indices
        vi = vars[i]
        idx = vi.value
        ci = MOI.ConstraintIndex{MOI.VariableIndex, MOI.LessThan{Float64}}(idx)
        MOI.is_valid(optimizer, ci) && MOI.delete(optimizer, ci)
        ci = MOI.ConstraintIndex{MOI.VariableIndex, MOI.GreaterThan{Float64}}(idx)
        MOI.is_valid(optimizer, ci) && MOI.delete(optimizer, ci)
        ci = MOI.ConstraintIndex{MOI.VariableIndex, MOI.EqualTo{Float64}}(idx)
        set = MOI.EqualTo(mip_solution[i])

        if MOI.is_valid(optimizer, ci)
            MOI.set(optimizer, MOI.ConstraintSet(), ci, set)
        else
            MOI.add_constraint(optimizer, vi, set)
        end
    end
end

# solve NLP subproblem defined by integer assignment
function solve_subproblem(model::Optimizer, comp::Function)
    fix_int_vars(model.cont_optimizer, model.cont_variables, model.mip_solution, model.int_indices)
    MOI.optimize!(model.cont_optimizer)
    if MOI.get(model.cont_optimizer, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        # subproblem is feasible, check if solution is new incumbent
        nlp_objective_value = MOI.get(model.cont_optimizer, MOI.ObjectiveValue())
        nlp_solution = MOI.get(model.cont_optimizer, MOI.VariablePrimal(), model.cont_variables)
        if comp(nlp_objective_value, model.objective_value)
            model.objective_value = nlp_objective_value
            copyto!(model.incumbent, nlp_solution)
            model.new_incumb = true
        end

        return nlp_solution
    end

    # assume subproblem is infeasible, so solve infeasible recovery NLP subproblem
    if (model.nlp_block !== nothing && !isempty(model.nlp_block.constraint_bounds) && model.nl_slack_variables === nothing) ||
        (!isempty(model.quad_less_than) && model.quad_less_than_slack_variables === nothing) ||
        (!isempty(model.quad_greater_than) && model.quad_greater_than_slack_variables === nothing)

        if model.nl_slack_variables !== nothing
            obj = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x) for x in model.nl_slack_variables], 0.0)
        else
            obj = MOI.ScalarAffineFunction(Array{MOI.ScalarAffineTerm{Float64}}([]), 0.0)
        end

        _add_to_obj(vi::MOI.VariableIndex) = push!(obj.terms, MOI.ScalarAffineTerm(1.0, vi))

        if model.nlp_block !== nothing && !isempty(model.nlp_block.constraint_bounds)
            bounds = copy(model.nlp_block.constraint_bounds)
            model.infeasible_evaluator = InfeasibleNLPEvaluator(model.nlp_block.evaluator, length(model.infeasible_variables), falses(length(model.nlp_block.constraint_bounds)))
            model.nl_slack_variables = MOI.add_variables(_infeasible(model), length(model.nlp_block.constraint_bounds))
            for i in eachindex(model.nlp_block.constraint_bounds)
                _add_to_obj(model.nl_slack_variables[i])
                push!(bounds, MOI.NLPBoundsPair(0.0, Inf))
                set = _bound_set(model, i)
                if set isa MOI.LessThan
                    model.infeasible_evaluator.minus[i] = true
                end
            end
            MOI.set(_infeasible(model), MOI.NLPBlock(), MOI.NLPBlockData(bounds, model.infeasible_evaluator, false))
        end
        # We need to add quadratic variables afterwards as `InfeasibleNLPEvaluator` assumes
        # that the original variables are directly followed by the NL slack variables.
        if !isempty(model.quad_less_than)
            model.quad_less_than_slack_variables = MOI.add_variables(_infeasible(model), length(model.quad_less_than))
            model.quad_less_than_infeasible_con = map(eachindex(model.quad_less_than)) do i
                func, set = model.quad_less_than[i]
                MOI.add_constraint(_infeasible(model), MOI.Utilities.operate(-, Float64, func, model.quad_less_than_slack_variables[i]), set)
            end
            for vi in model.quad_less_than_slack_variables
                _add_to_obj(vi)
            end
        end
        if !isempty(model.quad_greater_than)
            model.quad_greater_than_slack_variables = MOI.add_variables(_infeasible(model), length(model.quad_greater_than))
            model.quad_greater_than_infeasible_con = map(eachindex(model.quad_greater_than)) do i
                func, set = model.quad_greater_than[i]
                MOI.add_constraint(_infeasible(model), MOI.Utilities.operate(+, Float64, func, model.quad_greater_than_slack_variables[i]), set)
            end
            for vi in model.quad_greater_than_slack_variables
                _add_to_obj(vi)
            end
        end
        MOI.set(_infeasible(model), MOI.ObjectiveFunction{typeof(obj)}(), obj)
    end

    fix_int_vars(model.infeasible_optimizer, model.infeasible_variables, model.mip_solution, model.int_indices)
    MOI.set(_infeasible(model), MOI.VariablePrimalStart(), model.infeasible_variables, model.mip_solution)

    if model.nlp_block !== nothing && !isempty(model.nlp_block.constraint_bounds)
        fill!(model.infeasible_evaluator.minus, false)
        g = zeros(length(model.nlp_block.constraint_bounds))
        MOI.eval_constraint(model.nlp_block.evaluator, g, model.mip_solution)

        for i in eachindex(model.nlp_block.constraint_bounds)
            bounds = model.nlp_block.constraint_bounds[i]
            val = model.infeasible_evaluator.minus[i] ? (g[i] - bounds.upper) : (bounds.lower - g[i])
            # sign of the slack changes if the constraint direction changes
            MOI.set(_infeasible(model), MOI.VariablePrimalStart(), model.nl_slack_variables[i], max(0.0, val))
        end
    end

    for i in eachindex(model.quad_less_than)
        val = eval_func(model.mip_solution, model.quad_less_than[i][1]) - model.quad_less_than[i][2].upper
        MOI.set(_infeasible(model), MOI.VariablePrimalStart(), model.quad_less_than_slack_variables[i], max(0.0, val))
    end

    for i in eachindex(model.quad_greater_than)
        val = model.quad_greater_than[i][2].lower - eval_func(model.mip_solution, model.quad_greater_than[i][1])
        MOI.set(_infeasible(model), MOI.VariablePrimalStart(), model.quad_greater_than_slack_variables[i], max(0.0, val))
    end

    # solve
    MOI.optimize!(model.infeasible_optimizer)
    status = MOI.get(model.infeasible_optimizer, MOI.PrimalStatus())
    if status != MOI.FEASIBLE_POINT
        @warn "Infeasible NLP problem terminated with primal status: $status. This cannot be as this NLP problem was feasible by design."
    end

    return MOI.get(model.infeasible_optimizer, MOI.VariablePrimal(), model.infeasible_variables)
end

# By convexity of g(x), we know that g(x) >= g(c) + g'(c) * (x - c)
# Given a constraint ub >= g(x), we rewrite it as
# ub - g(x) + g'(c) * c >= g'(x) * x
# If the constraint is `g(x) <= lb` then we assume that `g(x)` is concave instead and,
# it is rewritten as
# lb - g(x) + g'(c) * c <= g'(x) * x
# If the constraint is `lb <= g(x) <= ub` or `g(x) == lb == ub` then we assume that
# `g(x)` is linear.

function add_cut(model::Optimizer, cont_solution, gc, dgc_idx, dgc_nzv, set, callback_data)
    Δ = 0.0
    for i in eachindex(dgc_idx)
        Δ += dgc_nzv[i] * cont_solution[dgc_idx[i]]
    end

    func = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(
            dgc_nzv[i],
            model.mip_variables[dgc_idx[i]])
         for i in eachindex(dgc_idx)],
        0.0
    )
    set = MOI.Utilities.shift_constant(set, Δ - gc)
    MOI.Utilities.canonicalize!(func)
    if !isempty(func.terms)
        # TODO should we check that the inequality is not trivially infeasible ?
        if callback_data === nothing
            MOI.add_constraint(model.mip_optimizer, func, set)
        else
            _lazy_constraint(model, callback_data, func, set, model.mip_solution)
        end
    end
end

function add_quad_cuts(model::Optimizer, cont_solution, cons, callback_data)
    for (func, set) in cons
        gc = eval_func(cont_solution, func)
        dgc_idx = Int64[]
        dgc_nzv = Float64[]
        for term in func.affine_terms
            push!(dgc_idx, term.variable.value)
            push!(dgc_nzv, term.coefficient)
        end
        for term in func.quadratic_terms
            push!(dgc_idx, term.variable_1.value)
            push!(dgc_nzv, term.coefficient * cont_solution[term.variable_2.value])
            # If variables are the same, the coefficient is already multiplied by 2
            # 2 by definition of `SQF`.
            if term.variable_1 != term.variable_2
                push!(dgc_idx, term.variable_2.value)
                push!(dgc_nzv, term.coefficient * cont_solution[term.variable_1.value])
            end
        end
        add_cut(model, cont_solution, gc, dgc_idx, dgc_nzv, set, callback_data)
    end
end

function add_cuts(model::Optimizer, cont_solution, jac_IJ, jac_V, grad_f, is_max, callback_data = nothing)
    if model.nlp_block !== nothing
        # eval g and jac_g at MIP solution
        num_constrs = length(model.nlp_block.constraint_bounds)
        g = zeros(num_constrs)
        MOI.eval_constraint(model.nlp_block.evaluator, g, cont_solution)
        MOI.eval_constraint_jacobian(model.nlp_block.evaluator, jac_V, cont_solution)

        # create rows corresponding to constraints in sparse format
        varidx_new = [zeros(Int, 0) for i in 1:num_constrs]
        coef_new = [zeros(0) for i in 1:num_constrs]

        for k in eachindex(jac_IJ)
            row, col = jac_IJ[k]
            push!(varidx_new[row], col)
            push!(coef_new[row], jac_V[k])
        end

        # create constraint cuts
        for i in 1:num_constrs
            # create supporting hyperplane
            set = _bound_set(model, i)
            add_cut(model, cont_solution, g[i], varidx_new[i], coef_new[i], set, callback_data)
        end
    end
    add_quad_cuts(model, cont_solution, model.quad_less_than, callback_data)
    add_quad_cuts(model, cont_solution, model.quad_greater_than, callback_data)

    # Given an objective `Min θ = f(x)` with a convex `f(x)`,
    # we have
    # -f(x) + f'(c) * c >= f'(x) * x - θ
    # If the objective is `Max θ = f(x)` then we assume that `g(x)` is concave instead and we have
    # -f(x) + f'(c) * c <= f'(x) * x - θ

    # create obj cut
    if (model.nlp_block !== nothing && model.nlp_block.has_objective) || model.objective isa SQF
        f = eval_objective(model, cont_solution)
        eval_objective_gradient(model, grad_f, cont_solution)
        constant = -f
        func = MOI.Utilities.operate(-, Float64, model.θ)
        for j in eachindex(grad_f)
            if !iszero(grad_f[j])
                constant += grad_f[j] * cont_solution[j]
                push!(func.terms, MOI.ScalarAffineTerm(grad_f[j], model.mip_variables[j]))
            end
        end
        set = is_max ? MOI.GreaterThan(constant) : MOI.LessThan(constant)
        if callback_data === nothing
            MOI.add_constraint(model.mip_optimizer, func, set)
        else
            θ = MOI.get(model.mip_optimizer, MOI.CallbackVariablePrimal(callback_data), model.θ)
            _lazy_constraint(model, callback_data, func, set, [model.mip_solution; θ])
        end
    end

    return
end

# `isapprox(0.0, 1e-16)` is false but `_is_approx(0.0, 1e-16)` is true.
_is_approx(x, y) = isapprox(x, y, atol=Base.rtoldefault(Float64))
approx_in(value, set::MOI.EqualTo)     = _is_approx(value, MOI.constant(set))
approx_in(value, set::MOI.LessThan)    = _is_approx(value, MOI.constant(set)) || value < MOI.constant(set)
approx_in(value, set::MOI.GreaterThan) = _is_approx(value, MOI.constant(set)) || value > MOI.constant(set)

function _lazy_constraint(model, callback_data, func, set, mip_solution)
    # GLPK does not check whether the new cut is redundant or not
    # so we should filter it out ourself:
    # See https://github.com/jump-dev/GLPK.jl/issues/153
    if approx_in(eval_func(mip_solution, func), set)
        return
    end
    MOI.submit(model.mip_optimizer, MOI.LazyConstraint(callback_data), func, set)
end

# print objective gap information for iterative
function printgap(model::Optimizer, start)
    if !model.silent && model.log_level >= 1
        if (model.num_iters_or_callbacks == 1) || (model.log_level >= 2)
            @printf "\n%-5s | %-14s | %-14s | %-11s | %-11s\n" "Iter." "Best feasible" "Best bound" "Rel. gap" "Time (s)"
        end
        if model.objective_gap < 1000
            @printf "%5d | %+14.6e | %+14.6e | %11.3e | %11.3e\n" model.num_iters_or_callbacks model.objective_value model.objective_bound model.objective_gap (time() - start)
        else
            @printf "%5d | %+14.6e | %+14.6e | %11s | %11.3e\n" model.num_iters_or_callbacks model.objective_value model.objective_bound (isnan(model.objective_gap) ? "Inf" : ">1000") (time() - start)
        end
        flush(stdout)
        flush(stderr)
    end
    return
end

# Taken from MatrixOptInterface.jl
_has_upper(bound) = bound != typemax(bound)
_has_lower(bound) = bound != typemin(bound)
function _bound_set(model::Optimizer, i::Integer)
    bounds = model.nlp_block.constraint_bounds[i]
    return _bound_set(bounds.lower, bounds.upper)
end
function _bound_set(lb::T, ub::T) where T
    if _has_upper(ub)
        if _has_lower(lb)
            error("Only one bound per constraint supported by Pavito for NLP constraints but one NLP constraint has lower bound $lb and upper bound $ub.")
        else
            return MOI.LessThan(ub)
        end
    else
        if _has_lower(lb)
            return MOI.GreaterThan(lb)
        else
            error("At least one bound per constraint needed by Pavito for NLP constraints.")
        end
    end
end
