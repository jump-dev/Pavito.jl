#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

function MOI.optimize!(model::Optimizer)
    start = time()
    nlp_time = 0.0
    mip_time = 0.0
    is_max = MOI.get(model.cont_optimizer, MOI.ObjectiveSense()) == MOI.MAX_SENSE
    comp = is_max ? (>) : (<)

    if model.log_level > 0
        println("\nMINLP has a ", (model.nlp_block.has_objective ? "nonlinear" : (model.quadratic_objective ? "quadratic" : "linear")), " objective, $(length(model.cont_variables)) variables (?? integer), ?? constraints ($(length(model.nlp_block.constraint_bounds)) nonlinear)")
        println("\nPavito started, using ", (model.mip_solver_drives ? "MIP-solver-driven" : "iterative"), " method...")
    end
    flush(stdout)

    MOI.initialize(model.nlp_block.evaluator, intersect([:Grad, :Jac, :Hess], MOI.features_available(model.nlp_block.evaluator)))
    jac_IJ = MOI.jacobian_structure(model.nlp_block.evaluator)
    jac_V = zeros(length(jac_IJ))
    grad_f = zeros(length(model.cont_variables))

    # solve initial continuous relaxation NLP model
    start_nlp = time()
    MOI.optimize!(model.cont_optimizer)
    nlp_time += time() - start_nlp
    ini_nlp_status = MOI.get(model.cont_optimizer, MOI.TerminationStatus())
    if ini_nlp_status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL]
        cont_solution = MOI.get(model.cont_optimizer, MOI.VariablePrimal(), model.cont_variables)
        add_cuts(model, cont_solution, jac_IJ, jac_V, grad_f, is_max)
    else
        if ini_nlp_status in [MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE, MOI.ALMOST_INFEASIBLE]
            # If the relaxation is infeasible then the original problem is infeasible as well
            @warn "initial NLP relaxation infeasible"
            model.status = ini_nlp_status
        elseif ini_nlp_status == MOI.DUAL_INFEASIBLE
            @warn "initial NLP relaxation unbounded"
            model.status = ini_nlp_status
        else
            @warn "NLP solver failure: initial NLP relaxation terminated with status $ini_nlp_status"
            model.status = ini_nlp_status
        end
        return
    end
    flush(stdout)

    # TODO if have warmstart, use objective_value and warmstart MIP model as in MPB version of Pavito
    # start main OA algorithm

    if model.mip_solver_drives
        # MIP-solver-driven method
        cache_contsol = Dict{Vector{Float64},Vector{Float64}}()

        function lazy_callback(cb)
            model.num_iters_or_callbacks += 1

            # if integer assignment has been seen before, use cached point
            mip_solution = MOI.get(model.mip_optimizer, MOI.VariablePrimal(), model.mip_variables)
            round_mipsol = round.(mip_solution)
            if haskey(cache_contsol, round_mipsol)
                # retrieve existing solution
                cont_solution = cache_contsol[round_mipsol]
            else
                # try to solve new subproblem, update incumbent solution if feasible
                start_nlp = time()
                cont_solution = solve_subproblem(model, mip_solution, comp)
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
        MOI.submit(model.mip_optimizer, MOI.LazyConstraintCallback(), lazy_callback)

#        function heuristic_callback(cb)
#            # if have a new best feasible solution since last heuristic solution added, set MIP solution to the new best feasible solution
#            if model.new_incumb
#                # TODO we should give the value of θ as well
#                MOI.submit(model.mip_optimizer, MOI.HeuristicSolution(cb), model.mip_variables, model.incumbent)
#                model.new_incumb = false
#            end
#        end
#        MOI.submit(model.mip_optimizer, MOI.HeuristicCallback(), heuristic_callback)

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
            mip_solution = MOI.get(model.mip_optimizer, MOI.VariablePrimal(), model.mip_variables)

            update_gap(model, is_max)
            printgap(model, start)
            check_progress(model, prev_mip_solution, mip_solution) && break

            # try to solve new subproblem, update incumbent solution if feasible
            start_nlp = time()
            cont_solution = solve_subproblem(model, mip_solution, comp)
            nlp_time += time() - start_nlp

            update_gap(model, is_max)
            check_progress(model, prev_mip_solution, mip_solution) && break

            # add gradient cuts to MIP model from NLP solution
            if all(isfinite, cont_solution)
                add_cuts(model, cont_solution, jac_IJ, jac_V, grad_f, is_max)
            else
                @warn "no cuts could be added, terminating Pavito"
                break
            end

            # TODO warmstart MIP from upper bound as MPB's version

            prev_mip_solution = mip_solution
            flush(stdout)
        end
    end
    flush(stdout)

    # finish
    model.total_time = time() - start
    if model.log_level > 0
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
        model.objective_gap = model.objective_value - model.objective_bound / (abs(model.objective_value) + 1e-5)
        if is_max
            model.objective_gap = -model.objective_gap
        end
    end
end

function check_progress(model::Optimizer, prev_mip_solution, mip_solution)
    # finish if optimal or cycling integer solutions
    int_ind = collect(model.int_indices)
    if model.objective_gap <= model.rel_gap
        model.status = MOI.OPTIMAL
        return true
    elseif round.(prev_mip_solution[int_ind]) == round.(mip_solution[int_ind])
        @warn "mixed-integer cycling detected, terminating Pavito"
        if isfinite(model.objective_gap)
            model.status = MOI.ALMOST_OPTIMAL
        else
            model.status = MOI.OTHER_ERROR
        end
        return true
    end
    return false
end

# solve NLP subproblem defined by integer assignment
function solve_subproblem(model::Optimizer, mip_solution, comp::Function)
    for i in model.int_indices
        ci = MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}(i)
        if MOI.is_valid(model.cont_optimizer, ci)
            MOI.delete(model.cont_optimizer, ci)
        end
        ci = MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}(i)
        if MOI.is_valid(model.cont_optimizer, ci)
            MOI.delete(model.cont_optimizer, ci)
        end
        ci = MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}}(i)
        set = MOI.EqualTo(mip_solution[i])
        if MOI.is_valid(model.cont_optimizer, ci)
            MOI.set(model.cont_optimizer, MOI.ConstraintSet(), ci, set)
        else
            func = MOI.SingleVariable(MOI.VariableIndex(i))
            MOI.add_constraint(model.cont_optimizer, func, set)
        end
    end
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

    error("Infeasible NLP not implemented yet.")
end

function add_cuts(model::Optimizer, cont_solution, jac_IJ, jac_V, grad_f, is_max, callback_data = nothing)
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

    # By convexity of g(x), we know that g(x) >= g(c) + g'(c) * (x - c)
    # Given a constraint ub >= g(x), we rewrite it as
    # ub - g(x) + g'(c) * c >= g'(x) * x
    # If the constraint is `g(x) <= lb` then we assume that `g(x)` is concave instead and,
    # it is rewritten as
    # lb - g(x) + g'(c) * c <= g'(x) * x
    # If the constraint is `lb <= g(x) <= ub` or `g(x) == lb == ub` then we assume that
    # `g(x)` is linear.

    # create constraint cuts
    for i in 1:num_constrs
        # create supporting hyperplane
        lb = model.nlp_block.constraint_bounds[i].lower - g[i]
        ub = model.nlp_block.constraint_bounds[i].upper - g[i]

        for j in eachindex(varidx_new[i])
            Δ = coef_new[i][j] * cont_solution[varidx_new[i][j]]
            lb += Δ
            ub += Δ
        end

        func = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(
                coef_new[i][j],
                model.cont_variables[varidx_new[i][j]])
             for j in eachindex(varidx_new[i])],
            0.0
        )
        set = _bound_set(lb, ub)
        if callback_data === nothing
            MOI.add_constraint(model.mip_optimizer, func, set)
        else
            MOI.submit(model.mip_optimizer, MOI.LazyConstraint(callback_data), func, set)
        end
    end

    # Given an objective `Min θ = f(x)` with a convex `f(x)`,
    # we have
    # -f(x) + f'(c) * c >= f'(x) * x - θ
    # If the objective is `Max θ = f(x)` then we assume that `g(x)` is concave instead and we have
    # -f(x) + f'(c) * c <= f'(x) * x - θ

    # create obj cut
    if model.nlp_block.has_objective
        f = MOI.eval_objective(model.nlp_block.evaluator, cont_solution)
        MOI.eval_objective_gradient(model.nlp_block.evaluator, grad_f, cont_solution)
        constant = -f
        func = MOI.Utilities.operate(-, Float64, MOI.SingleVariable(model.θ))
        for j in eachindex(grad_f)
            if !iszero(grad_f[j])
                constant += grad_f[j] * cont_solution[j]
                push!(func, MOI.ScalarAffineTerm(grad_f[j], model.cont_variables[j]))
            end
        end
        set = is_max ? MOI.GreaterThan(constant) : MOI.LessThan(constant)
        if callback_data === nothing
            MOI.add_constraint(model.mip_optimizer, func, set)
        else
            MOI.submit(model.mip_optimizer, MOI.LazyConstraint(callback_data), func, set)
        end
    elseif model.quadratic_objective
        error("TODO")
    end

    return
end

# print objective gap information for iterative
function printgap(m::Optimizer, start)
    if m.log_level >= 1
        if (m.num_iters_or_callbacks == 1) || (m.log_level >= 2)
            @printf "\n%-5s | %-14s | %-14s | %-11s | %-11s\n" "Iter." "Best feasible" "Best bound" "Rel. gap" "Time (s)"
        end
        if m.objgap < 1000
            @printf "%5d | %+14.6e | %+14.6e | %11.3e | %11.3e\n" m.num_iters_or_callbacks m.objective_value m.objective_bound m.objective_gap (time() - start)
        else
            @printf "%5d | %+14.6e | %+14.6e | %11s | %11.3e\n" m.num_iters_or_callbacks m.objective_value m.objective_bound (isnan(m.objective_gap) ? "Inf" : ">1000") (time() - start)
        end
        flush(stdout)
        flush(stderr)
    end
    return
end

# Taken from MatrixOptInterface.jl
_no_upper(bound) = bound != typemax(bound)
_no_lower(bound) = bound != typemin(bound)
function _bound_set(lb::T, ub::T) where T
    if _no_upper(ub)
        if _no_lower(lb)
            if ub == lb
                return MOI.EqualTo(lb)
            else
                return MOI.Interval(lb, ub)
            end
        else
            return MOI.LessThan(ub)
        end
    else
        if _no_lower(lb)
            return MOI.GreaterThan(lb)
        else
            error("Both bounds are infinite: $lb, $ub")
        end
    end
end
