#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 gradient cut utilities
=========================================================#

# by convexity of g(x), we know that g(x) >= g(c) + g'(c) * (x - c)
# given a constraint ub >= g(x), we rewrite it as:
# ub - g(x) + g'(c) * c >= g'(x) * x
# if the constraint is `g(x) <= lb`, we assume `g(x)` is concave, so:
# lb - g(x) + g'(c) * c <= g'(x) * x
# if the constraint is `lb <= g(x) <= ub` or `g(x) == lb == ub`, we assume
# `g(x)` is linear

function add_cut(
    model::Optimizer,
    cont_solution,
    gc,
    dgc_idx,
    dgc_nzv,
    set,
    callback_data,
)
    Δ = 0.0
    for i in eachindex(dgc_idx)
        Δ += dgc_nzv[i] * cont_solution[dgc_idx[i]]
    end

    safs = [
        MOI.ScalarAffineTerm(dgc_nzv[i], model.mip_variables[dgc_idx[i]])
        for i in eachindex(dgc_idx)
    ]
    func = MOI.ScalarAffineFunction(safs, 0.0)
    MOI.Utilities.canonicalize!(func)
    set = MOI.Utilities.shift_constant(set, Δ - gc)

    if !isempty(func.terms)
        if isnothing(callback_data)
            MOI.add_constraint(model.mip_optimizer, func, set)
        else
            _add_lazy_constraint(
                model,
                callback_data,
                func,
                set,
                model.mip_solution,
            )
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
            push!(
                dgc_nzv,
                term.coefficient * cont_solution[term.variable_2.value],
            )
            # if variables are the same, the coefficient is already multiplied by
            # 2 by definition of `MOI.ScalarQuadraticFunction{Float64}`
            if term.variable_1 != term.variable_2
                push!(dgc_idx, term.variable_2.value)
                push!(
                    dgc_nzv,
                    term.coefficient * cont_solution[term.variable_1.value],
                )
            end
        end

        add_cut(model, cont_solution, gc, dgc_idx, dgc_nzv, set, callback_data)
    end
end

function add_cuts(
    model::Optimizer,
    cont_solution,
    jac_IJ,
    jac_V,
    grad_f,
    is_max,
    callback_data = nothing,
)
    if !isnothing(model.nlp_block)
        # eval g and jac_g at MIP solution
        num_constrs = length(model.nlp_block.constraint_bounds)
        g = zeros(num_constrs)
        MOI.eval_constraint(model.nlp_block.evaluator, g, cont_solution)
        MOI.eval_constraint_jacobian(
            model.nlp_block.evaluator,
            jac_V,
            cont_solution,
        )

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
            add_cut(
                model,
                cont_solution,
                g[i],
                varidx_new[i],
                coef_new[i],
                set,
                callback_data,
            )
        end
    end
    add_quad_cuts(model, cont_solution, model.quad_LT, callback_data)
    add_quad_cuts(model, cont_solution, model.quad_GT, callback_data)

    # given an objective `Min nlp_obj_var = f(x)` with a convex `f(x)`:
    # -f(x) + f'(c) * c >= f'(x) * x - nlp_obj_var
    # if the objective is `Max`, we assume `g(x)` is concave, so:
    # -f(x) + f'(c) * c <= f'(x) * x - nlp_obj_var

    # create objective cut
    if (!isnothing(model.nlp_block) && model.nlp_block.has_objective) ||
       model.objective isa MOI.ScalarQuadraticFunction{Float64}
        f = eval_objective(model, cont_solution)
        eval_objective_gradient(model, grad_f, cont_solution)

        constant = -f
        func = MOI.Utilities.operate(-, Float64, model.nlp_obj_var)
        for j in eachindex(grad_f)
            if !iszero(grad_f[j])
                constant += grad_f[j] * cont_solution[j]
                push!(
                    func.terms,
                    MOI.ScalarAffineTerm(grad_f[j], model.mip_variables[j]),
                )
            end
        end
        set = (
            is_max ? MOI.GreaterThan{Float64}(constant) :
            MOI.LessThan{Float64}(constant)
        )

        if isnothing(callback_data)
            MOI.add_constraint(model.mip_optimizer, func, set)
        else
            nlp_obj_var = MOI.get(
                model.mip_optimizer,
                MOI.CallbackVariablePrimal(callback_data),
                model.nlp_obj_var,
            )
            _add_lazy_constraint(
                model,
                callback_data,
                func,
                set,
                vcat(model.mip_solution, nlp_obj_var),
            )
        end
    end

    return
end

function _add_lazy_constraint(model, callback_data, func, set, mip_solution)
    # GLPK does not check whether the new cut is redundant, so we filter it out
    # see https://github.com/jump-dev/GLPK.jl/issues/153
    if !approx_in(eval_func(mip_solution, func), set)
        MOI.submit(
            model.mip_optimizer,
            MOI.LazyConstraint(callback_data),
            func,
            set,
        )
    end
    return
end

function eval_objective(model::Optimizer, values)
    if !isnothing(model.nlp_block) && model.nlp_block.has_objective
        return MOI.eval_objective(model.nlp_block.evaluator, values)
    else
        return eval_func(values, model.objective)
    end
end

function eval_gradient(
    func::MOI.ScalarQuadraticFunction{Float64},
    grad_f,
    values,
)
    fill!(grad_f, 0.0)
    for term in func.affine_terms
        grad_f[term.variable.value] += term.coefficient
    end
    for term in func.quadratic_terms
        grad_f[term.variable_1.value] +=
            term.coefficient * values[term.variable_2.value]
        # if variables are the same, the coefficient is already multiplied by 2
        if term.variable_1 != term.variable_2
            grad_f[term.variable_2.value] +=
                term.coefficient * values[term.variable_1.value]
        end
    end
end

function eval_objective_gradient(model::Optimizer, grad_f, values)
    if (!isnothing(model.nlp_block) && model.nlp_block.has_objective)
        MOI.eval_objective_gradient(model.nlp_block.evaluator, grad_f, values)
    else
        eval_gradient(model.objective, grad_f, values)
    end
end

# `isapprox(0.0, 1e-16)` is false but `_is_approx(0.0, 1e-16)` is true.
_is_approx(x, y) = isapprox(x, y, atol = Base.rtoldefault(Float64))

approx_in(value, set::MOI.EqualTo) = _is_approx(value, MOI.constant(set))
function approx_in(value, set::MOI.LessThan{Float64})
    return (_is_approx(value, MOI.constant(set)) || value < MOI.constant(set))
end
function approx_in(value, set::MOI.GreaterThan{Float64})
    return (_is_approx(value, MOI.constant(set)) || value > MOI.constant(set))
end
