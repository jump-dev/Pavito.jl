#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 MathOptInterface wrapper
=========================================================#

function MOI.is_empty(model::Optimizer)
    return (
        isnothing(model.mip_optimizer) || MOI.is_empty(model.mip_optimizer)
    ) && (
        isnothing(model.cont_optimizer) || MOI.is_empty(model.cont_optimizer)
    )
end

function MOI.empty!(model::Optimizer)
    model.mip_optimizer = nothing
    model.cont_optimizer = nothing
    model.infeasible_optimizer = nothing
    model.nlp_obj_var = nothing
    model.mip_variables = MOI.VariableIndex[]
    model.cont_variables = MOI.VariableIndex[]
    model.infeasible_variables = MOI.VariableIndex[]
    model.nl_slack_variables = nothing
    model.quad_LT_slack = nothing
    model.quad_GT_slack = nothing
    model.quad_LT_infeasible_con = nothing
    model.quad_GT_infeasible_con = nothing
    model.int_indices = BitSet()

    model.nlp_block = nothing
    model.objective = nothing
    model.quad_LT =
        Tuple{MOI.ScalarQuadraticFunction{Float64},MOI.LessThan{Float64}}[]
    model.quad_GT =
        Tuple{MOI.ScalarQuadraticFunction{Float64},MOI.GreaterThan{Float64}}[]
    model.incumbent = Float64[]
    model.status = MOI.OPTIMIZE_NOT_CALLED
    return
end

MOI.get(::Optimizer, ::MOI.SolverName) = "Pavito"

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(model, src)
end

MOI.get(model::Optimizer, ::MOI.NumberOfVariables) = length(model.incumbent)

function MOI.add_variable(model::Optimizer)
    push!(model.mip_variables, MOI.add_variable(_mip(model)))
    push!(model.cont_variables, MOI.add_variable(_cont(model)))
    push!(model.infeasible_variables, MOI.add_variable(_infeasible(model)))
    if !isnothing(model.nl_slack_variables)
        # the slack variables are assumed to be added after all the
        # `infeasible_variables`, so we delete them now and add back during
        # `optimize!` if needed
        _clean_slacks(model)
    end
    push!(model.incumbent, NaN)
    return MOI.VariableIndex(length(model.mip_variables))
end

function MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex},
)
    return true
end

function MOI.set(
    model::Optimizer,
    attr::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value,
)
    MOI.set(_cont(model), attr, vi, value)
    model.incumbent[vi.value] = something(value, NaN)
    return
end

function _map(variables::Vector{MOI.VariableIndex}, x)
    return MOI.Utilities.map_indices(vi -> variables[vi.value], x)
end

is_discrete(::Type{<:MOI.AbstractSet}) = false
function is_discrete(
    ::Type{<:Union{MOI.Integer,MOI.ZeroOne,MOI.Semiinteger{Float64}}},
)
    return true
end

function MOI.supports_constraint(
    model::Optimizer,
    F::Type{MOI.VariableIndex},
    S::Type{<:MOI.AbstractScalarSet},
)
    return (
        MOI.supports_constraint(_mip(model), F, S) &&
        (is_discrete(S) || MOI.supports_constraint(_cont(model), F, S))
    )
end

function MOI.add_constraint(
    model::Optimizer,
    func::MOI.VariableIndex,
    set::MOI.AbstractScalarSet,
)
    if is_discrete(typeof(set))
        push!(model.int_indices, func.value)
    else
        MOI.add_constraint(_cont(model), _map(model.cont_variables, func), set)
        MOI.add_constraint(
            _infeasible(model),
            _map(model.infeasible_variables, func),
            set,
        )
    end
    return MOI.add_constraint(_mip(model), _map(model.mip_variables, func), set)
end

function MOI.supports_constraint(
    model::Optimizer,
    F::Type{MOI.ScalarQuadraticFunction{Float64}},
    S::Type{<:Union{MOI.LessThan{Float64},MOI.GreaterThan{Float64}}},
)
    return MOI.supports_constraint(_cont(model), F, S)
end

function MOI.add_constraint(
    model::Optimizer,
    func::MOI.ScalarQuadraticFunction{Float64},
    set::MOI.LessThan{Float64},
)
    _clean_slacks(model)
    MOI.add_constraint(_cont(model), _map(model.cont_variables, func), set)
    push!(model.quad_LT, (MOI.Utilities.canonical(func), copy(set)))
    return MOI.ConstraintIndex{typeof(func),typeof(set)}(length(model.quad_LT))
end

function MOI.add_constraint(
    model::Optimizer,
    func::MOI.ScalarQuadraticFunction{Float64},
    set::MOI.GreaterThan{Float64},
)
    _clean_slacks(model)
    MOI.add_constraint(_cont(model), _map(model.cont_variables, func), set)
    push!(model.quad_GT, (MOI.Utilities.canonical(func), copy(set)))
    return MOI.ConstraintIndex{typeof(func),typeof(set)}(length(model.quad_GT))
end

function MOI.get(
    model::Optimizer,
    attr::MOI.NumberOfConstraints{
        MOI.ScalarQuadraticFunction{Float64},
        <:Union{MOI.LessThan{Float64},MOI.GreaterThan{Float64}},
    },
)
    return MOI.get(_cont(model), attr)
end

function MOI.supports_constraint(
    model::Optimizer,
    F::Type{<:MOI.AbstractFunction},
    S::Type{<:MOI.AbstractSet},
)
    return (
        MOI.supports_constraint(_mip(model), F, S) &&
        MOI.supports_constraint(_cont(model), F, S)
    )
end

function MOI.add_constraint(
    model::Optimizer,
    func::MOI.AbstractFunction,
    set::MOI.AbstractSet,
)
    MOI.add_constraint(_cont(model), _map(model.cont_variables, func), set)
    MOI.add_constraint(
        _infeasible(model),
        _map(model.infeasible_variables, func),
        set,
    )
    return MOI.add_constraint(_mip(model), _map(model.mip_variables, func), set)
end

function MOI.get(model::Optimizer, attr::MOI.NumberOfConstraints)
    return MOI.get(_mip(model), attr)
end

MOI.supports(::Optimizer, ::MOI.NLPBlock) = true

function MOI.set(model::Optimizer, attr::MOI.NLPBlock, block::MOI.NLPBlockData)
    _clean_slacks(model)
    model.nlp_block = block
    if !MOI.supports(_cont(model), MOI.NLPBlock())
        error(
            "Continuous solver (`cont_solver`) is not a derivative-based " *
            "NLP solver recognized by MathOptInterface (try Pajarito solver " *
            "if your continuous solver is conic)",
        )
    end
    return MOI.set(_cont(model), attr, block)
end

MOI.supports(model::Optimizer, attr::MOI.ObjectiveSense) = true

function MOI.set(model::Optimizer, attr::MOI.ObjectiveSense, sense)
    if sense == MOI.FEASIBILITY_SENSE
        model.objective = nothing
    end
    MOI.set(_mip(model), attr, sense)
    return MOI.set(_cont(model), attr, sense)
end

MOI.get(model::Optimizer, attr::MOI.ObjectiveSense) = MOI.get(_mip(model), attr)

function MOI.supports(model::Optimizer, attr::MOI.ObjectiveFunction)
    return MOI.supports(_mip(model), attr) && MOI.supports(_cont(model), attr)
end

function MOI.supports(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}},
)
    return MOI.supports(_cont(model), attr)
end

function MOI.set(model::Optimizer, attr::MOI.ObjectiveFunction, func)
    # make a copy (as the user might modify it)
    model.objective = copy(func)
    MOI.set(_mip(model), attr, _map(model.mip_variables, func))
    return MOI.set(_cont(model), attr, _map(model.cont_variables, func))
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}},
    func::MOI.ScalarQuadraticFunction{Float64},
)
    # make a copy (as the user might modify it) and canonicalize
    model.objective = MOI.Utilities.canonical(func)
    return MOI.set(_cont(model), attr, _map(model.cont_variables, func))
end

function MOI.get(model::Optimizer, param::MOI.RawOptimizerAttribute)
    return getproperty(model, Symbol(param.name))
end

function MOI.supports(::Optimizer, param::MOI.RawOptimizerAttribute)
    return (Symbol(param.name) in fieldnames(Optimizer))
end

function MOI.set(model::Optimizer, param::MOI.RawOptimizerAttribute, value)
    return setproperty!(model, Symbol(param.name), value)
end

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(model::Optimizer, ::MOI.Silent, value::Bool)
    return (model.log_level = (value ? 0 : 1))
end

MOI.get(model::Optimizer, ::MOI.Silent) = (model.log_level <= 0)

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Nothing)
    return MOI.set(model, MOI.RawOptimizerAttribute("timeout"), Inf)
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value)
    return MOI.set(model, MOI.RawOptimizerAttribute("timeout"), value)
end

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    value = MOI.get(model, MOI.RawOptimizerAttribute("timeout"))
    return (isfinite(value) ? value : nothing)
end

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.total_time

MOI.get(model::Optimizer, ::MOI.TerminationStatus) = model.status

MOI.get(model::Optimizer, ::MOI.RawStatusString) = string(model.status)

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    v::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attr)
    return model.incumbent[v.value]
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return model.objective_value
end

MOI.get(model::Optimizer, ::MOI.ObjectiveBound) = model.objective_bound

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index != 1
        return MOI.NO_SOLUTION
    end
    term_status = MOI.get(model, MOI.TerminationStatus())
    if term_status == MOI.LOCALLY_SOLVED
        return MOI.FEASIBLE_POINT
    elseif term_status == MOI.ALMOST_LOCALLY_SOLVED
        return MOI.NEARLY_FEASIBLE_POINT
    else
        return MOI.NO_SOLUTION
    end
end

MOI.get(::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return (MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT ? 1 : 0)
end

# utilities:

function _mip(model::Optimizer)
    if isnothing(model.mip_optimizer)
        if isnothing(model.mip_solver)
            error("No MIP solver specified (set `mip_solver` attribute)\n")
        end

        model.mip_optimizer =
            MOI.instantiate(model.mip_solver, with_bridge_type = Float64)

        supports_lazy =
            MOI.supports(model.mip_optimizer, MOI.LazyConstraintCallback())
        if isnothing(model.mip_solver_drives)
            model.mip_solver_drives = supports_lazy
        elseif model.mip_solver_drives && !supports_lazy
            error(
                "MIP solver (`mip_solver`) does not support lazy constraint " *
                "callbacks (cannot set `mip_solver_drives` attribute to `true`)",
            )
        end
    end
    return model.mip_optimizer
end

function _new_cont(optimizer_constructor)
    if isnothing(optimizer_constructor)
        error(
            "No continuous NLP solver specified (set `cont_solver` attribute)",
        )
    end
    optimizer =
        MOI.instantiate(optimizer_constructor, with_bridge_type = Float64)
    return optimizer
end

function _cont(model::Optimizer)
    if isnothing(model.cont_optimizer)
        model.cont_optimizer = _new_cont(model.cont_solver)
    end
    return model.cont_optimizer
end

function _infeasible(model::Optimizer)
    if isnothing(model.infeasible_optimizer)
        model.infeasible_optimizer = _new_cont(model.cont_solver)
        MOI.set(model.infeasible_optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    end
    return model.infeasible_optimizer
end

function _clean_slacks(model::Optimizer)
    if !isnothing(model.nl_slack_variables)
        MOI.delete(_infeasible(model), model.nl_slack_variables)
        model.nl_slack_variables = nothing
    end
    if !isnothing(model.quad_LT_slack)
        MOI.delete(_infeasible(model), model.quad_LT_slack)
        model.quad_LT_slack = nothing
        MOI.delete(_infeasible(model), model.quad_LT_infeasible_con)
        model.quad_LT_infeasible_con = nothing
    end
    if !isnothing(model.quad_GT_slack)
        MOI.delete(_infeasible(model), model.quad_GT_slack)
        model.quad_GT_slack = nothing
        MOI.delete(_infeasible(model), model.quad_GT_infeasible_con)
        model.quad_GT_infeasible_con = nothing
    end
end
