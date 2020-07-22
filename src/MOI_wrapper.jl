#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 Pavito optimizer object
=========================================================#

const SQF = MOI.ScalarQuadraticFunction{Float64}

# Pavito solver
mutable struct Optimizer <: MOI.AbstractOptimizer
    log_level::Int                          # Verbosity flag: 0 for quiet, higher for basic solve info
    timeout::Float64                        # Time limit for algorithm (in seconds)
    rel_gap::Float64                        # Relative optimality gap termination condition
    mip_solver_drives::Union{Nothing, Bool} # Let MILP solver manage convergence ("branch and cut")
    mip_solver                              # MILP solver
    cont_solver                             # Continuous NLP solver

    mip_optimizer::Union{Nothing, MOI.ModelLike}
    cont_optimizer::Union{Nothing, MOI.ModelLike}
    inf_optimizer::Union{Nothing, MOI.ModelLike}

    θ::Union{Nothing, MOI.VariableIndex}
    mip_variables::Vector{MOI.VariableIndex}
    cont_variables::Vector{MOI.VariableIndex}
    inf_variables::Vector{MOI.VariableIndex}
    slack_variables::Union{Nothing, Vector{MOI.VariableIndex}}
    inf_evaluator::InfeasibleNLPEvaluator
    int_indices::BitSet

    nlp_block::Union{Nothing, MOI.NLPBlockData}
    objective::Union{Nothing, MOI.AbstractScalarFunction}
    status::MOI.TerminationStatusCode
    incumbent::Vector{Float64}
    new_incumb::Bool
    total_time::Float64
    objective_value::Float64
    objective_bound::Float64
    objective_gap::Float64
    num_iters_or_callbacks::Int

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

function MOI.is_empty(model::Optimizer)
    return (model.mip_optimizer === nothing || MOI.is_empty(model.mip_optimizer)) &&
        (model.cont_optimizer === nothing || MOI.is_empty(model.cont_optimizer))
end
function MOI.empty!(model::Optimizer)
    model.mip_optimizer = nothing
    model.cont_optimizer = nothing
    model.inf_optimizer = nothing
    model.θ = nothing
    model.mip_variables = MOI.VariableIndex[]
    model.cont_variables = MOI.VariableIndex[]
    model.inf_variables = MOI.VariableIndex[]
    model.slack_variables = nothing
    model.int_indices = BitSet()

    model.nlp_block = nothing
    model.objective = nothing
    model.status = MOI.OPTIMIZE_NOT_CALLED
    model.incumbent = Float64[]
    model.new_incumb = false
    model.total_time = 0.0
    model.objective_value = NaN
    model.objective_bound = NaN
    model.objective_gap = Inf
    model.num_iters_or_callbacks = 0
    return
end

function _mip(model::Optimizer)
    if model.mip_optimizer === nothing
        if model.mip_solver === nothing
            error("No MILP solver specified (set `mip_solver` attribute)\n")
        end

        model.mip_optimizer = MOI.instantiate(model.mip_solver, with_bridge_type=Float64)

        supports_lazy = MOI.supports(model.mip_optimizer, MOI.LazyConstraintCallback())
        if model.mip_solver_drives === nothing
            model.mip_solver_drives = supports_lazy
        elseif model.mip_solver_drives && !supports_lazy
            error("MIP solver (`mip_solver`) does not support lazy constraint callbacks (cannot set `mip_solver_drives` attribute to `true`)")
        end
    end
    return model.mip_optimizer
end

function _new_cont(optimizer_constructor)
    if optimizer_constructor === nothing
        error("No continuous NLP solver specified (set `cont_solver` attribute)\n")
    end

    optimizer = MOI.instantiate(optimizer_constructor, with_bridge_type=Float64)
    if !MOI.supports(optimizer, MOI.NLPBlock())
        error("Continuous solver (`cont_solver`) specified is not a derivative-based NLP solver recognized by MathOptInterface (try Pajarito solver if your continuous solver is conic)\n")
    end
    return optimizer
end

function _cont(model::Optimizer)
    if model.cont_optimizer === nothing
        model.cont_optimizer = _new_cont(model.cont_solver)
    end
    return model.cont_optimizer
end

function _inf(model::Optimizer)
    if model.inf_optimizer === nothing
        model.inf_optimizer = _new_cont(model.cont_solver)
        MOI.set(model.inf_optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    end
    return model.inf_optimizer
end

function MOI.add_variable(model::Optimizer)
    push!(model.mip_variables, MOI.add_variable(_mip(model)))
    push!(model.cont_variables, MOI.add_variable(_cont(model)))
    if model.slack_variables !== nothing
        # The slack variables are assumed to be added after all the `inf_variables`
        # so we delete them now and will add it back during `optimize!` if needed.
        MOI.delete(_inf(model), model.slack_variables)
        model.slack_variables = nothing
    end
    push!(model.inf_variables, MOI.add_variable(_inf(model)))
    push!(model.incumbent, NaN)
    return MOI.VariableIndex(length(model.mip_variables))
end
function MOI.add_variables(model::Optimizer, n)
    return [MOI.add_variable(model) for i in 1:n]
end
MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex}) = true
function MOI.set(model::Optimizer, attr::MOI.VariablePrimalStart, vi::MOI.VariableIndex, value)
    MOI.set(_cont(model), attr, vi, value)
    model.incumbent[vi.value] = something(value, NaN)
    return
end

_map(variables::Vector{MOI.VariableIndex}, x) = MOI.Utilities.map_indices(vi -> variables[vi.value], x)

is_discrete(::Type{<:MOI.AbstractSet}) = false
is_discrete(::Type{<:Union{MOI.Integer, MOI.ZeroOne}}) = true
function MOI.supports_constraint(model::Optimizer, F::Type{<:MOI.AbstractFunction}, S::Type{<:MOI.AbstractSet})
    return MOI.supports_constraint(_mip(model), F, S) &&
        (is_discrete(S) || MOI.supports_constraint(_cont(model), F, S))
end
function MOI.add_constraint(model::Optimizer, func::MOI.SingleVariable, set::MOI.AbstractSet)
    if is_discrete(typeof(set))
        @show model.mip_variables
        @show func.variable.value
        push!(model.int_indices, func.variable.value)
    else
        MOI.add_constraint(_cont(model), _map(model.cont_variables, func), set)
        MOI.add_constraint(_inf(model), _map(model.inf_variables, func), set)
    end
    return MOI.add_constraint(_mip(model), _map(model.mip_variables, func), set)
end
function MOI.add_constraint(model::Optimizer, func::MOI.AbstractFunction, set::MOI.AbstractSet)
    MOI.add_constraint(_cont(model), _map(model.cont_variables, func), set)
    MOI.add_constraint(_inf(model), _map(model.inf_variables, func), set)
    return MOI.add_constraint(_mip(model), _map(model.mip_variables, func), set)
end
MOI.supports(::Optimizer, ::MOI.NLPBlock) = true
function MOI.set(model::Optimizer, attr::MOI.NLPBlock, block::MOI.NLPBlockData)
    model.nlp_block = block
    MOI.set(_cont(model), attr, block)
end

function MOI.supports(model::Optimizer, attr::MOI.ObjectiveSense)
    return true
end
function MOI.set(model::Optimizer, attr::MOI.ObjectiveSense, sense)
    if sense == MOI.FEASIBILITY_SENSE
        model.objective = nothing
    end
    MOI.set(model.mip_optimizer, attr, sense)
    MOI.set(model.cont_optimizer, attr, sense)
end
function MOI.supports(model::Optimizer, attr::MOI.ObjectiveFunction)
    return MOI.supports(_mip(model), attr) && MOI.supports(_cont(model), attr)
end
function MOI.supports(model::Optimizer, attr::MOI.ObjectiveFunction{SQF})
    return MOI.supports(_cont(model), attr)
end
function MOI.set(model::Optimizer, attr::MOI.ObjectiveFunction, func)
    # We make a copy (as the user might modify it)
    model.objective = copy(func)
    MOI.set(model.mip_optimizer, attr, _map(model.mip_variables, func))
    MOI.set(model.cont_optimizer, attr, _map(model.cont_variables, func))
end
function MOI.set(model::Optimizer, attr::MOI.ObjectiveFunction{SQF}, func::SQF)
    # We make a copy (as the user might modify it) and canonicalize
    # (as we're going to use it many times so it will be worth it to remove some duplicates).
    model.objective = MOI.Utilities.canonical(func)
    MOI.set(model.cont_optimizer, attr, _map(model.cont_variables, func))
end


function MOI.get(model::Optimizer, param::MOI.RawParameter)
    return getproperty(model, Symbol(param.name))
end
MOI.supports(::Optimizer, param::MOI.RawParameter) = Symbol(param.name) in fieldnames(Optimizer)
function MOI.set(model::Optimizer, param::MOI.RawParameter, value)
    setproperty!(model, Symbol(param.name), value)
end

MOI.get(model::Optimizer, ::MOI.SolveTime) = model.total_time

MOI.get(model::Optimizer, ::MOI.TerminationStatus) = model.status
function MOI.get(model::Optimizer, ::MOI.VariablePrimal, vi::MOI.VariableIndex)
    return model.incumbent[vi.value]
end
MOI.get(model::Optimizer, ::MOI.ObjectiveValue) = model.objective_value
MOI.get(model::Optimizer, ::MOI.ObjectiveBound) = model.objective_bound
