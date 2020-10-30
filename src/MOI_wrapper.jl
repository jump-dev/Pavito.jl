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
    silent::Bool
    log_level::Int                          # Verbosity flag: 0 for quiet, higher for basic solve info
    timeout::Float64                        # Time limit for algorithm (in seconds)
    rel_gap::Float64                        # Relative optimality gap termination condition
    mip_solver_drives::Union{Nothing, Bool} # Let MILP solver manage convergence ("branch and cut")
    mip_solver                              # MILP solver
    cont_solver                             # Continuous NLP solver

    mip_optimizer::Union{Nothing, MOI.ModelLike}        # MILP optimizer instantiated from `mip_solver`
    cont_optimizer::Union{Nothing, MOI.ModelLike}       # Continuous NLP optimizer instantiated from `cont_solver`
    infeasible_optimizer::Union{Nothing, MOI.ModelLike} # Continuous NLP optimizer instantiated from `cont_solver` used when `cont_optimizer` is infeasible

    θ::Union{Nothing, MOI.VariableIndex}                # MILP objective function when the original one is nonlineaer
    mip_variables::Vector{MOI.VariableIndex}            # Variable indices of `mip_optimizer`
    cont_variables::Vector{MOI.VariableIndex}           # Variable indices of `cont_optimizer`
    infeasible_variables::Vector{MOI.VariableIndex}     # Variable indices of `infeasible_optimizer`

    # Slack variable indices for `infeasible_optimizer`
    nl_slack_variables::Union{Nothing, Vector{MOI.VariableIndex}}                # for the nonlinear constraints
    quad_less_than_slack_variables::Union{Nothing, Vector{MOI.VariableIndex}}    # for the less than constraints
    quad_greater_than_slack_variables::Union{Nothing, Vector{MOI.VariableIndex}} # for the greater than constraints

    # Quadratic constraints for `infeasible_optimizer`
    quad_less_than_infeasible_con::Union{Nothing, Vector{MOI.ConstraintIndex{SQF, MOI.LessThan{Float64}}}}       # `q - slack <= ub`
    quad_greater_than_infeasible_con::Union{Nothing, Vector{MOI.ConstraintIndex{SQF, MOI.GreaterThan{Float64}}}} # `q + slack >= lb`
    infeasible_evaluator::InfeasibleNLPEvaluator # NLP evaluator used for `infeasible_optimizer`
    int_indices::BitSet # Indices of discrete variables

    nlp_block::Union{Nothing, MOI.NLPBlockData}           # NLP block set to `Optimizer`
    objective::Union{Nothing, MOI.AbstractScalarFunction} # Objective function set to `Optimizer`
    quad_less_than::Vector{Tuple{SQF, MOI.LessThan{Float64}}}       # Cached quadratic less than constraints
    quad_greater_than::Vector{Tuple{SQF, MOI.GreaterThan{Float64}}} # Cached quadratic greater than constraints
    status::MOI.TerminationStatusCode # Termination status to be returned
    incumbent::Vector{Float64}        # Starting values set and then current best nonlinear feasible solution
    new_incumb::Bool                  # `true` if a better nonlinear feasible solution was found
    mip_solution::Vector{Float64}     # MIP solution cached for used to check redundancy of Lazy Constraint
    total_time::Float64               # Total solve time
    objective_value::Float64          # Objective value corresponding to `incumbent`
    objective_bound::Float64          # Best objective bound found by MILP
    objective_gap::Float64            # Objective gap between objective value and bound
    num_iters_or_callbacks::Int       # Either the number of iterations or the number of calls to the lazy constraint callback if `mip_solver_drives`

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
    model.infeasible_optimizer = nothing
    model.θ = nothing
    model.mip_variables = MOI.VariableIndex[]
    model.cont_variables = MOI.VariableIndex[]
    model.infeasible_variables = MOI.VariableIndex[]
    model.nl_slack_variables = nothing
    model.quad_less_than_slack_variables = nothing
    model.quad_greater_than_slack_variables = nothing
    model.quad_less_than_infeasible_con = nothing
    model.quad_greater_than_infeasible_con = nothing
    model.int_indices = BitSet()

    model.nlp_block = nothing
    model.objective = nothing
    model.quad_less_than = Tuple{SQF, MOI.LessThan{Float64}}[]
    model.quad_greater_than = Tuple{SQF, MOI.GreaterThan{Float64}}[]
    model.incumbent = Float64[]
    model.status = MOI.OPTIMIZE_NOT_CALLED
    return
end

MOI.Utilities.supports_default_copy_to(::Optimizer, copy_names::Bool) = !copy_names

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike; copy_names = false)
    return MOI.Utilities.default_copy_to(model, src, copy_names)
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
    return optimizer
end

function _cont(model::Optimizer)
    if model.cont_optimizer === nothing
        model.cont_optimizer = _new_cont(model.cont_solver)
    end
    return model.cont_optimizer
end

function _infeasible(model::Optimizer)
    if model.infeasible_optimizer === nothing
        model.infeasible_optimizer = _new_cont(model.cont_solver)
        MOI.set(model.infeasible_optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    end
    return model.infeasible_optimizer
end

function clean_slacks(model::Optimizer)
    if model.nl_slack_variables !== nothing
        MOI.delete(_infeasible(model), model.nl_slack_variables)
        model.nl_slack_variables = nothing
    end
    if model.quad_less_than_slack_variables !== nothing
        MOI.delete(_infeasible(model), model.quad_less_than_slack_variables)
        model.quad_less_than_slack_variables = nothing
        MOI.delete(_infeasible(model), model.quad_less_than_infeasible_con)
        model.quad_less_than_infeasible_con = nothing
    end
    if model.quad_greater_than_slack_variables !== nothing
        MOI.delete(_infeasible(model), model.quad_greater_than_slack_variables)
        model.quad_greater_than_slack_variables = nothing
        MOI.delete(_infeasible(model), model.quad_greater_than_infeasible_con)
        model.quad_greater_than_infeasible_con = nothing
    end
end

MOI.get(model::Optimizer, ::MOI.NumberOfVariables) = length(model.incumbent)
function MOI.add_variable(model::Optimizer)
    push!(model.mip_variables, MOI.add_variable(_mip(model)))
    push!(model.cont_variables, MOI.add_variable(_cont(model)))
    push!(model.infeasible_variables, MOI.add_variable(_infeasible(model)))
    if model.nl_slack_variables !== nothing
        # The slack variables are assumed to be added after all the `infeasible_variables`
        # so we delete them now and will add it back during `optimize!` if needed.
        clean_slacks(model)
    end
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
is_discrete(::Type{<:Union{MOI.Integer, MOI.ZeroOne, MOI.Semiinteger{Float64}}}) = true
function MOI.supports_constraint(model::Optimizer, F::Type{MOI.SingleVariable}, S::Type{<:MOI.AbstractScalarSet})
    return MOI.supports_constraint(_mip(model), F, S) &&
        (is_discrete(S) || MOI.supports_constraint(_cont(model), F, S))
end
function MOI.add_constraint(model::Optimizer, func::MOI.SingleVariable, set::MOI.AbstractScalarSet)
    if is_discrete(typeof(set))
        push!(model.int_indices, func.variable.value)
    else
        MOI.add_constraint(_cont(model), _map(model.cont_variables, func), set)
        MOI.add_constraint(_infeasible(model), _map(model.infeasible_variables, func), set)
    end
    return MOI.add_constraint(_mip(model), _map(model.mip_variables, func), set)
end

function MOI.supports_constraint(model::Optimizer, F::Type{SQF}, S::Type{<:Union{MOI.LessThan{Float64}, MOI.GreaterThan{Float64}}})
    return MOI.supports_constraint(_cont(model), F, S)
end
function MOI.add_constraint(model::Optimizer, func::SQF, set::MOI.LessThan{Float64})
    clean_slacks(model)
    MOI.add_constraint(_cont(model), _map(model.cont_variables, func), set)
    push!(model.quad_less_than, (MOI.Utilities.canonical(func), copy(set)))
    return MOI.ConstraintIndex{typeof(func), typeof(set)}(length(model.quad_less_than))
end
function MOI.add_constraint(model::Optimizer, func::SQF, set::MOI.GreaterThan{Float64})
    clean_slacks(model)
    MOI.add_constraint(_cont(model), _map(model.cont_variables, func), set)
    push!(model.quad_greater_than, (MOI.Utilities.canonical(func), copy(set)))
    return MOI.ConstraintIndex{typeof(func), typeof(set)}(length(model.quad_greater_than))
end
function MOI.get(model::Optimizer, attr::MOI.NumberOfConstraints{SQF, <:Union{MOI.LessThan{Float64}, MOI.GreaterThan{Float64}}})
    return MOI.get(_cont(model), attr)
end

function MOI.supports_constraint(model::Optimizer, F::Type{<:MOI.AbstractFunction}, S::Type{<:MOI.AbstractSet})
    return MOI.supports_constraint(_mip(model), F, S) && MOI.supports_constraint(_cont(model), F, S)
end
function MOI.add_constraint(model::Optimizer, func::MOI.AbstractFunction, set::MOI.AbstractSet)
    MOI.add_constraint(_cont(model), _map(model.cont_variables, func), set)
    MOI.add_constraint(_infeasible(model), _map(model.infeasible_variables, func), set)
    return MOI.add_constraint(_mip(model), _map(model.mip_variables, func), set)
end
function MOI.get(model::Optimizer, attr::MOI.NumberOfConstraints)
    return MOI.get(_mip(model), attr)
end

MOI.supports(::Optimizer, ::MOI.NLPBlock) = true
function MOI.set(model::Optimizer, attr::MOI.NLPBlock, block::MOI.NLPBlockData)
    clean_slacks(model)
    model.nlp_block = block
    if !MOI.supports(_cont(model), MOI.NLPBlock())
        error("Continuous solver (`cont_solver`) specified is not a derivative-based NLP solver recognized by MathOptInterface (try Pajarito solver if your continuous solver is conic)\n")
    end
    MOI.set(_cont(model), attr, block)
end

function MOI.supports(model::Optimizer, attr::MOI.ObjectiveSense)
    return true
end
function MOI.set(model::Optimizer, attr::MOI.ObjectiveSense, sense)
    if sense == MOI.FEASIBILITY_SENSE
        model.objective = nothing
    end
    MOI.set(_mip(model), attr, sense)
    MOI.set(_cont(model), attr, sense)
end
MOI.get(model::Optimizer, attr::MOI.ObjectiveSense) = MOI.get(_mip(model), attr)
function MOI.supports(model::Optimizer, attr::MOI.ObjectiveFunction)
    return MOI.supports(_mip(model), attr) && MOI.supports(_cont(model), attr)
end
function MOI.supports(model::Optimizer, attr::MOI.ObjectiveFunction{SQF})
    return MOI.supports(_cont(model), attr)
end
function MOI.set(model::Optimizer, attr::MOI.ObjectiveFunction, func)
    # We make a copy (as the user might modify it)
    model.objective = copy(func)
    MOI.set(_mip(model), attr, _map(model.mip_variables, func))
    MOI.set(_cont(model), attr, _map(model.cont_variables, func))
end
function MOI.set(model::Optimizer, attr::MOI.ObjectiveFunction{SQF}, func::SQF)
    # We make a copy (as the user might modify it) and canonicalize
    # (as we're going to use it many times so it will be worth it to remove some duplicates).
    model.objective = MOI.Utilities.canonical(func)
    MOI.set(_cont(model), attr, _map(model.cont_variables, func))
end


function MOI.get(model::Optimizer, param::MOI.RawParameter)
    return getproperty(model, Symbol(param.name))
end
MOI.supports(::Optimizer, param::MOI.RawParameter) = Symbol(param.name) in fieldnames(Optimizer)
function MOI.set(model::Optimizer, param::MOI.RawParameter, value)
    setproperty!(model, Symbol(param.name), value)
end
MOI.supports(::Optimizer, ::MOI.Silent) = true
function MOI.set(model::Optimizer, ::MOI.Silent, value::Bool)
    model.silent = value
end
MOI.get(model::Optimizer, ::MOI.Silent) = model.silent
MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Nothing)
    MOI.set(model, MOI.RawParameter("timeout"), Inf)
end
function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value)
    MOI.set(model, MOI.RawParameter("timeout"), value)
end
function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    value = MOI.get(model, MOI.RawParameter("timeout"))
    return isfinite(value) ? value : nothing
end

MOI.get(model::Optimizer, ::MOI.SolveTime) = model.total_time

MOI.get(model::Optimizer, ::MOI.TerminationStatus) = model.status
function MOI.get(model::Optimizer, ::MOI.VariablePrimal, v::MOI.VariableIndex)
    return model.incumbent[v.value]
end
MOI.get(model::Optimizer, ::MOI.ObjectiveValue) = model.objective_value
MOI.get(model::Optimizer, ::MOI.ObjectiveBound) = model.objective_bound

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
        return MOI.FEASIBLE_POINT
    else
        return MOI.NO_SOLUTION
    end
end

function MOI.get(::Optimizer, ::MOI.DualStatus)
    return MOI.NO_SOLUTION
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT ? 1 : 0
end

function MOI.get(::Optimizer, ::MOI.SolverName)
    return "Pavito"
end
