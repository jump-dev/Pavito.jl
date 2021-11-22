#  Copyright 2017, Iain Dunning, Joey Huchette, Miles Lubin, and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# JuMP
# An algebraic modelling langauge for Julia
# See http://github.com/jump-dev/JuMP.jl
#############################################################################
# test/solvers.jl
# Detect and load available solvers
#############################################################################

function try_import(name::Symbol)
    try
        @eval import $name
        return true
    catch e
        return false
    end
end

mip_solvers = Dict{String,MOI.OptimizerWithAttributes}()
if try_import(:GLPK)
    mip_solvers["GLPK"] = MOI.OptimizerWithAttributes(
        GLPK.Optimizer,
        "msg_lev" => 0,
        "tol_int" => tol_int,
        "tol_bnd" => tol_feas,
        "mip_gap" => tol_gap,
    )
end
if try_import(:CPLEX)
    mip_solvers["CPLEX"] = MOI.OptimizerWithAttributes(
        CPLEX.Optimizer,
        "CPX_PARAM_SCRIND" => 0,
        "CPX_PARAM_EPINT" => tol_int,
        "CPX_PARAM_EPRHS" => tol_feas,
        "CPX_PARAM_EPGAP" => tol_gap,
    )
end
if try_import(:Gurobi)
    mip_solvers["Gurobi"] = MOI.OptimizerWithAttributes(
        Gurobi.Optimizer,
        "OutputFlag" => 0,
        "IntFeasTol" => tol_int,
        "FeasibilityTol" => tol_feas,
        "MIPGap" => tol_gap,
    )
end
if try_import(:Cbc)
    mip_solvers["Cbc"] = MOI.OptimizerWithAttributes(
        Cbc.Optimizer,
        "logLevel" => 0,
        "integerTolerance" => tol_int,
        "primalTolerance" => tol_feas,
        "ratioGap" => tol_gap,
        "check_warmstart" => false,
    )
end

# NLP solvers
cont_solvers = Dict{String,MOI.OptimizerWithAttributes}()
if try_import(:Ipopt)
    # the default for `diverging_iterates_tol`, `1e-20`, makes Ipopt terminate
    # with `ITERATION_LIMIT` for most infeasible problems instead of `NORM_LIMIT`
    cont_solvers["Ipopt"] = MOI.OptimizerWithAttributes(
        Ipopt.Optimizer,
        MOI.Silent() => true,
        # "diverging_iterates_tol" => 1e-18,
    )
end
if try_import(:KNITRO)
    cont_solvers["Knitro"] = MOI.OptimizerWithAttributes(
        KNITRO.Optimizer,
        "objrange" => 1e16,
        "outlev" => 0,
        "maxit" => 100000,
    )
end

# print solvers
println("\nMIP solvers:")
for (i, sname) in enumerate(keys(mip_solvers))
    @printf "%2d  %s\n" i sname
end
println("\nNLP solvers:")
for (i, sname) in enumerate(keys(cont_solvers))
    @printf "%2d  %s\n" i sname
end
println()
