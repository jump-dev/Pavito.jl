#  Copyright 2017, Iain Dunning, Joey Huchette, Miles Lubin, and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# JuMP
# An algebraic modelling langauge for Julia
# See http://github.com/JuliaOpt/JuMP.jl
#############################################################################
# test/solvers.jl
# Detect and load solvers
# Should be run as part of runtests.jl
#############################################################################

function try_import(name::Symbol)
    try
        @eval import $name
        return true
    catch e
        return false
    end
end

# Load available solvers
grb = try_import(:Gurobi)
cpx = try_import(:CPLEX)
glp = try_import(:GLPK)
ipt = try_import(:Ipopt)
kni = try_import(:KNITRO)

mip_solvers = Dict{String, MOI.OptimizerWithAttributes}()
if glp
    mip_solvers["GLPK"] = optimizer_with_attributes(GLPK.Optimizer, "msg_lev" => 0, "tol_int" => tol_int, "tol_bnd" => tol_feas, "mip_gap" => tol_gap)
end
if cpx
    mip_solvers["CPLEX"] = optimizer_with_attributes(CPLEX.Optimizer, "CPX_PARAM_SCRIND" => 0, "CPX_PARAM_EPINT" => tol_int, "CPX_PARAM_EPRHS" => tol_feas, "CPX_PARAM_EPGAP" => tol_gap)
end
if grb
    mip_solvers["Gurobi"] = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0, "IntFeasTol" => tol_int, "FeasibilityTol" => tol_feas, "MIPGap" => tol_gap)
end
#if cbc
#    mip_solvers["CBC"] = Cbc.CbcSolver(logLevel=0, integerTolerance=tol_int, primalTolerance=tol_feas, ratioGap=tol_gap, check_warmstart=false)
#end

# NLP solvers
cont_solvers = Dict{String, MOI.OptimizerWithAttributes}()
if ipt
    # We add a cache because of https://github.com/jump-dev/Ipopt.jl/issues/211
    cont_solvers["Ipopt"] = MOI.OptimizerWithAttributes(() -> MOIU.CachingOptimizer(MOIU.UniversalFallback(MOIU.Model{Float64}()), MOI.instantiate(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))))
end
if kni
    cont_solvers["Knitro"] = optimizer_with_attributes(KNITRO.Optimizer, "objrange" => 1e16, "outlev" => 0, "maxit" => 100000)
end

# print solvers
println("\nMILP solvers:")
for (i, sname) in enumerate(keys(mip_solvers))
    @printf "%2d  %s\n" i sname
end
println("\nNLP solvers:")
for (i, sname) in enumerate(keys(cont_solvers))
    @printf "%2d  %s\n" i sname
end
println()
