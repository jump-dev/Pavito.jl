#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using JuMP
import ConicBenchmarkUtilities
using Pavito
using Base.Test


# Tests absolute tolerance and Pajarito printing level
TOL = 1e-3
ll = 3
redirect = true

# Define dictionary of solvers, using JuMP list of available solvers
include(Pkg.dir("JuMP", "test", "solvers.jl"))
include("nlptest.jl")
include("conictest.jl")

solvers = Dict{String,Dict{String,MathProgBase.AbstractMathProgSolver}}()

# MIP solvers
solvers["MILP"] = Dict{String,MathProgBase.AbstractMathProgSolver}()

tol_int = 1e-9
tol_feas = 1e-7
tol_gap = 0.0

if glp
    solvers["MILP"]["GLPK"] = GLPKMathProgInterface.GLPKSolverMIP(msg_lev=GLPK.MSG_OFF, tol_int=tol_int, tol_bnd=tol_feas, mip_gap=tol_gap)
end
if cpx
    solvers["MILP"]["CPLEX"] = CPLEX.CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_EPINT=tol_int, CPX_PARAM_EPRHS=tol_feas, CPX_PARAM_EPGAP=tol_gap)
end
if grb
    solvers["MILP"]["Gurobi"] = Gurobi.GurobiSolver(OutputFlag=0, IntFeasTol=tol_int, FeasibilityTol=tol_feas, MIPGap=tol_gap)
end
#if cbc
#    solvers["MILP"]["CBC"] = Cbc.CbcSolver(logLevel=0, integerTolerance=tol_int, primalTolerance=tol_feas, ratioGap=tol_gap, check_warmstart=false)
#end

# NLP solvers
solvers["NLP"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
if ipt
    solvers["NLP"]["Ipopt"] = Ipopt.IpoptSolver(print_level=0)
end
if kni
    solvers["NLP"]["Knitro"] = KNITRO.KnitroSolver(objrange=1e16, outlev=0, maxit=100000)
end


println("\nSolvers:")
for (stype, snames) in solvers
    println("\n$stype")
    for (i, sname) in enumerate(keys(snames))
        @printf "%2d  %s\n" i sname
    end
end
println()


@testset "Algorithm - $(msd ? "MSD" : "Iter")" for msd in [false, true]
    @testset "MILP solver - $mipname" for (mipname, mip) in solvers["MILP"]
        if msd && !applicable(MathProgBase.setlazycallback!, MathProgBase.ConicModel(mip), x -> x)
            # Only test MSD on lazy callback solvers
            continue
        end

        @testset "NLP models - $conname" for (conname, con) in solvers["NLP"]
            println("\nNLP models: $(msd ? "MSD" : "Iter"), $mipname, $conname")
            run_qp(msd, mip, con, ll, redirect)
            run_nlp(msd, mip, con, ll, redirect)
        end

        @testset "Exp+SOC models - $conname" for (conname, con) in solvers["NLP"]
            println("\nExp+SOC models: $(msd ? "MSD" : "Iter"), $mipname, $conname")
            run_soc(msd, mip, con, ll, redirect)
            run_expsoc(msd, mip, con, ll, redirect)
        end

        flush(STDOUT)
        flush(STDERR)
    end
    println()
end
