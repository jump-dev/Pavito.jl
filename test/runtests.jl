#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 Pavito solver unit tests
=========================================================#

using JuMP
import ConicBenchmarkUtilities
using Pavito
using MathProgBase

using Compat.Test
using Compat.Printf

import Compat: stdout
import Compat: stderr

if VERSION < v"0.7.0-"
    jump_path = Pkg.dir("JuMP")
end

if VERSION > v"0.7.0-"
    using Logging
    disable_logging(Logging.Error)

    jump_path = joinpath(dirname(pathof(JuMP)), "..")
end


include("nlptest.jl")
include("conictest.jl")

# test absolute tolerance and Pavito printing level
TOL = 1e-3
ll = 2
redirect = true

# use JuMP list of available solvers
include(joinpath(jump_path, "test", "solvers.jl"))


# MIP solvers
tol_int = 1e-9
tol_feas = 1e-7
tol_gap = 0.0

mip_solvers = Dict{String,MathProgBase.AbstractMathProgSolver}()
if glp
    mip_solvers["GLPK"] = GLPKMathProgInterface.GLPKSolverMIP(msg_lev=0, tol_int=tol_int, tol_bnd=tol_feas, mip_gap=tol_gap)
end
if cpx
    mip_solvers["CPLEX"] = CPLEX.CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_EPINT=tol_int, CPX_PARAM_EPRHS=tol_feas, CPX_PARAM_EPGAP=tol_gap)
end
if grb
    mip_solvers["Gurobi"] = Gurobi.GurobiSolver(OutputFlag=0, IntFeasTol=tol_int, FeasibilityTol=tol_feas, MIPGap=tol_gap)
end
#if cbc
#    mip_solvers["CBC"] = Cbc.CbcSolver(logLevel=0, integerTolerance=tol_int, primalTolerance=tol_feas, ratioGap=tol_gap, check_warmstart=false)
#end

# NLP solvers
cont_solvers = Dict{String,MathProgBase.AbstractMathProgSolver}()
if ipt
    cont_solvers["Ipopt"] = Ipopt.IpoptSolver(print_level=0)
end
if kni
    cont_solvers["Knitro"] = KNITRO.KnitroSolver(objrange=1e16, outlev=0, maxit=100000)
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

# run tests
@testset "Algorithm - $(msd ? "MSD" : "Iter")" for msd in [false, true]
    @testset "MILP solver - $mipname" for (mipname, mip) in mip_solvers
        if msd && !applicable(MathProgBase.setlazycallback!, MathProgBase.ConicModel(mip), x -> x)
            # Only test MSD on lazy callback solvers
            continue
        end
        @testset "NLP models - $conname" for (conname, con) in cont_solvers
            println("\nNLP models: $(msd ? "MSD" : "Iter"), $mipname, $conname")
            run_qp(msd, mip, con, ll, redirect)
            run_nlp(msd, mip, con, ll, redirect)
        end
        @testset "Exp+SOC models - $conname" for (conname, con) in cont_solvers
            println("\nExp+SOC models: $(msd ? "MSD" : "Iter"), $mipname, $conname")
            run_soc(msd, mip, con, ll, redirect)
            run_expsoc(msd, mip, con, ll, redirect)
        end
        flush(stdout)
        flush(stderr)
    end
    println()
end
