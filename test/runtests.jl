#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 Pavito solver unit tests
=========================================================#

using JuMP
import Pavito

using Test
using Printf

#using Logging
#disable_logging(Logging.Error)

include("nlptest.jl")
include("conictest.jl")

# test absolute tolerance and Pavito printing level
TOL = 1e-3
ll = 2
redirect = true

# MIP solvers
tol_int = 1e-9
tol_feas = 1e-7
tol_gap = 0.0

include("solvers.jl")

# run tests
@testset "Algorithm - $(msd ? "MSD" : "Iter")" for msd in [false] #, true]
    @testset "MILP solver - $mipname" for (mipname, mip) in mip_solvers
        if msd && !MOI.supports(MOI.instantiate(mip), MOI.LazyConstraintCallback())
            # Only test MSD on lazy callback solvers
            continue
        end
        @testset "NLP models - $conname" for (conname, con) in cont_solvers
            println("\nNLP models: $(msd ? "MSD" : "Iter"), $mipname, $conname")
            run_qp(msd, mip, con, ll, redirect)
            run_nlp(msd, mip, con, ll, redirect)
        end
#        @testset "Exp+SOC models - $conname" for (conname, con) in cont_solvers
#            println("\nExp+SOC models: $(msd ? "MSD" : "Iter"), $mipname, $conname")
#            run_soc(msd, mip, con, ll, redirect)
#            run_expsoc(msd, mip, con, ll, redirect)
#        end
        flush(stdout)
        flush(stderr)
    end
    println()
end
