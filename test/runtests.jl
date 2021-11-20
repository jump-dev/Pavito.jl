#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 Pavito solver unit tests
=========================================================#

using Test
using Printf
import MathOptInterface
const MOI = MathOptInterface
const MOIT = MOI.Test
import JuMP
import Ipopt
import Pavito

# test options
TOL = 1e-3 # test absolute tolerance
redirect = true # printing

# Pavito algorithms to run
use_msd = [
    false,
    true,
]
alg(msd::Bool) = (msd ? "MSD" : "Iter")

# options for MIP solvers
tol_int = 1e-9
tol_feas = 1e-7
tol_gap = 0.0

# load MIP and NLP solvers
include("solvers.jl")

# run MOI tests
println("starting MOI tests")
include("MOI_wrapper.jl")
@testset "MOI tests - $(alg(msd))" for msd in use_msd
    println("\n", alg(msd))
    run_moi_tests(msd)
end
println()

# run instance tests
println("starting instance tests")
include("nlptest.jl")
# include("conictest.jl") TODO rewrite for MathProgBase -> MathOptInterface
@testset "instance tests - $(alg(msd)), $mipname, $conname" for
    msd in use_msd, (mipname, mip) in mip_solvers, (conname, con) in cont_solvers
    if msd && !MOI.supports(MOI.instantiate(mip), MOI.LazyConstraintCallback())
        # only test MSD on lazy callback solvers
        continue
    end
    println("\n$(alg(msd)), $mipname, $conname")

    run_qp(msd, mip, con, redirect)
    run_nlp(msd, mip, con, redirect)

    # TODO enable SOC tests: https://github.com/jump-dev/MathOptInterface.jl/pull/1046
    # run_soc(msd, mip, con, redirect)
    # TODO for Exp tests, need: https://github.com/jump-dev/MathOptInterface.jl/issues/846
    # run_expsoc(msd, mip, con, redirect)
end
println()
