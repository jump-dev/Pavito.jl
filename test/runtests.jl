#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 Pavito solver tests
=========================================================#

using Test
using Printf
import MathOptInterface
const MOI = MathOptInterface
const MOIT = MOI.Test
import JuMP
import Ipopt
import MINLPTests
import Pavito

TOL = 1e-3 # test absolute tolerance
log_level = 0

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
include("import_solvers.jl")
@assert haskey(mip_solvers, "GLPK")
@assert haskey(cont_solvers, "Ipopt")

println("starting MOI tests")
include("MOI_wrapper.jl")
@testset "MOI tests - $(alg(msd))" for msd in use_msd
    println("\n", alg(msd))
    run_moi_tests(msd, mip_solvers["GLPK"], cont_solvers["Ipopt"])
end
println()

println("starting instance tests and printing tests")
include("qp_nlp_tests.jl")
@testset "instance tests - $(alg(msd)), $mipname, $conname" for
    msd in use_msd, (mipname, mip) in mip_solvers, (conname, con) in cont_solvers
    if msd && !MOI.supports(MOI.instantiate(mip), MOI.LazyConstraintCallback())
        continue # only test MSD on lazy callback solvers
    end
    println("\n$(alg(msd)), $mipname, $conname")

    run_qp(msd, mip, con, log_level, TOL)
    run_nlp(msd, mip, con, log_level, TOL)
end
println()
@testset "printing tests - $(alg(msd)), log_level $ll" for msd in use_msd,
    ll in 0:2
    run_log_level(msd, first(values(mip_solvers)), first(values(cont_solvers)),
        ll, TOL)
end
println()

println("starting MINLPTests tests")
@testset "MINLPTests - $(alg(msd))" for msd in use_msd
    pavito = JuMP.optimizer_with_attributes(
        Pavito.Optimizer,
        "timeout" => 120.0,
        "mip_solver_drives" => msd,
        "mip_solver" => first(values(mip_solvers)),
        "cont_solver" => first(values(cont_solvers)),
        "log_level" => log_level,
    )

    exclude = String[
        # TODO fix failures:
        "003_010",
        "003_011",
        "003_012",
        "003_013",
        "003_014",
        "003_015",
        "003_016",
        "006_010",
        "007_010",
        "007_020",
    ]
    MINLPTests.test_nlp_mi(pavito, exclude = exclude, objective_tol = TOL,
        primal_tol = TOL, dual_tol = NaN)
end
