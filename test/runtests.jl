#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using Test

import Cbc
import GLPK
import Ipopt
import MathOptInterface

const MOI = MathOptInterface

include("MOI_wrapper.jl")
include("jump_tests.jl")

# !!! info
#     We test with both Cbc and GLPK because they have very different
#     implementations of the MOI API: GLPK supports incremental modification and
#     supports lazy constraints, whereas Cbc supports copy_to and does not
#     support lazy constraints. In addition, Cbc uses MatrixOfConstraints to
#     simplify the copy process, needing an additional cache if we modify after
#     the solve.

@testset "MOI" begin
    TestMOIWrapper.runtests(
        MOI.OptimizerWithAttributes(
            GLPK.Optimizer,
            "msg_lev" => 0,
            "tol_int" => 1e-9,
            "tol_bnd" => 1e-7,
            "mip_gap" => 0.0,
        ),
        MOI.OptimizerWithAttributes(Ipopt.Optimizer, MOI.Silent() => true),
    )
end

@testset "Cbc" begin
    TestMOIWrapper._run_moi_tests(
        false,  # mip_solver_drives
        MOI.OptimizerWithAttributes(Cbc.Optimizer, MOI.Silent() => true),
        MOI.OptimizerWithAttributes(Ipopt.Optimizer, MOI.Silent() => true),
    )
end

@testset "JuMP" begin
    TestJuMP.runtests(
        MOI.OptimizerWithAttributes(
            GLPK.Optimizer,
            "msg_lev" => 0,
            "tol_int" => 1e-9,
            "tol_bnd" => 1e-7,
            "mip_gap" => 0.0,
        ),
        MOI.OptimizerWithAttributes(Ipopt.Optimizer, MOI.Silent() => true),
    )
end
