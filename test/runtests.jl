#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using Test

import GLPK
import Ipopt
import MathOptInterface

const MOI = MathOptInterface

include("MOI_wrapper.jl")
include("jump_tests.jl")

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
