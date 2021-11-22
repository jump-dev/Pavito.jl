#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

module TestMOIWrapper

import MathOptInterface
import Pavito
import Test

const MOI = MathOptInterface

function runtests(mip_solver, cont_solver)
    Test.@testset "$(msd)" for msd in (true, false)
        _run_moi_tests(msd, mip_solver, cont_solver)
    end
    return
end

function _run_moi_tests(msd::Bool, mip_solver, cont_solver)
    pavito = Pavito.Optimizer()
    MOI.set(pavito, MOI.Silent(), true)
    MOI.set(pavito, MOI.RawOptimizerAttribute("mip_solver_drives"), msd)
    MOI.set(pavito, MOI.RawOptimizerAttribute("mip_solver"), mip_solver)
    MOI.set(pavito, MOI.RawOptimizerAttribute("cont_solver"), cont_solver)
    MOI.Test.runtests(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            MOI.Bridges.full_bridge_optimizer(pavito, Float64),
        ),
        MOI.Test.Config(
            atol = 1e-4,
            rtol = 1e-4,
            optimal_status = MOI.LOCALLY_SOLVED,
            exclude = Any[
                MOI.ConstraintPrimal,
                MOI.ConstraintDual,
                MOI.ConstraintBasisStatus,
                MOI.DualObjectiveValue,
            ],
        ),
        exclude = String[
            # not implemented:
            "test_attribute_SolverVersion",
            # TODO Pavito only returns LOCALLY_INFEASIBLE, not INFEASIBLE:
            # see https://github.com/jump-dev/MathOptInterface.jl/issues/1671
            "INFEASIBLE",
            "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_",
            # invalid model:
            "test_constraint_ZeroOne_bounds_3",
            "test_linear_VectorAffineFunction_empty_row",
            # CachingOptimizer does not throw if optimizer not attached:
            "test_model_copy_to_UnsupportedAttribute",
            "test_model_copy_to_UnsupportedConstraint",
            # NLP features not supported:
            "test_nonlinear_hs071_NLPBlockDual",
            "test_nonlinear_invalid",
            # conic mostly unsupported:
            # TODO when ConstraintPrimal is fixed, use some conic tests e.g. SOC
            # see https://github.com/jump-dev/MathOptInterface.jl/pull/1046
            # see https://github.com/jump-dev/MathOptInterface.jl/issues/846
            "test_conic",
            # TODO ConstraintPrimal not supported, should use a fallback in future:
            # see https://github.com/jump-dev/MathOptInterface.jl/issues/1310
            "test_solve_result_index",
            "test_quadratic_constraint",
            "test_quadratic_nonconvex",
            "test_quadratic_nonhomogeneous",
            "test_linear_integration",
            "test_linear_integer",
            "test_linear_Semi",
            "test_linear_Interval_inactive",
            "test_linear_FEASIBILITY_SENSE",
        ],
    )
    return
end

end
