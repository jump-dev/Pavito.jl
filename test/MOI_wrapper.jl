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
                MOI.NLPBlockDual,
            ],
        ),
        exclude = String[
            # TODO(odow): this looks like a failure. ObjectiveValue if `Inf`
            "test_quadratic_Integer_SecondOrderCone",
            # NLP features not supported:
            "test_nonlinear_invalid",
            # TODO Pavito only returns LOCALLY_INFEASIBLE, not INFEASIBLE:
            # https://github.com/jump-dev/MathOptInterface.jl/issues/1671
            "INFEASIBLE",
            "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_",
            "test_conic_SecondOrderCone_negative_post_bound_ii",
            "test_conic_SecondOrderCone_negative_post_bound_iii",
            # TODO: ConstraintPrimal not supported.
            # We should use a fallback in future:
            # https://github.com/jump-dev/MathOptInterface.jl/issues/1310
            "test_solve_result_index",
            "test_quadratic_constraint",
            "test_quadratic_nonconvex",
            "test_quadratic_nonhomogeneous",
            "test_linear_integration",
            "test_linear_integer",
            "test_linear_Semi",
            "test_linear_Interval_inactive",
            "test_linear_FEASIBILITY_SENSE",
            "test_conic_GeometricMeanCone_",
            "test_conic_NormInfinityCone_",
            "test_conic_NormOneCone",
            "test_conic_RotatedSecondOrderCone_VectorOfVariables",
            "test_conic_RotatedSecondOrderCone_out_of_order",
            "test_conic_SecondOrderCone_Nonnegatives",
            "test_conic_SecondOrderCone_Nonpositives",
            "test_conic_SecondOrderCone_VectorAffineFunction",
            "test_conic_SecondOrderCone_VectorOfVariables",
            "test_conic_SecondOrderCone_no_initial_bound",
            "test_conic_SecondOrderCone_out_of_order",
            "test_conic_linear_VectorAffineFunction",
            "test_conic_linear_VectorAffineFunction_2",
            "test_conic_linear_VectorOfVariables",
            "test_conic_linear_VectorOfVariables_2",
            # Ipopt throws InvalidModel for the NLP:
            "test_constraint_ZeroOne_bounds_3",
            "test_linear_VectorAffineFunction_empty_row",
            # CachingOptimizer does not throw if optimizer not attached:
            "test_model_copy_to_UnsupportedAttribute",
            "test_model_copy_to_UnsupportedConstraint",
            # Not implemented:
            "test_attribute_SolverVersion",
        ],
    )
    return
end

end
