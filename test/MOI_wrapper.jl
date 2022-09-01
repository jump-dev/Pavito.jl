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
            infeasible_status = MOI.LOCALLY_INFEASIBLE,
            optimal_status = MOI.LOCALLY_SOLVED,
            exclude = Any[
                MOI.ConstraintDual,
                MOI.ConstraintBasisStatus,
                MOI.DualObjectiveValue,
                MOI.NLPBlockDual,
            ],
        ),
        exclude = String[
            # Not implemented:
            "test_attribute_SolverVersion",
            # Invalid model:
            "test_constraint_ZeroOne_bounds_3",
            "test_linear_VectorAffineFunction_empty_row",
            # CachingOptimizer does not throw if optimizer not attached:
            "test_model_copy_to_UnsupportedAttribute",
            "test_model_copy_to_UnsupportedConstraint",
            # NLP features not supported:
            "test_nonlinear_invalid",
            # NORM_LIMIT instead of DUAL_INFEASIBLE
            "test_linear_DUAL_INFEASIBLE",
            "test_linear_DUAL_INFEASIBLE_2",
            "test_solve_TerminationStatus_DUAL_INFEASIBLE",
            # ITERATION_LIMIT instead of OPTIMAL
            "test_linear_integer_knapsack",
            "test_linear_integer_solve_twice",
            # INFEASIBLE instead of LOCALLY_INFEASIBLE?
            "test_linear_Semicontinuous_integration",
            "test_linear_Semiinteger_integration",
        ],
    )
    return
end

end
