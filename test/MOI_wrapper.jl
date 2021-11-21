# MOI tests

function run_moi_tests(
    msd::Bool,
    mip_solver,
    cont_solver,
)
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.Bridges.full_bridge_optimizer(Pavito.Optimizer(), Float64),
    )
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.RawOptimizerAttribute("mip_solver_drives"), msd)
    MOI.set(model, MOI.RawOptimizerAttribute("mip_solver"), mip_solver)
    MOI.set(model, MOI.RawOptimizerAttribute("cont_solver"), cont_solver)

    config = MOIT.Config(
        atol = 1e-4,
        rtol = 1e-4,
        optimal_status = MOI.LOCALLY_SOLVED,
        exclude = Any[
            MOI.ConstraintDual,
            MOI.ConstraintBasisStatus,
            MOI.DualObjectiveValue,
            MOI.ObjectiveBound,
        ],
    )

    exclude = String[
        # not implemented:
        "test_attribute_SolverVersion",
    ]
    if msd
        # exclude for MSD algorithm only
        push!(exclude, "...")
    end

    MOIT.runtests(
        model,
        config,
        # include = [],
        exclude = exclude,
        warn_unsupported = true,
    )

    # CONFIG = MOI.DeprecatedTest.Config(atol=1e-6, rtol=1e-6, duals=false, query=false)

    # @testset "Unit" begin
    #      MOI.DeprecatedTest.feasibility_sense(optimizer, CONFIG)
    #      MOI.DeprecatedTest.max_sense(optimizer, CONFIG)
    #      MOI.DeprecatedTest.min_sense(optimizer, CONFIG)
    #      MOI.DeprecatedTest.time_limit_sec(optimizer, CONFIG)
    #      MOI.DeprecatedTest.silent(optimizer, CONFIG)
    # end
    #
    # @testset "Integer Linear" begin
    #     excludes = [
    #          # `ConstraintPrimal` not implemented
    #          "int1", "semiinttest",
    #          # Not supported by continuous solver and not discrete.
    #          "semiconttest",
    #          # Not supported by GLPK
    #          "int2", "indicator1", "indicator2", "indicator3", "indicator4"
    #     ]
    #     if msd
    #         # GLPK has an integer-infeasible solution
    #         push!(excludes, "knapsack")
    #     end
    #     MOI.DeprecatedTest.intlineartest(optimizer, CONFIG, excludes)
    # end
end
