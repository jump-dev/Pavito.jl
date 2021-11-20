# MOI tests

function run_moi_tests(msd::Bool)
    # The default for `diverging_iterates_tol` is `1e-20`, which makes Ipopt
    # terminate with `ITERATION_LIMIT` for most infeasible problems instead of
    # `NORM_LIMIT`.
    ipopt = MOI.OptimizerWithAttributes(
        Ipopt.Optimizer,
        MOI.Silent() => true,
        "diverging_iterates_tol" => 1e-18,
    )
    optimizer_constructor = MOI.OptimizerWithAttributes(
        Pavito.Optimizer,
        MOI.Silent() => true,
        "mip_solver" => first(values(mip_solvers)),
        "cont_solver" => ipopt,
        "mip_solver_drives" => msd,
    )
    optimizer = MOI.instantiate(optimizer_constructor)

    @test MOI.get(optimizer, MOI.SolverName()) == "Pavito"
    @test MOI.supports_incremental_interface(optimizer)

    config = MOIT.Config(
        atol = 1e-6, rtol = 1e-6,
        optimal_status = MOI.OPTIMAL,
        exclude = Any[
            MOI.ConstraintDual,
            MOI.VariableName,
            MOI.ConstraintName,
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

    MOIT.runtests(optimizer, config,
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
