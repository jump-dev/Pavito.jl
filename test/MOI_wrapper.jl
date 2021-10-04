using Test

#using MathOptInterface
#const MOI = MathOptInterface
const MOIT = MOI.Test

import Pavito

const CONFIG = MOIT.Config(Float64, atol=1e-6, rtol=1e-6, exclude=Any[MOI.ConstraintDual])

@testset "MOI tests - $(msd ? "MSD" : "Iter")" for msd in [false, true]
    # The default for `diverging_iterates_tol` is `1e-20` which makes Ipopt terminates with `ITERATION_LIMIT` for most infeasible
    # problems instead of `NORM_LIMIT`.
    ipopt = optimizer_with_attributes(Ipopt.Optimizer, MOI.Silent() => true, "diverging_iterates_tol" => 1e-18)
    optimizer_constructor = MOI.OptimizerWithAttributes(Pavito.Optimizer, MOI.Silent() => true, "mip_solver" => first(values(mip_solvers)), "cont_solver" => ipopt, "mip_solver_drives" => msd)
    optimizer = MOI.instantiate(optimizer_constructor)

    @testset "SolverName" begin
        @test MOI.get(optimizer, MOI.SolverName()) == "Pavito"
    end

    @testset "supports_default_copy_to" begin
        @test MOI.Utilities.supports_default_copy_to(optimizer, false)
        @test !MOI.Utilities.supports_default_copy_to(optimizer, true)
    end

    @testset "Unit" begin
         MOIT.feasibility_sense(optimizer, CONFIG)
         MOIT.max_sense(optimizer, CONFIG)
         MOIT.min_sense(optimizer, CONFIG)
         MOIT.time_limit_sec(optimizer, CONFIG)
         MOIT.silent(optimizer, CONFIG)
    end

    @testset "Integer Linear" begin
        excludes = [
             # `ConstraintPrimal` not implemented
             "int1", "semiinttest",
             # Not supported by continuous solver and not discrete.
             "semiconttest",
             # Not supported by GLPK
             "int2", "indicator1", "indicator2", "indicator3", "indicator4"
        ]
        if msd
            # GLPK has an integer-infeasible solution
            push!(excludes, "knapsack")
        end
        MOIT.intlineartest(optimizer, CONFIG, excludes)
    end
end
