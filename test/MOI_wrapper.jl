using Test

#using MathOptInterface
#const MOI = MathOptInterface
const MOIT = MOI.Test

import Pavito
#const OPTIMIZER_CONSTRUCTOR = MOI.OptimizerWithAttributes(Pavito.Optimizer, MOI.Silent() => true, "mip_solver" => first(values(mip_solvers)), "cont_solver" => first(values(cont_solvers)), "mip_solver_drives" => false)
const OPTIMIZER_CONSTRUCTOR = MOI.OptimizerWithAttributes(Pavito.Optimizer, "log_level" => 100, "mip_solver" => first(values(mip_solvers)), "cont_solver" => first(values(cont_solvers)), "mip_solver_drives" => false)
const OPTIMIZER = MOI.instantiate(OPTIMIZER_CONSTRUCTOR)

@testset "SolverName" begin
    @test MOI.get(OPTIMIZER, MOI.SolverName()) == "Pavito"
end

@testset "supports_default_copy_to" begin
    @test MOI.Utilities.supports_default_copy_to(OPTIMIZER, false)
    @test !MOI.Utilities.supports_default_copy_to(OPTIMIZER, true)
end

const CACHED = MOI.Utilities.CachingOptimizer(MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()), OPTIMIZER)
#const BRIDGED = MOI.instantiate(OPTIMIZER_CONSTRUCTOR, with_bridge_type = Float64)
const CONFIG = MOIT.TestConfig(atol=1e-6, rtol=1e-6, duals=false, query=false)

@testset "Unit" begin
     MOIT.feasibility_sense(OPTIMIZER, CONFIG)
     MOIT.max_sense(OPTIMIZER, CONFIG)
     MOIT.min_sense(OPTIMIZER, CONFIG)
     MOIT.time_limit_sec(OPTIMIZER, CONFIG)
     MOIT.silent(OPTIMIZER, CONFIG)
end
