# MOI tests

function run_moi_tests(
    msd::Bool,
    mip_solver,
    cont_solver,
)
    pavito = Pavito.Optimizer()
    MOI.set(pavito, MOI.Silent(), true)
    MOI.set(pavito, MOI.RawOptimizerAttribute("mip_solver_drives"), msd)
    MOI.set(pavito, MOI.RawOptimizerAttribute("mip_solver"), mip_solver)
    MOI.set(pavito, MOI.RawOptimizerAttribute("cont_solver"), cont_solver)
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.Bridges.full_bridge_optimizer(pavito, Float64),
    )

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
end
