#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 Pavito solver object
=========================================================#

export PavitoSolver

# Dummy solver
struct UnsetSolver <: MathProgBase.AbstractMathProgSolver end

# Pavito solver
mutable struct PavitoSolver <: MathProgBase.AbstractMathProgSolver
    log_level::Int              # Verbosity flag: 0 for quiet, higher for basic solve info
    timeout::Float64            # Time limit for algorithm (in seconds)
    rel_gap::Float64            # Relative optimality gap termination condition
    mip_solver_drives::Bool     # Let MILP solver manage convergence ("branch and cut")
    mip_solver::MathProgBase.AbstractMathProgSolver # MILP solver
    cont_solver::MathProgBase.AbstractMathProgSolver # Continuous NLP solver
end

function PavitoSolver(;
    log_level = 1,
    timeout = Inf,
    rel_gap = 1e-5,
    mip_solver_drives = nothing,
    mip_solver = UnsetSolver(),
    cont_solver = UnsetSolver(),
    )

    if mip_solver == UnsetSolver()
        error("No MILP solver specified (set mip_solver)\n")
    end
    if mip_solver_drives == nothing
        mip_solver_drives = applicable(MathProgBase.setlazycallback!, MathProgBase.ConicModel(mip_solver), x -> x)
    elseif mip_solver_drives && !applicable(MathProgBase.setlazycallback!, MathProgBase.ConicModel(mip_solver), x -> x)
        error("MIP solver does not support callbacks (cannot set mip_solver_drives = true)")
    end

    if cont_solver == UnsetSolver()
        error("No continuous NLP solver specified (set cont_solver)\n")
    end
    if !applicable(MathProgBase.NonlinearModel, cont_solver)
        error("Continuous solver (cont_solver) specified is not a derivative-based NLP solver recognized by MathProgBase (try Pajarito solver if your continuous solver is conic)\n")
    end

    # Deepcopy the solvers because we may change option values inside Pavito
    return PavitoSolver(log_level, timeout, rel_gap, mip_solver_drives, deepcopy(mip_solver), deepcopy(cont_solver))
end

# Create Pavito conic model
MathProgBase.ConicModel(s::PavitoSolver) = MathProgBase.ConicModel(ConicNonlinearBridge.ConicNLPWrapper(nlp_solver=s))

# Create Pavito LinearQuadratic model
MathProgBase.LinearQuadraticModel(s::PavitoSolver) = MathProgBase.NonlinearToLPQPBridge(MathProgBase.NonlinearModel(s))

# Create Pavito nonlinear model
MathProgBase.NonlinearModel(s::PavitoSolver) = PavitoNonlinearModel(s.log_level, s.timeout, s.rel_gap, s.mip_solver_drives, s.mip_solver, s.cont_solver)
