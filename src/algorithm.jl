#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 Nonlinear model object
=========================================================#

mutable struct PavitoNonlinearModel <: MathProgBase.AbstractNonlinearModel
    log_level::Int              # Verbosity flag: 0 for quiet, higher for basic solve info
    timeout::Float64            # Time limit for algorithm (in seconds)
    rel_gap::Float64            # Relative optimality gap termination condition
    mip_solver_drives::Bool     # Let MILP solver manage convergence ("branch and cut")
    mip_solver::MathProgBase.AbstractMathProgSolver # MILP solver
    cont_solver::MathProgBase.AbstractMathProgSolver # Continuous NLP solver

    numvar::Int
    numconstr::Int
    numnlconstr::Int
    A::SparseMatrixCSC{Float64,Int}
    A_lb::Vector{Float64}
    A_ub::Vector{Float64}
    lb::Vector{Float64}
    ub::Vector{Float64}
    l::Vector{Float64}
    u::Vector{Float64}
    c::Vector{Float64}
    vartypes::Vector{Symbol}
    constrtype::Vector{Symbol}
    constrlinear::Vector{Bool}
    objsense::Symbol
    fixsense::Float64
    objlinear::Bool
    d

    cb
    mip_x::Vector{JuMP.Variable}
    status::Symbol
    incumbent::Vector{Float64}
    oa_started::Bool
    new_incumb::Bool
    totaltime::Float64
    objval::Float64
    objbound::Float64
    objgap::Float64
    iterscbs::Int

    function PavitoNonlinearModel(log_level, timeout, rel_gap, mip_solver_drives, mip_solver, cont_solver)
        m = new()

        m.log_level = log_level
        m.timeout = timeout
        m.rel_gap = rel_gap
        m.mip_solver_drives = mip_solver_drives
        m.mip_solver = mip_solver
        m.cont_solver = cont_solver

        m.status = :NotLoaded
        m.incumbent = Float64[]
        m.oa_started = false
        m.new_incumb = false
        m.totaltime = 0.0
        m.objval = Inf
        m.objbound = -Inf
        m.objgap = Inf
        m.iterscbs = 0

        return m
    end
end


#=========================================================
 MathProgBase functions
=========================================================#

MathProgBase.numvar(m::PavitoNonlinearModel) = m.numvar
MathProgBase.numconstr(m::PavitoNonlinearModel) = m.numconstr
MathProgBase.setwarmstart!(m::PavitoNonlinearModel, x) = (m.incumbent = x)
MathProgBase.setvartype!(m::PavitoNonlinearModel, v::Vector{Symbol}) = (m.vartypes = v)

function MathProgBase.loadproblem!(m::PavitoNonlinearModel, numvar, numconstr, l, u, lb, ub, sense, d)
    m.numvar = numvar
    m.numconstr = numconstr
    m.lb = lb
    m.ub = ub
    m.l = l
    m.u = u
    m.objsense = sense
    m.fixsense = (sense == :Max) ? -1 : 1
    m.d = d
    m.vartypes = fill(:Cont, numvar)

    MathProgBase.initialize(d,  intersect([:Grad, :Jac, :Hess], MathProgBase.features_available(d)))

    m.constrtype = Array{Symbol}(undef, numconstr)
    for i in 1:numconstr
        if (lb[i] > -Inf) && (ub[i] < Inf)
            m.constrtype[i] = :(==)
        elseif lb[i] > -Inf
            m.constrtype[i] = :(>=)
        else
            m.constrtype[i] = :(<=)
        end
    end

    m.incumbent = fill(NaN, numvar)
    m.status = :Loaded

    return nothing
end

function MathProgBase.optimize!(m::PavitoNonlinearModel)
    start = time()
    nlptime = 0.0
    miptime = 0.0
    if any(isnan, m.incumbent)
        m.incumbent = zeros(m.numvar)
    end

    # build MILP OA model
    constructlinearcons(m)
    (mip_x, mipmodel) = constructoamodel(m)
    m.mip_x = mip_x
    int_ind = filter(i -> (m.vartypes[i] in (:Int, :Bin)), 1:m.numvar)

    if m.log_level > 0
        println("\nMINLP has a ", (m.objlinear ? "linear" : "nonlinear"), " objective, $(m.numvar) variables ($(length(int_ind)) integer), $(m.numconstr) constraints ($(m.numnlconstr) nonlinear)")
        println("\nPavito started, using ", (m.mip_solver_drives ? "MIP-solver-driven" : "iterative"), " method...")
    end
    flush(stdout)

    # solve initial continuous relaxation NLP model
    (jac_I, jac_J) = MathProgBase.jac_structure(m.d)
    jac_V = zeros(length(jac_I))
    grad_f = zeros(m.numvar+1)
    ini_nlpmodel = MathProgBase.NonlinearModel(m.cont_solver)
    MathProgBase.loadproblem!(ini_nlpmodel, m.numvar, m.numconstr, m.l, m.u, m.lb, m.ub, m.objsense, m.d)
    MathProgBase.setwarmstart!(ini_nlpmodel, m.incumbent[1:m.numvar])
    start_nlp = time()
    MathProgBase.optimize!(ini_nlpmodel)
    nlptime += time() - start_nlp
    ini_nlp_status = MathProgBase.status(ini_nlpmodel)
    if (ini_nlp_status == :Optimal) || (ini_nlp_status == :Suboptimal)
        contsolution = MathProgBase.getsolution(ini_nlpmodel)
        addcuts(m, mipmodel, contsolution, jac_I, jac_J, jac_V, grad_f)
    else
        if ini_nlp_status == :Infeasible
            @warn "initial NLP relaxation infeasible"
            m.status = :Infeasible
        elseif ini_nlp_status == :Unbounded
            @warn "initial NLP relaxation unbounded"
            m.status = :Unbounded
        else
            @warn "NLP solver failure"
            m.status = :Error
        end
        return nothing
    end
    ini_nlp_objval = MathProgBase.getobjval(ini_nlpmodel)
    m.status = :SolvedRelax
    flush(stdout)

    # if have warmstart, use objval and warmstart MIP model
    if !isempty(m.incumbent) && all(isfinite, m.incumbent) && (checkinfeas(m, m.incumbent) < 1e-5)
        if m.objsense == :Max
            m.objval = -MathProgBase.eval_f(m.d, m.incumbent)
        else
            m.objval = MathProgBase.eval_f(m.d, m.incumbent)
        end
        if applicable(MathProgBase.setwarmstart!, internalmodel(mipmodel), m.incumbent)
            MathProgBase.setwarmstart!(internalmodel(mipmodel), [m.incumbent; m.objval])
        end
    end

    # set remaining time limit on MIP solver
    if isfinite(m.timeout) && applicable(MathProgBase.setparameters!, m.mip_solver)
        MathProgBase.setparameters!(m.mip_solver, TimeLimit=max(1., m.timeout - (time() - start)))
    end
    setsolver(mipmodel, m.mip_solver)

    # start main OA algorithm
    flush(stdout)
    m.oa_started = true

    if m.mip_solver_drives
        # MIP-solver-driven method
        cache_contsol = Dict{Vector{Float64},Vector{Float64}}()

        function lazycallback(cb)
            m.iterscbs += 1

            # if integer assignment has been seen before, use cached point
            mipsolution = getvalue(m.mip_x)
            round_mipsol = round.(mipsolution)
            if haskey(cache_contsol, round_mipsol)
                # retrieve existing solution
                contsolution = cache_contsol[round_mipsol]
            else
                # try to solve new subproblem, update incumbent solution if feasible
                start_nlp = time()
                contsolution = solvesubproblem(m, mipsolution)
                nlptime += time() - start_nlp
                cache_contsol[round_mipsol] = contsolution
            end

            # add gradient cuts to MIP model from NLP solution
            if all(isfinite, contsolution)
                m.cb = cb
                addcuts(m, mipmodel, contsolution, jac_I, jac_J, jac_V, grad_f)
            else
                @warn "no cuts could be added, terminating Pavito"
                return JuMP.StopTheSolver
            end
        end
        addlazycallback(mipmodel, lazycallback)

        function heuristiccallback(cb)
            # if have a new best feasible solution since last heuristic solution added, set MIP solution to the new best feasible solution
            if m.new_incumb
                for i in 1:m.numvar
                    setsolutionvalue(cb, m.mip_x[i], m.incumbent[i])
                end
                addsolution(cb)
                m.new_incumb = false
            end
        end
        addheuristiccallback(mipmodel, heuristiccallback)

        m.status = solve(mipmodel, suppress_warnings=true)
        m.objbound = getobjbound(mipmodel)
    else
        # iterative method
        prev_mipsolution = fill(NaN, m.numvar)
        while (time() - start) < m.timeout
            m.iterscbs += 1

            # solve MIP model
            start_mip = time()
            mip_status = solve(mipmodel, suppress_warnings=true)
            miptime += time() - start_mip

            # finish if MIP was infeasible or if problematic status
            if (mip_status == :Infeasible) || (mip_status == :InfeasibleOrUnbounded)
                m.status = :Infeasible
                break
            elseif (mip_status != :Optimal) && (mip_status != :Suboptimal)
                @warn "MIP solver status was $mip_status, terminating Pavito"
                m.status = mip_status
                break
            end
            mipsolution = getvalue(m.mip_x)

            # update best bound from MIP bound
            mipobjbound = getobjbound(mipmodel)
            if isfinite(mipobjbound) && (mipobjbound > m.objbound)
                m.objbound = mipobjbound
            end

            # update gap if best bound and best objective are finite
            if isfinite(m.objval) && isfinite(m.objbound)
                m.objgap = (m.objval - m.objbound)/(abs(m.objval) + 1e-5)
            end
            printgap(m, start)

            # finish if optimal or cycling integer solutions
            if m.objgap <= m.rel_gap
                m.status = :Optimal
                break
            elseif round.(prev_mipsolution[int_ind]) == round.(mipsolution[int_ind])
                @warn "mixed-integer cycling detected, terminating Pavito"
                if isfinite(m.objgap)
                    m.status = :Suboptimal
                else
                    m.status = :FailedOA
                end
                break
            end

            # try to solve new subproblem, update incumbent solution if feasible
            start_nlp = time()
            contsolution = solvesubproblem(m, mipsolution)
            nlptime += time() - start_nlp

            # update gap if best bound and best objective are finite
            if isfinite(m.objval) && isfinite(m.objbound)
                m.objgap = (m.objval - m.objbound)/(abs(m.objval) + 1e-5)
            end
            if m.objgap <= m.rel_gap
                m.status = :Optimal
                break
            elseif round.(prev_mipsolution[int_ind]) == round.(mipsolution[int_ind])
                @warn "mixed-integer cycling detected, terminating Pavito"
                if isfinite(m.objgap)
                    m.status = :Suboptimal
                else
                    m.status = :FailedOA
                end
                break
            end

            # add gradient cuts to MIP model from NLP solution
            if all(isfinite, contsolution)
                addcuts(m, mipmodel, contsolution, jac_I, jac_J, jac_V, grad_f)
            else
                @warn "no cuts could be added, terminating Pavito"
                break
            end

            # warmstart MIP from upper bound
            if !any(isnan, m.incumbent) && !isempty(m.incumbent)
                if applicable(MathProgBase.setwarmstart!, internalmodel(mipmodel), m.incumbent)
                    # extend solution with the objective variable
                    MathProgBase.setwarmstart!(internalmodel(mipmodel), [m.incumbent; m.objval])
                end
            end

            # set remaining time limit on MIP solver
            if isfinite(m.timeout) && applicable(MathProgBase.setparameters!, m.mip_solver)
                MathProgBase.setparameters!(m.mip_solver, TimeLimit=max(1., m.timeout - (time() - start)))
            end
            setsolver(mipmodel, m.mip_solver)

            prev_mipsolution = mipsolution
            flush(stdout)
        end
    end
    flush(stdout)

    # finish
    m.totaltime = time() - start
    if m.log_level > 0
        println("\nPavito finished...\n")
        @printf "Status           %13s\n" m.status
        @printf "Objective value  %13.5f\n" m.fixsense*m.objval
        @printf "Objective bound  %13.5f\n" m.fixsense*m.objbound
        @printf "Objective gap    %13.5f\n" m.fixsense*m.objgap
        if !m.mip_solver_drives
            @printf "Iterations       %13d\n" m.iterscbs
        else
            @printf "Callbacks        %13d\n" m.iterscbs
        end
        @printf "Total time       %13.5f sec\n" m.totaltime
        @printf "MIP total time   %13.5f sec\n" miptime
        @printf "NLP total time   %13.5f sec\n" nlptime
        println()
    end
    flush(stdout)

    return nothing
end

MathProgBase.status(m::PavitoNonlinearModel) = m.status
MathProgBase.getobjval(m::PavitoNonlinearModel) = m.fixsense*m.objval
MathProgBase.getobjbound(m::PavitoNonlinearModel) = m.fixsense*m.objbound
MathProgBase.getobjgap(m::PavitoNonlinearModel) = m.fixsense*m.objgap
MathProgBase.getsolution(m::PavitoNonlinearModel) = m.incumbent
MathProgBase.getsolvetime(m::PavitoNonlinearModel) = m.totaltime


#=========================================================
 NLP utilities
=========================================================#

mutable struct InfeasibleNLPEvaluator <: MathProgBase.AbstractNLPEvaluator
    d
    numconstr::Int
    numnlconstr::Int
    numvar::Int
    constrtype::Vector{Symbol}
    constrlinear::Vector{Bool}
end

MathProgBase.eval_f(d::InfeasibleNLPEvaluator, x) = sum(x[i] for i in (d.numvar+1):length(x))
MathProgBase.eval_hesslag(d::InfeasibleNLPEvaluator, H, x, σ, μ) = MathProgBase.eval_hesslag(d.d, H, x[1:d.numvar], 0.0, μ)
MathProgBase.hesslag_structure(d::InfeasibleNLPEvaluator) = MathProgBase.hesslag_structure(d.d)
MathProgBase.initialize(d::InfeasibleNLPEvaluator, requested_features::Vector{Symbol}) = MathProgBase.initialize(d.d, requested_features)
MathProgBase.features_available(d::InfeasibleNLPEvaluator) = intersect([:Grad, :Jac, :Hess], MathProgBase.features_available(d.d))
MathProgBase.isobjlinear(d::InfeasibleNLPEvaluator) = true
MathProgBase.isobjquadratic(d::InfeasibleNLPEvaluator) = true
MathProgBase.isconstrlinear(d::InfeasibleNLPEvaluator, i::Int) = MathProgBase.isconstrlinear(d.d, i)
MathProgBase.obj_expr(d::InfeasibleNLPEvaluator) = MathProgBase.obj_expr(d.d)
MathProgBase.constr_expr(d::InfeasibleNLPEvaluator, i::Int) = MathProgBase.constr_expr(d.d, i)

function MathProgBase.eval_grad_f(d::InfeasibleNLPEvaluator, g, x)
    g[1:d.numvar] .= 0.0
    g[1+d.numnlconstr:end] .= 1.0
    return nothing
end

function MathProgBase.eval_g(d::InfeasibleNLPEvaluator, g, x)
    MathProgBase.eval_g(d.d, g, x[1:d.numvar])
    k = 1
    for i in 1:d.numconstr
        if d.constrlinear[i]
            continue
        end
        if d.constrtype[i] == :(<=)
            g[i] -= x[k+d.numvar]
        else
            g[i] += x[k+d.numvar]
        end
        k += 1
    end
    return nothing
end

function MathProgBase.jac_structure(d::InfeasibleNLPEvaluator)
    (I, J) = MathProgBase.jac_structure(d.d)
    I_new = copy(I)
    J_new = copy(J)
    k = 1
    for i in 1:(d.numconstr)
        if d.constrlinear[i]
            continue
        end
        push!(I_new, i)
        push!(J_new, k+d.numvar)
        k += 1
    end
    return (I_new, J_new)
end

function MathProgBase.eval_jac_g(d::InfeasibleNLPEvaluator, J, x)
    MathProgBase.eval_jac_g(d.d, J, x[1:d.numvar])
    k = length(J) - d.numnlconstr + 1
    for i in 1:d.numconstr
        if d.constrlinear[i]
            continue
        end
        if d.constrtype[i] == :(<=)
            J[k] = -1.0
        else
            J[k] = 1.0
        end
        k += 1
    end
    return nothing
end


#=========================================================
 Algorithmic utilities
=========================================================#

function constructlinearcons(m::PavitoNonlinearModel)
    # set up map of linear rows
    constrlinear = falses(m.numconstr)
    numlinear = 0
    constraint_to_linear = fill(-1, m.numconstr)
    for i in 1:m.numconstr
        if MathProgBase.isconstrlinear(m.d, i)
            constrlinear[i] = true
            numlinear += 1
            constraint_to_linear[i] = numlinear
        elseif m.constrtype[i] == :(==)
            error("nonlinear equality or two-sided constraints not accepted")
        end
    end
    m.numnlconstr = m.numconstr - numlinear

    # extract sparse jacobian structure
    (jac_I, jac_J) = MathProgBase.jac_structure(m.d)

    # evaluate jacobian at x = 0
    c = zeros(m.numvar)
    x = m.incumbent
    jac_V = zeros(length(jac_I))
    MathProgBase.eval_jac_g(m.d, jac_V, x)
    MathProgBase.eval_grad_f(m.d, c, x)
    m.objlinear = MathProgBase.isobjlinear(m.d)

    if m.objlinear
        m.c = c
    else
        m.c = zeros(m.numvar)
    end

    # build up sparse matrix for linear constraints
    A_I = Int[]
    A_J = Int[]
    A_V = Float64[]

    for k in 1:length(jac_I)
        row = jac_I[k]
        if !constrlinear[row]
            continue
        end
        row = constraint_to_linear[row]
        push!(A_I,row)
        push!(A_J, jac_J[k])
        push!(A_V, jac_V[k])
    end

    m.A = sparse(A_I, A_J, A_V, numlinear, m.numvar)

    # g(x) might have a constant, i.e., a'x + b; find b
    constraint_value = zeros(m.numconstr)
    MathProgBase.eval_g(m.d, constraint_value, x)
    b = constraint_value[constrlinear] - m.A * x
    # linear constraints are of the form lb ≤ a'x + b ≤ ub

    m.A_lb = m.lb[constrlinear] - b
    m.A_ub = m.ub[constrlinear] - b
    m.constrlinear = constrlinear
    return nothing
end

# solve NLP subproblem defined by integer assignment
function solvesubproblem(m::PavitoNonlinearModel, mipsolution)
    new_u = m.u
    new_l = m.l
    for i in 1:m.numvar
        if (m.vartypes[i] == :Int) || (m.vartypes[i] == :Bin)
            new_u[i] = new_l[i] = mipsolution[i]
        end
    end
    nlpmodel = MathProgBase.NonlinearModel(m.cont_solver)
    MathProgBase.loadproblem!(nlpmodel, m.numvar, m.numconstr, new_l, new_u, m.lb, m.ub, m.objsense, m.d)
    MathProgBase.setwarmstart!(nlpmodel, mipsolution[1:m.numvar])
    MathProgBase.optimize!(nlpmodel)

    if MathProgBase.status(nlpmodel) == :Optimal
        # subproblem is feasible, check if solution is new incumbent
        nlp_objval = m.fixsense*MathProgBase.getobjval(nlpmodel)
        nlp_solution = MathProgBase.getsolution(nlpmodel)
        if nlp_objval < m.objval
            m.objval = nlp_objval
            m.incumbent = nlp_solution[1:m.numvar]
            m.new_incumb = true
        end

        return nlp_solution
    end

    # assume subproblem is infeasible, so solve infeasible recovery NLP subproblem
    l_inf = [new_l; zeros(m.numnlconstr)]
    u_inf = [new_u; fill(Inf, m.numnlconstr)]
    d_inf = InfeasibleNLPEvaluator(m.d, m.numconstr, m.numnlconstr, m.numvar, m.constrtype, m.constrlinear)
    infnlpmodel = MathProgBase.NonlinearModel(m.cont_solver)
    MathProgBase.loadproblem!(infnlpmodel, (m.numvar + m.numnlconstr), m.numconstr, l_inf, u_inf, m.lb, m.ub, :Min, d_inf)

    # create the warmstart solution
    inf_initial_solution = zeros(m.numvar + m.numnlconstr);
    inf_initial_solution[1:m.numvar] = mipsolution[1:m.numvar]
    g = zeros(m.numconstr)
    MathProgBase.eval_g(m.d, g, inf_initial_solution[1:m.numvar])
    k = 1
    for i in 1:m.numconstr
        if !m.constrlinear[i]
            if m.constrtype[i] == :(<=)
                val = g[i] - m.ub[i]
            else
                val = m.lb[i] - g[i]
            end
            if val > 0
                # sign of the slack changes if the constraint direction changes
                inf_initial_solution[m.numvar + k] = val
            else
                inf_initial_solution[m.numvar + k] = 0.0
            end
            k += 1
        end
    end
    MathProgBase.setwarmstart!(infnlpmodel, inf_initial_solution)

    # solve
    MathProgBase.optimize!(infnlpmodel)
    infnlpmodel_status = MathProgBase.status(infnlpmodel)
    if (infnlpmodel_status != :Optimal) && (infnlpmodel_status != :Suboptimal)
        @warn "NLP solver failure"
    end

    return MathProgBase.getsolution(infnlpmodel)
end

function addcuts(m::PavitoNonlinearModel, mipmodel, contsolution, jac_I, jac_J, jac_V, grad_f)
    # eval g and jac_g at MIP solution
    g = zeros(m.numconstr)
    MathProgBase.eval_g(m.d, g, contsolution[1:m.numvar])
    MathProgBase.eval_jac_g(m.d, jac_V, contsolution[1:m.numvar])

    # create rows corresponding to constraints in sparse format
    varidx_new = [zeros(Int, 0) for i in 1:m.numconstr]
    coef_new = [zeros(0) for i in 1:m.numconstr]

    for k in 1:length(jac_I)
        row = jac_I[k]
        push!(varidx_new[row], jac_J[k])
        push!(coef_new[row], jac_V[k])
    end

    # create constraint cuts
    for i in 1:m.numconstr
        if !m.constrlinear[i]
            # create supporting hyperplane
            if m.constrtype[i] == :(<=)
                val = g[i] - m.ub[i]
                new_rhs = m.ub[i] - g[i]
            else
                val = m.lb[i] - g[i]
                new_rhs = m.lb[i] - g[i]
            end

            for j in 1:length(varidx_new[i])
                new_rhs += coef_new[i][j] * contsolution[Int(varidx_new[i][j])]
            end

            if m.constrtype[i] == :(<=)
                if m.mip_solver_drives && m.oa_started
                    @lazyconstraint(m.cb, dot(coef_new[i], m.mip_x[varidx_new[i]]) <= new_rhs)
                else
                    @constraint(mipmodel, dot(coef_new[i], m.mip_x[varidx_new[i]]) <= new_rhs)
                end
            else
                if m.mip_solver_drives && m.oa_started
                    @lazyconstraint(m.cb, dot(coef_new[i], m.mip_x[varidx_new[i]]) >= new_rhs)
                else
                    @constraint(mipmodel, dot(coef_new[i], m.mip_x[varidx_new[i]]) >= new_rhs)
                end
            end
        end
    end

    # create obj cut
    if !m.objlinear
        f = MathProgBase.eval_f(m.d, contsolution[1:m.numvar])
        MathProgBase.eval_grad_f(m.d, grad_f, contsolution[1:m.numvar])
        if m.objsense == :Max
            f = -f
            grad_f = -grad_f
        end
        new_rhs = -f
        varidx = zeros(Int, m.numvar+1)
        for j in 1:m.numvar
            varidx[j] = j
            new_rhs += grad_f[j] * contsolution[j]
        end
        varidx[m.numvar+1] = m.numvar + 1
        grad_f[m.numvar+1] = -1.0

        if m.mip_solver_drives && m.oa_started
            @lazyconstraint(m.cb, dot(grad_f, m.mip_x[varidx]) <= new_rhs)
        else
            @constraint(mipmodel, dot(grad_f, m.mip_x[varidx]) <= new_rhs)
        end
    end

    return nothing
end

function constructoamodel(m::PavitoNonlinearModel)
    mipmodel = Model(solver=m.mip_solver)

    lb = [m.l; -1e6]
    ub = [m.u; 1e6]
    @variable(mipmodel, lb[i] <= x[i=1:m.numvar+1] <= ub[i])

    numintvar = 0
    for i in 1:m.numvar
        setcategory(x[i], m.vartypes[i])
        if (m.vartypes[i] == :Int) || (m.vartypes[i] == :Bin)
            numintvar += 1
        end
    end
    if numintvar == 0
        error("no variables of type integer or binary; call the conic continuous solver directly for pure continuous problems")
    end
    setcategory(x[m.numvar+1], :Cont)

    for i in 1:m.numconstr-m.numnlconstr
        if (m.A_lb[i] > -Inf) && (m.A_ub[i] < Inf)
            if m.A_lb[i] == m.A_ub[i]
                @constraint(mipmodel, m.A[i:i,:]*x[1:m.numvar] .== m.A_lb[i])
            else
                @constraint(mipmodel, m.A[i:i,:]*x[1:m.numvar] .>= m.A_lb[i])
                @constraint(mipmodel, m.A[i:i,:]*x[1:m.numvar] .<= m.A_ub[i])
            end
        elseif m.A_lb[i] > -Inf
            @constraint(mipmodel, m.A[i:i,:]*x[1:m.numvar] .>= m.A_lb[i])
        else
            @constraint(mipmodel, m.A[i:i,:]*x[1:m.numvar] .<= m.A_ub[i])
        end
    end

    c_new = vcat(((m.objsense == :Max) ? -m.c : m.c), (m.objlinear ? 0.0 : 1.0))
    @objective(mipmodel, Min, dot(c_new, x))

    return (x, mipmodel)
end

function checkinfeas(m::PavitoNonlinearModel, solution)
    g = zeros(m.numconstr)
    MathProgBase.eval_g(m.d, g, solution[1:m.numvar])
    max_infeas = 0.0
    for i in 1:m.numconstr
        max_infeas = max(max_infeas, g[i]-m.ub[i], m.lb[i]-g[i])
    end
    for i in 1:m.numvar
        max_infeas = max(max_infeas, solution[i]-m.u[i], m.l[i]-solution[i])
    end
    return max_infeas
end

# print objective gap information for iterative
function printgap(m::PavitoNonlinearModel, start)
    if m.log_level >= 1
        if (m.iterscbs == 1) || (m.log_level >= 2)
            @printf "\n%-5s | %-14s | %-14s | %-11s | %-11s\n" "Iter." "Best feasible" "Best bound" "Rel. gap" "Time (s)"
        end
        if m.objgap < 1000
            @printf "%5d | %+14.6e | %+14.6e | %11.3e | %11.3e\n" m.iterscbs m.fixsense*m.objval m.fixsense*m.objbound m.fixsense*m.objgap (time() - start)
        else
            @printf "%5d | %+14.6e | %+14.6e | %11s | %11.3e\n" m.iterscbs m.fixsense*m.objval m.fixsense*m.objbound (isnan(m.objgap) ? "Inf" : ">1000") (time() - start)
        end
        flush(stdout)
        flush(stderr)
    end
    return nothing
end
