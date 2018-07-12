#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 Nonlinear model object
=========================================================#

type PavitoNonlinearModel <: MathProgBase.AbstractNonlinearModel
    log_level::Int              # Verbosity flag: 0 for quiet, higher for basic solve info
    timeout::Float64            # Time limit for algorithm (in seconds)
    rel_gap::Float64            # Relative optimality gap termination condition
    mip_solver_drives::Bool     # Let MILP solver manage convergence ("branch and cut")
    mip_solver::MathProgBase.AbstractMathProgSolver # MILP solver
    cont_solver::MathProgBase.AbstractMathProgSolver # Continuous NLP solver

    solution::Vector{Float64}
    status::Symbol
    totaltime::Float64
    objval::Float64
    objbound::Float64
    iterations::Int

    numVar::Int
    numIntVar::Int
    numConstr::Int
    numNLConstr::Int
    A::SparseMatrixCSC{Float64,Int64}
    A_lb::Vector{Float64}
    A_ub::Vector{Float64}
    lb::Vector{Float64}
    ub::Vector{Float64}
    l::Vector{Float64}
    u::Vector{Float64}
    c::Vector{Float64}
    vartype::Vector{Symbol}
    constrtype::Vector{Symbol}
    constrlinear::Vector{Bool}
    objsense::Symbol
    objlinear::Bool

    mip_x::Vector{JuMP.Variable}
    d

    function PavitoNonlinearModel(log_level, timeout, rel_gap, mip_solver_drives, mip_solver, cont_solver)
        m = new()

        m.log_level = log_level
        m.timeout = timeout
        m.rel_gap = rel_gap
        m.mip_solver_drives = mip_solver_drives
        m.mip_solver = mip_solver
        m.cont_solver = cont_solver

        m.solution = Vector{Float64}()
        m.status = :NotLoaded
        m.totaltime = 0.0

        return m
    end
end


#=========================================================
 MathProgBase functions
=========================================================#

MathProgBase.setwarmstart!(m::PavitoNonlinearModel, x) = (m.solution = x)
MathProgBase.setvartype!(m::PavitoNonlinearModel, v::Vector{Symbol}) = (m.vartype = v)
#MathProgBase.setvarUB!(m::IpoptMathProgModel, v::Vector{Float64}) = (m.u = v)
#MathProgBase.setvarLB!(m::IpoptMathProgModel, v::Vector{Float64}) = (m.l = v)

function MathProgBase.loadproblem!(m::PavitoNonlinearModel, numVar, numConstr, l, u, lb, ub, sense, d)
    m.numVar = numVar
    m.numConstr = numConstr
    m.lb = lb
    m.ub = ub
    m.l = l
    m.u = u
    m.objsense = sense
    m.d = d
    m.vartype = fill(:Cont, numVar)

    MathProgBase.initialize(d, [:Grad,:Jac,:Hess])

    m.constrtype = Array{Symbol}(numConstr)
    for i = 1:numConstr
        if lb[i] > -Inf && ub[i] < Inf
            m.constrtype[i] = :(==)
        elseif lb[i] > -Inf
            m.constrtype[i] = :(>=)
        else
            m.constrtype[i] = :(<=)
        end
    end

    m.solution = fill(NaN, m.numVar)
    m.status = :Loaded

    return nothing
end

function MathProgBase.optimize!(m::PavitoNonlinearModel)
    start = time()

    cputime_nlp = 0.0
    cputime_mip = 0.0

    # if we haven't gotten a starting point (e.g., if acting as MIQP solver) assume zero is okay
    if any(isnan,m.solution)
        m.solution = zeros(length(m.solution))
    end

    populatelinearmatrix(m)

    mip_model = Model(solver=m.mip_solver)
    loadMIPModel(m, mip_model)

    for i in 1:m.numConstr
        if !(m.constrlinear[i]) && m.constrtype[i] == :(==)
            error("Nonlinear equality or two-sided constraints not accepted.")
        end
    end

    jac_I, jac_J = MathProgBase.jac_structure(m.d)
    jac_V = zeros(length(jac_I))
    grad_f = zeros(m.numVar+1)

    ini_nlp_model = MathProgBase.NonlinearModel(m.cont_solver)
    MathProgBase.loadproblem!(ini_nlp_model,
    m.numVar, m.numConstr, m.l, m.u, m.lb, m.ub, m.objsense, m.d)

    # pass in starting point
    MathProgBase.setwarmstart!(ini_nlp_model, m.solution[1:m.numVar])

    start_nlp = time()
    MathProgBase.optimize!(ini_nlp_model)
    cputime_nlp += time() - start_nlp

    ini_nlp_status = MathProgBase.status(ini_nlp_model)
    if ini_nlp_status == :Optimal || ini_nlp_status == :Suboptimal
        if m.numIntVar == 0
            m.solution = MathProgBase.getsolution(ini_nlp_model)
            m.objval = MathProgBase.getobjval(ini_nlp_model)
            m.status = ini_nlp_status
            return
        end
        separator = MathProgBase.getsolution(ini_nlp_model)
        addCuttingPlanes!(m, mip_model, separator, jac_I, jac_J, jac_V, grad_f, [], zeros(m.numVar+1))
    elseif ini_nlp_status == :Infeasible
        (m.log_level > 0) && println("Initial NLP Relaxation Infeasible.")
        m.status = :Infeasible
        return
    elseif ini_nlp_status == :Unbounded
        warn("Initial NLP Relaxation Unbounded.")
        m.status = :Unbounded
        return
    else
        warn("NLP Solver Failure.")
        m.status = :Error
        return
    end
    ini_nlp_objval = MathProgBase.getobjval(ini_nlp_model)

    (m.log_level > 0) && println("\nPavito started...\n")
    (m.log_level > 0) && println("MINLP has $(m.numVar) variables, $(m.numConstr - m.numNLConstr) linear constraints, $(m.numNLConstr) nonlinear constraints.")
    (m.log_level > 0) && @printf "Initial objective = %13.5f.\n\n" ini_nlp_objval

    m.status = :UserLimit
    m.objval = Inf
    iter = 0
    prev_mip_solution = fill(NaN,m.numVar)
    cut_added = false

    nlp_status = :Infeasible
    nlp_solution = zeros(m.numVar)

    function nonlinearcallback(cb)
        if cb != []
            mip_objval = -Inf
            mip_solution = MathProgBase.cbgetmipsolution(cb)[1:m.numVar+1]
        else
            m.objbound = mip_objval = getobjbound(mip_model)
            mip_solution = getvalue(m.mip_x)
        end
        (m.log_level > 2) && println("MIP Solution: $mip_solution")

        # solve NLP model for the MIP solution
        new_u = m.u
        new_l = m.l
        for i in 1:m.numVar
            if m.vartype[i] == :Int || m.vartype[i] == :Bin
                new_u[i] = mip_solution[i]
                new_l[i] = mip_solution[i]
            end
        end

        # set up ipopt to solve continuous relaxation
        nlp_model = MathProgBase.NonlinearModel(m.cont_solver)
        MathProgBase.loadproblem!(nlp_model,
        m.numVar, m.numConstr, new_l, new_u, m.lb, m.ub, m.objsense, m.d)

        # pass in starting point
        MathProgBase.setwarmstart!(nlp_model, mip_solution[1:m.numVar])

        l_inf = [new_l;zeros(m.numNLConstr)]
        u_inf = [new_u;Inf*ones(m.numNLConstr)]

        d_inf = InfeasibleNLPEvaluator(m.d, m.numConstr, m.numNLConstr, m.numVar, m.constrtype, m.constrlinear)
        inf_model = MathProgBase.NonlinearModel(m.cont_solver)
        MathProgBase.loadproblem!(inf_model,
        m.numVar+m.numNLConstr, m.numConstr, l_inf, u_inf, m.lb, m.ub, :Min, d_inf)

        # optimize the NLP problem
        start_nlp = time()
        MathProgBase.optimize!(nlp_model)
        cputime_nlp += time() - start_nlp
        nlp_status = MathProgBase.status(nlp_model)
        nlp_objval = (m.objsense == :Max ? -Inf : Inf)

        if nlp_status == :Optimal
            (m.log_level > 2) && println("NLP Solved")
            nlp_objval = MathProgBase.getobjval(nlp_model)
            nlp_solution = MathProgBase.getsolution(nlp_model)
            separator = copy(nlp_solution)
            (m.log_level > 2) && println("NLP Solution: $separator")

            # keep track of best integer solution
            if m.objsense == :Max
                if nlp_objval > -m.objval
                    m.objval = -nlp_objval
                    m.solution = separator[1:m.numVar]
                end
            else
                if nlp_objval < m.objval
                    m.objval = nlp_objval
                    m.solution = separator[1:m.numVar]
                end
            end
        else
            # create the warm start solution for inf model
            inf_initial_solution = zeros(m.numVar + m.numNLConstr);
            inf_initial_solution[1:m.numVar] = mip_solution[1:m.numVar]
            g = zeros(m.numConstr)
            MathProgBase.eval_g(m.d, g, inf_initial_solution[1:m.numVar])
            k = 1

            for i in 1:m.numConstr
                if !m.constrlinear[i]
                    if m.constrtype[i] == :(<=)
                        val = g[i] - m.ub[i]
                    else
                        val = m.lb[i] - g[i]
                    end
                    if val > 0
                        # because the sign of the slack changes if the constraint direction change
                        inf_initial_solution[m.numVar + k] = val
                    else
                        inf_initial_solution[m.numVar + k] = 0.0
                    end
                    k += 1
                end
            end

            MathProgBase.setwarmstart!(inf_model, inf_initial_solution)
            (m.log_level > 2) && println("NLP Infeasible")
            start_nlp = time()
            MathProgBase.optimize!(inf_model)
            cputime_nlp += time() - start_nlp

            inf_model_status = MathProgBase.status(inf_model)
            if inf_model_status != :Optimal && inf_model_status != :Suboptimal
                warn("NLP Solver Failure.")
                m.status = :Error
                return
            end

            (m.log_level > 2) && println("INF NLP Solved")
            separator = MathProgBase.getsolution(inf_model)
            (m.log_level > 2) && println("INF NLP Solution: $separator")
        end

        # add supporting hyperplanes
        cycle_indicator = (!m.mip_solver_drives ? compareIntegerSolutions(m, prev_mip_solution, mip_solution) : false)
        # objval and mip_objval are always in minimization sense
        optimality_gap = m.objval - mip_objval
        primal_infeasibility = checkInfeasibility(m, mip_solution)
        OA_infeasibility = 0.0

        if (optimality_gap > (abs(mip_objval) + 1e-5)*m.rel_gap && !cycle_indicator) || cb != []
            OA_infeasibility = addCuttingPlanes!(m, mip_model, separator, jac_I, jac_J, jac_V, grad_f, cb, mip_solution)
            #(m.cut_switch > 0) && addCuttingPlanes!(m, mip_model, mip_solution, jac_I, jac_J, jac_V, grad_f, cb, mip_solution)
            cut_added = true
        else
            if optimality_gap < (abs(mip_objval) + 1e-5)*m.rel_gap
                (m.log_level > 1) && println("MINLP Solved")
                m.status = :Optimal
                m.iterations = iter
                (m.log_level > 1) && println("Number of OA iterations: $iter")
            else
                @assert cycle_indicator
                m.status = :Suboptimal
            end
        end

        fixsense = (m.objsense == :Max) ? -1 : 1
        (m.log_level > 0) && !m.mip_solver_drives && OAprintLevel(iter, fixsense*mip_objval, nlp_objval, optimality_gap, fixsense*m.objval, primal_infeasibility, OA_infeasibility)
        (cycle_indicator && m.status != :Optimal) && warn("Mixed-integer cycling detected, terminating Pavito...")
    end

    function heuristiccallback(cb)
        if nlp_status == :Optimal
            for i = 1:m.numVar
                setsolutionvalue(cb, m.mip_x[i], nlp_solution[i])
            end
            addsolution(cb)
        end
    end

    if m.mip_solver_drives
        addlazycallback(mip_model, nonlinearcallback)
        addheuristiccallback(mip_model, heuristiccallback)
        m.status = solve(mip_model, suppress_warnings=true)
        m.objbound = getobjbound(mip_model)
    else
        (m.log_level > 0) && println("Iteration   MIP Objective     NLP Objective   Optimality Gap   Best Solution    Primal Inf.      OA Inf.")
        # check if we've been provided with an initial feasible solution
        if all(isfinite,m.solution) && checkInfeasibility(m, m.solution) < 1e-5
            if m.objsense == :Max
                m.objval = -MathProgBase.eval_f(m.d, m.solution)
            else
                m.objval = MathProgBase.eval_f(m.d, m.solution)
            end
        end
        while (time() - start) < m.timeout
            flush(STDOUT)
            cut_added = false
            # warmstart mip from upper bound
            if !any(isnan,m.solution) && !isempty(m.solution)
                if applicable(MathProgBase.setwarmstart!, internalmodel(mip_model), m.solution)
                    # extend solution with the objective variable
                    MathProgBase.setwarmstart!(internalmodel(mip_model), [m.solution; m.objval])
                end
            end

            # solve MIP model
            start_mip = time()
            mip_status = solve(mip_model, suppress_warnings=true)
            cputime_mip += time() - start_mip

            if mip_status == :Infeasible || mip_status == :InfeasibleOrUnbounded
                (m.log_level > 1) && println("MIP Infeasible")
                m.status = :Infeasible
                break
            else
                (m.log_level > 1) && println("MIP Status: $mip_status")
            end
            mip_solution = getvalue(m.mip_x)

            nonlinearcallback([])

            if cut_added == false
                break
            end

            prev_mip_solution = mip_solution
            iter += 1
        end
    end

    if m.objsense == :Max
        m.objval = -m.objval
        m.objbound = -m.objbound
    end
    m.totaltime = time()-start
    if m.log_level > 0
        println("\nPavito finished...\n")
        @printf "Status            = %13s.\n" m.status
        (m.status == :Optimal) && @printf "Optimum objective = %13.5f.\n" m.objval
        !m.mip_solver_drives && @printf "Iterations        = %13d.\n" iter
        @printf "Total time        = %13.5f sec.\n" m.totaltime
        @printf "MIP total time    = %13.5f sec.\n" cputime_mip
        @printf "NLP total time    = %13.5f sec.\n" cputime_nlp
    end

    return nothing
end

MathProgBase.status(m::PavitoNonlinearModel) = m.status
MathProgBase.getobjval(m::PavitoNonlinearModel) = m.objval
MathProgBase.getobjbound(m::PavitoNonlinearModel) = m.objbound
MathProgBase.getsolution(m::PavitoNonlinearModel) = m.solution
MathProgBase.getsolvetime(m::PavitoNonlinearModel) = m.totaltime


#=========================================================
 NLP utilities
=========================================================#

type InfeasibleNLPEvaluator <: MathProgBase.AbstractNLPEvaluator
    d
    numConstr::Int
    numNLConstr::Int
    numVar::Int
    constrtype::Vector{Symbol}
    constrlinear::Vector{Bool}
end

function MathProgBase.eval_f(d::InfeasibleNLPEvaluator, x)
    retval = 0.0
    # sum the slacks
    for i in d.numVar+1:length(x)
        retval += x[i]
    end
    return retval
end

function MathProgBase.eval_grad_f(d::InfeasibleNLPEvaluator, g, x)
    g[:] = [zeros(d.numVar); ones(d.numNLConstr)]
    return nothing
end

function MathProgBase.eval_g(d::InfeasibleNLPEvaluator, g, x)
    MathProgBase.eval_g(d.d, g, x[1:d.numVar])
    k = 1
    for i in 1:d.numConstr
        d.constrlinear[i] && continue
        if d.constrtype[i] == :(<=)
            g[i] -= x[k+d.numVar]
        else
            g[i] += x[k+d.numVar]
        end
        k += 1
    end
    return nothing
end

function MathProgBase.jac_structure(d::InfeasibleNLPEvaluator)
    I, J = MathProgBase.jac_structure(d.d)
    I_new = copy(I)
    J_new = copy(J)
    k = 1
    for i in 1:(d.numConstr)
        d.constrlinear[i] && continue
        push!(I_new, i); push!(J_new, k+d.numVar);
        k += 1
    end
    return I_new, J_new
end

function MathProgBase.eval_jac_g(d::InfeasibleNLPEvaluator, J, x)
    MathProgBase.eval_jac_g(d.d, J, x[1:d.numVar])
    k = length(J) - d.numNLConstr + 1
    for i in 1:d.numConstr
        d.constrlinear[i] && continue
        if d.constrtype[i] == :(<=)
            J[k] = -(1.0)
        else
            J[k] = 1.0
        end
        k += 1
    end
    return nothing
end

MathProgBase.eval_hesslag(d::InfeasibleNLPEvaluator, H, x, σ, μ) = MathProgBase.eval_hesslag(d.d, H, x[1:d.numVar], 0.0, μ)

MathProgBase.hesslag_structure(d::InfeasibleNLPEvaluator) = MathProgBase.hesslag_structure(d.d)
MathProgBase.initialize(d::InfeasibleNLPEvaluator, requested_features::Vector{Symbol}) =
MathProgBase.initialize(d.d, requested_features)
MathProgBase.features_available(d::InfeasibleNLPEvaluator) = [:Grad,:Jac,:Hess]

MathProgBase.isobjlinear(d::InfeasibleNLPEvaluator) = true
MathProgBase.isobjquadratic(d::InfeasibleNLPEvaluator) = true
MathProgBase.isconstrlinear(d::InfeasibleNLPEvaluator, i::Int) = MathProgBase.isconstrlinear(d.d, i)
MathProgBase.obj_expr(d::InfeasibleNLPEvaluator) = MathProgBase.obj_expr(d.d)
MathProgBase.constr_expr(d::InfeasibleNLPEvaluator, i::Int) = MathProgBase.constr_expr(d.d, i)


#=========================================================
 Algorithmic utilities
=========================================================#

function populatelinearmatrix(m::PavitoNonlinearModel)
    # set up map of linear rows
    constrlinear = Array{Bool}(m.numConstr)
    numlinear = 0
    constraint_to_linear = fill(-1,m.numConstr)
    for i = 1:m.numConstr
        constrlinear[i] = MathProgBase.isconstrlinear(m.d, i)
        if constrlinear[i]
            numlinear += 1
            constraint_to_linear[i] = numlinear
        end
    end
    m.numNLConstr = m.numConstr - numlinear

    # extract sparse jacobian structure
    jac_I, jac_J = MathProgBase.jac_structure(m.d)

    # evaluate jacobian at x = 0
    c = zeros(m.numVar)
    x = m.solution
    jac_V = zeros(length(jac_I))
    MathProgBase.eval_jac_g(m.d, jac_V, x)
    MathProgBase.eval_grad_f(m.d, c, x)
    m.objlinear = MathProgBase.isobjlinear(m.d)
    if m.objlinear
        (m.log_level > 0) && println("Objective function is linear")
        m.c = c
    else
        (m.log_level > 0) && println("Objective function is nonlinear")
        m.c = zeros(m.numVar)
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
        push!(A_I,row); push!(A_J, jac_J[k]); push!(A_V, jac_V[k])
    end

    m.A = sparse(A_I, A_J, A_V, numlinear, m.numVar)

    # g(x) might have a constant, i.e., a'x + b
    # find b
    constraint_value = zeros(m.numConstr)
    MathProgBase.eval_g(m.d, constraint_value, x)
    b = constraint_value[constrlinear] - m.A * x
    # so linear constraints are of the form lb ≤ a'x + b ≤ ub

    # set up A_lb and A_ub vectors
    m.A_lb = m.lb[constrlinear] - b
    m.A_ub = m.ub[constrlinear] - b

    # linear parts
    m.constrlinear = constrlinear

    return nothing
end

function addCuttingPlanes!(m::PavitoNonlinearModel, mip_model, separator, jac_I, jac_J, jac_V, grad_f, cb, mip_solution)
    max_violation = -1e+5
    # eval g and jac_g at infeasible MIP solution
    g = zeros(m.numConstr)
    MathProgBase.eval_g(m.d, g, separator[1:m.numVar])
    MathProgBase.eval_jac_g(m.d, jac_V, separator[1:m.numVar])

    # create rows corresponding to constraints in sparse format
    varidx_new = [zeros(Int, 0) for i in 1:m.numConstr]
    coef_new = [zeros(0) for i in 1:m.numConstr]

    for k in 1:length(jac_I)
        row = jac_I[k]
        push!(varidx_new[row], jac_J[k]); push!(coef_new[row], jac_V[k])
    end

    # create constraint cuts
    for i in 1:m.numConstr
        if m.constrtype[i] == :(<=)
            val = g[i] - m.ub[i]
        else
            val = m.lb[i] - g[i]
        end

        lin = m.constrlinear[i]
        (m.log_level > 2) && println("Constraint $i value $val linear $lin")

        if !lin
            # create supporting hyperplane
            (m.log_level > 2) && println("Create supporting hyperplane for constraint $i")
            local new_rhs::Float64

            if m.constrtype[i] == :(<=)
                new_rhs = m.ub[i] - g[i]
            else
                new_rhs = m.lb[i] - g[i]
            end
            for j = 1:length(varidx_new[i])
                new_rhs += coef_new[i][j] * separator[Int(varidx_new[i][j])]
            end

            (m.log_level > 2) && println("varidx $(varidx_new[i])")
            (m.log_level > 2) && println("coef $(coef_new[i])")
            (m.log_level > 2) && println("rhs $new_rhs")

            if m.constrtype[i] == :(<=)
                if cb != []
                    @lazyconstraint(cb, dot(coef_new[i], m.mip_x[varidx_new[i]]) <= new_rhs)
                else
                    @constraint(mip_model, dot(coef_new[i], m.mip_x[varidx_new[i]]) <= new_rhs)
                end
                viol = vecdot(coef_new[i], mip_solution[varidx_new[i]]) - new_rhs
                if viol > max_violation
                    max_violation = viol
                end
            else
                if cb != []
                    @lazyconstraint(cb, dot(coef_new[i], m.mip_x[varidx_new[i]]) >= new_rhs)
                else
                    @constraint(mip_model, dot(coef_new[i], m.mip_x[varidx_new[i]]) >= new_rhs)
                end
                viol = new_rhs - vecdot(coef_new[i], mip_solution[varidx_new[i]])
                if viol > max_violation
                    max_violation = viol
                end
            end
        end
    end

    # create obj cut
    if !(m.objlinear)
        (m.log_level > 2) && println("Create supporting hyperplane for objective f(x) <= t")
        f = MathProgBase.eval_f(m.d, separator[1:m.numVar])
        MathProgBase.eval_grad_f(m.d, grad_f, separator[1:m.numVar])
        if m.objsense == :Max
            f = -f
            grad_f = -grad_f
        end
        new_rhs = -f
        varidx = zeros(Int, m.numVar+1)
        for j = 1:m.numVar
            varidx[j] = j
            new_rhs += grad_f[j] * separator[j]
        end
        varidx[m.numVar+1] = m.numVar+1
        grad_f[m.numVar+1] = -(1.0)
        (m.log_level > 2) && println("varidx $(varidx)")
        (m.log_level > 2) && println("coef $(grad_f)")
        (m.log_level > 2) && println("rhs $new_rhs")

        if cb != []
            @lazyconstraint(cb, dot(grad_f, m.mip_x[varidx]) <= new_rhs)
        else
            @constraint(mip_model, dot(grad_f, m.mip_x[varidx]) <= new_rhs)
        end
        viol = vecdot(grad_f, mip_solution[varidx]) - new_rhs
        if viol > max_violation
            max_violation = viol
        end
    end

    return max_violation
end

function loadMIPModel(m::PavitoNonlinearModel, mip_model)
    lb = [m.l; -1e6]
    ub = [m.u; 1e6]
    @variable(mip_model, lb[i] <= x[i=1:m.numVar+1] <= ub[i])
    numIntVar = 0
    for i = 1:m.numVar
        setcategory(x[i], m.vartype[i])
        if m.vartype[i] == :Int || m.vartype[i] == :Bin
            numIntVar += 1
        end
    end
    if numIntVar == 0
        error("No variables of type integer or binary; call the conic continuous solver directly for pure continuous problems")
    end
    setcategory(x[m.numVar+1], :Cont)

    for i = 1:m.numConstr-m.numNLConstr
        if m.A_lb[i] > -Inf && m.A_ub[i] < Inf
            if m.A_lb[i] == m.A_ub[i]
                @constraint(mip_model, m.A[i:i,:]*x[1:m.numVar] .== m.A_lb[i])
            else
                @constraint(mip_model, m.A[i:i,:]*x[1:m.numVar] .>= m.A_lb[i])
                @constraint(mip_model, m.A[i:i,:]*x[1:m.numVar] .<= m.A_ub[i])
            end
        elseif m.A_lb[i] > -Inf
            @constraint(mip_model, m.A[i:i,:]*x[1:m.numVar] .>= m.A_lb[i])
        else
            @constraint(mip_model, m.A[i:i,:]*x[1:m.numVar] .<= m.A_ub[i])
        end
    end
    c_new = [m.objsense == :Max ? -m.c : m.c; m.objlinear ? 0.0 : 1.0]
    @objective(mip_model, Min, dot(c_new, x))

    m.mip_x = x
    m.numIntVar = numIntVar

    return nothing
end

function checkInfeasibility(m::PavitoNonlinearModel, solution)
    g = zeros(m.numConstr)
    MathProgBase.eval_g(m.d, g, solution[1:m.numVar])
    max_infeas = 0.0
    for i = 1:m.numConstr
        max_infeas = max(max_infeas, g[i]-m.ub[i], m.lb[i]-g[i])
    end
    for i = 1:m.numVar
        max_infeas = max(max_infeas, solution[i]-m.u[i], m.l[i]-solution[i])
    end
    return max_infeas
end

function compareIntegerSolutions(m::PavitoNonlinearModel, sol1, sol2)
    int_ind = filter(i->m.vartype[i] == :Int || m.vartype[i] == :Bin, 1:m.numVar)
    return round.(sol1[int_ind]) == round.(sol2[int_ind])
end

function OAprintLevel(iter, mip_objval, conic_objval, optimality_gap, best_objval, primal_infeasibility, OA_infeasibility)
    if abs(conic_objval) == Inf || isnan(conic_objval)
        conic_objval_str = @sprintf "%s" "              "
    else
        conic_objval_str = @sprintf "%+.7e" conic_objval
    end
    if abs(optimality_gap) == Inf || isnan(optimality_gap)
        optimality_gap_str = @sprintf "%s" "              "
    else
        optimality_gap_str = @sprintf "%+.7e" optimality_gap
    end
    if abs(best_objval) == Inf || isnan(best_objval)
        best_objval_str = @sprintf "%s" "              "
    else
        best_objval_str = @sprintf "%+.7e" best_objval
    end

    @printf "%9d   %+.7e   %s   %s   %s   %+.7e   %+.7e\n" iter mip_objval conic_objval_str optimality_gap_str best_objval_str primal_infeasibility OA_infeasibility

    return nothing
end
