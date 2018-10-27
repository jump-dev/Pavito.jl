#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 conic model unit tests
=========================================================#

# take a CBF (conic benchmark format) file and solve, redirecting output
function solve_cbf(testname, probname, solver, redirect)
    flush(stdout)
    flush(stderr)
    @printf "%-30s... " testname
    start_time = time()

    dat = ConicBenchmarkUtilities.readcbfdata("cbf/$(probname).cbf")
    (c, A, b, con_cones, var_cones, vartypes, sense, objoffset) = ConicBenchmarkUtilities.cbftompb(dat)
    if sense == :Max
        c = -c
    end
    flush(stdout)
    flush(stderr)

    m = MathProgBase.ConicModel(solver)

    if redirect
        mktemp() do path,io
            out = stdout
            err = stderr
            redirect_stdout(io)
            redirect_stderr(io)

            MathProgBase.loadproblem!(m, c, A, b, con_cones, var_cones)
            MathProgBase.setvartype!(m, vartypes)
            MathProgBase.optimize!(m)

            flush(io)
            redirect_stdout(out)
            redirect_stderr(err)
        end
    else
        MathProgBase.loadproblem!(m, c, A, b, con_cones, var_cones)
        MathProgBase.setvartype!(m, vartypes)
        MathProgBase.optimize!(m)
    end
    flush(stdout)
    flush(stderr)

    status = MathProgBase.status(m)
    solve_time = MathProgBase.getsolvetime(m)
    if sense == :Max
        objval = -MathProgBase.getobjval(m)
        objbound = -MathProgBase.getobjbound(m)
    else
        objval = MathProgBase.getobjval(m)
        objbound = MathProgBase.getobjbound(m)
    end
    sol = MathProgBase.getsolution(m)
    rt_time = time() - start_time
    @printf ":%-16s %5.2f s\n" status rt_time
    flush(stdout)
    flush(stderr)

    return (status, solve_time, objval, objbound, sol)
end

# second-order cone model tests
function run_soc(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    testname = "SOC optimal"
    probname = "soc_optimal"
    @testset "$testname" begin
        solver = PavitoSolver(timeout=120., mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(sol[1], 3, atol=TOL)
        @test isapprox(objval, -9, atol=TOL)
    end

    testname = "SOC infeasible"
    probname = "soc_infeasible"
    @testset "$testname" begin
        solver = PavitoSolver(timeout=120., mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Infeasible
    end

    testname = "SOCRot optimal"
    probname = "socrot_optimal"
    @testset "$testname" begin
        solver = PavitoSolver(timeout=120., mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -9, atol=TOL)
        @test isapprox(objbound, -9, atol=TOL)
        @test isapprox(sol, [1.5, 3, 3, 3], atol=TOL)
    end

    testname = "SOCRot infeasible"
    probname = "socrot_infeasible"
    @testset "$testname" begin
        solver = PavitoSolver(timeout=120., mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Infeasible
    end

    testname = "Equality constraint"
    probname = "soc_equality"
    @testset "$testname" begin
        solver = PavitoSolver(timeout=120., mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -sqrt(2), atol=TOL)
        @test isapprox(objbound, -sqrt(2), atol=TOL)
        @test isapprox(sol, [1, 1/sqrt(2), 1/sqrt(2)], atol=TOL)
    end

    testname = "Zero cones"
    probname = "soc_zero"
    @testset "$testname" begin
        solver = PavitoSolver(timeout=120., mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -sqrt(2), atol=TOL)
        @test isapprox(objbound, -sqrt(2), atol=TOL)
        @test isapprox(sol, [1, 1/sqrt(2), 1/sqrt(2), 0, 0], atol=TOL)
    end

    testname = "SOC infeasible binary"
    probname = "soc_infeasible2"
    @testset "$testname" begin
        solver = PavitoSolver(timeout=120., mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Infeasible
    end
end

# exponential and second-order cone tests
function run_expsoc(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    testname = "Exp optimal"
    probname = "exp_optimal"
    @testset "$testname" begin
        solver = PavitoSolver(timeout=120., mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -8, atol=TOL)
        @test isapprox(objbound, -8, atol=TOL)
        @test isapprox(sol[1:2], [2, 2], atol=TOL)
    end

    testname = "ExpSOC optimal"
    probname = "expsoc_optimal"
    @testset "$testname" begin
        solver = PavitoSolver(timeout=120., mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.609438, atol=TOL)
        @test isapprox(objbound, -7.609438, atol=TOL)
        @test isapprox(sol[1:2], [2, 1.609438], atol=TOL)
    end

    testname = "ExpSOC optimal 3"
    probname = "expsoc_optimal3"
    @testset "$testname" begin
        solver = PavitoSolver(timeout=120., mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7, atol=TOL)
        @test isapprox(objbound, -7, atol=TOL)
        @test isapprox(sol[1:2], [1, 2], atol=TOL)
    end

    testname = "Exp large (gatesizing)"
    probname = "exp_gatesizing"
    @testset "$testname" begin
        solver = PavitoSolver(timeout=120., mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, 8.33333, atol=TOL)
        @test isapprox(objbound, 8.33333, atol=TOL)
        @test isapprox(exp.(sol[1:7]), [2, 3, 3, 3, 2, 3, 3], atol=TOL)
    end

    testname = "Exp large 2 (Ising)"
    probname = "exp_ising"
    @testset "$testname" begin
        solver = PavitoSolver(timeout=120., mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, 0.696499, atol=TOL)
        @test isapprox(objbound, 0.696499, atol=TOL)
        @test isapprox(sol[end-8:end], [0, 0, 1, 0, 0, 0, 2, 1, 0], atol=TOL)
    end
end
