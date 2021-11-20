#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 NLP model unit tests
=========================================================#

# take a JuMP model and solve, redirecting output
function solve_jump(
    testname::String,
    m::JuMP.Model,
    redirect::Bool,
)
    flush(stdout)
    flush(stderr)
    @printf "%-30s... " testname
    start_time = time()

    if redirect
        mktemp() do path, io
            out = stdout
            err = stderr
            redirect_stdout(io)
            redirect_stderr(io)

            status = try
                JuMP.optimize!(m)
                MOI.get(m, MOI.TerminationStatus())
            catch e
                e
            end

            flush(io)
            redirect_stdout(out)
            redirect_stderr(err)
        end
    else
        status = begin
            JuMP.optimize!(m)
            MOI.get(m, MOI.TerminationStatus())
        end
    end
    flush(stdout)
    flush(stderr)

    rt_time = time() - start_time
    if isa(status, ErrorException)
        @printf ":%-16s %5.2f s\n" "ErrorException" rt_time
    else
        @printf ":%-16s %5.2f s\n" status rt_time
    end

    flush(stdout)
    flush(stderr)

    return status
end

# quadratic program tests
function run_qp(
    mip_solver_drives::Bool,
    mip_solver,
    cont_solver,
    redirect::Bool,
)
    solver = JuMP.optimizer_with_attributes(
        Pavito.Optimizer,
        "timeout" => 120.0,
        "mip_solver_drives" => mip_solver_drives,
        "mip_solver" => mip_solver,
        "cont_solver" => cont_solver,
        "log_level" => (redirect ? 0 : 3)
    )

    testname = "QP optimal"
    @testset "$testname" begin
        m = JuMP.Model(solver)

        JuMP.@variable(m, x >= 0, Int)
        JuMP.@variable(m, y >= 0)
        JuMP.@variable(m, 0 <= u <= 10, Int)
        JuMP.@variable(m, w == 1)

        JuMP.@objective(m, Min, -3x - y)

        JuMP.@constraint(m, 3x + 10 <= 20)
        JuMP.@constraint(m, y^2 <= u*w)

        status = solve_jump(testname, m, redirect)

        @test status == MOI.OPTIMAL
        @test isapprox(JuMP.objective_value(m), -12.162277, atol=TOL)
        @test isapprox(JuMP.objective_bound(m), -12.162277, atol=TOL)
        @test isapprox(JuMP.value(x), 3, atol=TOL)
        @test isapprox(JuMP.value(y), 3.162277, atol=TOL)
    end

    testname = "QP maximize"
    @testset "$testname" begin
        m = JuMP.Model(solver)

        JuMP.@variable(m, x >= 0, Int)
        JuMP.@variable(m, y >= 0)
        JuMP.@variable(m, 0 <= u <= 10, Int)
        JuMP.@variable(m, w == 1)

        JuMP.@objective(m, Max, 3x + y)

        JuMP.@constraint(m, 3x + 2y + 10 <= 20)
        JuMP.@constraint(m, x^2 <= u*w)

        status = solve_jump(testname, m, redirect)

        @test status == MOI.OPTIMAL
        @test isapprox(JuMP.objective_value(m), 9.5, atol=TOL)
        @test isapprox(JuMP.objective_bound(m), 9.5, atol=TOL)
        @test isapprox(JuMP.value(x), 3, atol=TOL)
        @test isapprox(JuMP.value(y), 0.5, atol=TOL)
    end
end

# NLP model tests
function run_nlp(
    mip_solver_drives::Bool,
    mip_solver,
    cont_solver,
    redirect::Bool,
)
    solver = JuMP.optimizer_with_attributes(
        Pavito.Optimizer,
        "timeout" => 120.0,
        "mip_solver_drives" => mip_solver_drives,
        "mip_solver" => mip_solver,
        "cont_solver" => cont_solver,
        "log_level" => (redirect ? 0 : 3)
    )

    testname = "Nonconvex error"
    @testset "$testname" begin
        m = JuMP.Model(solver)

        JuMP.@variable(m, x >= 0, start = 1, Int)
        JuMP.@variable(m, y >= 0, start = 1)

        JuMP.@objective(m, Min, -3x - y)

        JuMP.@constraint(m, 3x + 2y + 10 <= 20)
        JuMP.@NLconstraint(m, 8 <= x^2 <= 10)

        status = solve_jump(testname, m, redirect)

        @test isa(status, ErrorException)
    end

    testname = "Optimal"
    @testset "$testname" begin
        m = JuMP.Model(solver)

        JuMP.@variable(m, x >= 0, start = 1, Int)
        JuMP.@variable(m, y >= 0, start = 1)

        JuMP.@objective(m, Min, -3x - y)

        JuMP.@constraint(m, 3x + 2y + 10 <= 20)
        JuMP.@constraint(m, x >= 1)
        JuMP.@NLconstraint(m, x^2 <= 5)
        JuMP.@NLconstraint(m, exp(y) + x <= 7)

        status = solve_jump(testname, m, redirect)

        @test status == MOI.OPTIMAL
        @test isapprox(JuMP.value(x), 2.0)
    end

    testname = "Infeasible 1"
    @testset "$testname" begin
        m = JuMP.Model(solver)

        JuMP.@variable(m, x >= 0, start = 1, Int)
        JuMP.@variable(m, y >= 0, start = 1)

        JuMP.@objective(m, Min, -3x - y)

        JuMP.@constraint(m, 3x + 2y + 10 <= 20)
        JuMP.@NLconstraint(m, x^2 >= 9)
        JuMP.@NLconstraint(m, exp(y) + x <= 2)

        status = solve_jump(testname, m, redirect)

        @test status in [MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE]
    end

    testname = "Infeasible 2"
    @testset "$testname" begin
        m = JuMP.Model(solver)

        JuMP.@variable(m, x >= 0, start = 1, Int)
        JuMP.@variable(m, y >= 0, start = 1)

        JuMP.@objective(m, Min, -3x - y)

        JuMP.@constraint(m, 3x + 2y + 10 <= 20)
        JuMP.@constraint(m, 6x + 5y >= 30)
        JuMP.@NLconstraint(m, x^2 >= 8)
        JuMP.@NLconstraint(m, exp(y) + x <= 7)

        status = solve_jump(testname, m, redirect)

        @test status in [MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE]
    end

    testname = "Continuous error"
    @testset "$testname" begin
        m = JuMP.Model(solver)

        JuMP.@variable(m, x >= 0, start = 1)
        JuMP.@variable(m, y >= 0, start = 1)

        JuMP.@objective(m, Min, -3x - y)

        JuMP.@constraint(m, 3x + 2y + 10 <= 20)
        JuMP.@constraint(m, x >= 1)

        JuMP.@NLconstraint(m, x^2 <= 5)
        JuMP.@NLconstraint(m, exp(y) + x <= 7)

        status = solve_jump(testname, m, redirect)

        @test isa(status, ErrorException)
    end

    testname = "Maximization"
    @testset "$testname" begin
        m = JuMP.Model(solver)

        JuMP.@variable(m, x >= 0, start = 1, Int)
        JuMP.@variable(m, y >= 0, start = 1)

        JuMP.@objective(m, Max, 3x + y)

        JuMP.@constraint(m, 3x + 2y + 10 <= 20)
        JuMP.@NLconstraint(m, x^2 <= 9)

        status = solve_jump(testname, m, redirect)

        @test status == MOI.OPTIMAL
        @test isapprox(JuMP.objective_value(m), 9.5, atol=TOL)
    end

    testname = "Nonlinear objective"
    @testset "$testname" begin
        m = JuMP.Model(solver)

        JuMP.@variable(m, x >= 0, start = 1, Int)
        JuMP.@variable(m, y >= 0, start = 1)

        JuMP.@objective(m, Max, -x^2 - y)

        JuMP.@constraint(m, x + 2y >= 4)
        JuMP.@NLconstraint(m, x^2 <= 9)

        status = solve_jump(testname, m, redirect)

        @test status == MOI.OPTIMAL
        @test isapprox(JuMP.objective_value(m), -2.0, atol=TOL)
        @test isapprox(JuMP.objective_bound(m), -2.0, atol=TOL)
    end
end
