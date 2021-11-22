#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 NLP model unit tests
=========================================================#

# quadratic program tests
function run_qp(
    mip_solver_drives::Bool,
    mip_solver,
    cont_solver,
    log_level::Int,
    TOL::Real,
)
    solver = JuMP.optimizer_with_attributes(
        Pavito.Optimizer,
        "timeout" => 120.0,
        "mip_solver_drives" => mip_solver_drives,
        "mip_solver" => mip_solver,
        "cont_solver" => cont_solver,
        "log_level" => log_level,
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
        JuMP.@constraint(m, y^2 <= u * w)

        JuMP.optimize!(m)
        status = MOI.get(m, MOI.TerminationStatus())

        @test status == MOI.LOCALLY_SOLVED
        @test isapprox(JuMP.objective_value(m), -12.162277, atol = TOL)
        @test isapprox(JuMP.objective_bound(m), -12.162277, atol = TOL)
        @test isapprox(JuMP.value(x), 3, atol = TOL)
        @test isapprox(JuMP.value(y), 3.162277, atol = TOL)
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
        JuMP.@constraint(m, x^2 <= u * w)

        JuMP.optimize!(m)
        status = MOI.get(m, MOI.TerminationStatus())

        @test status == MOI.LOCALLY_SOLVED
        @test isapprox(JuMP.objective_value(m), 9.5, atol = TOL)
        @test isapprox(JuMP.objective_bound(m), 9.5, atol = TOL)
        @test isapprox(JuMP.value(x), 3, atol = TOL)
        @test isapprox(JuMP.value(y), 0.5, atol = TOL)
    end

    testname = "QP infeasible"
    @testset "$testname" begin
        m = JuMP.Model(solver)

        JuMP.@variable(m, x >= 0, Int)

        JuMP.@objective(m, Max, x)

        JuMP.@constraint(m, x^2 <= 3.9)
        JuMP.@constraint(m, x >= 1.1)

        JuMP.optimize!(m)
        status = MOI.get(m, MOI.TerminationStatus())

        @test status in [MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE]
    end
end

# NLP model tests
function run_nlp(
    mip_solver_drives::Bool,
    mip_solver,
    cont_solver,
    log_level::Int,
    TOL::Real,
)
    solver = JuMP.optimizer_with_attributes(
        Pavito.Optimizer,
        "timeout" => 120.0,
        "mip_solver_drives" => mip_solver_drives,
        "mip_solver" => mip_solver,
        "cont_solver" => cont_solver,
        "log_level" => log_level,
    )

    testname = "Nonconvex error"
    @testset "$testname" begin
        m = JuMP.Model(solver)

        JuMP.@variable(m, x >= 0, start = 1, Int)
        JuMP.@variable(m, y >= 0, start = 1)

        JuMP.@objective(m, Min, -3x - y)

        JuMP.@constraint(m, 3x + 2y + 10 <= 20)
        JuMP.@NLconstraint(m, 8 <= x^2 <= 10)

        @test_throws ErrorException JuMP.optimize!(m)
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

        JuMP.optimize!(m)
        status = MOI.get(m, MOI.TerminationStatus())

        @test status == MOI.LOCALLY_SOLVED
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

        JuMP.optimize!(m)
        status = MOI.get(m, MOI.TerminationStatus())

        @test status in [MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE]
    end

    testname = "Infeasible 2"
    @testset "$testname" begin
        m = JuMP.Model(solver)

        JuMP.@variable(m, x >= 0, start = 1, Int)

        JuMP.@objective(m, Max, x)

        JuMP.@NLconstraint(m, log(x) >= 0.75)
        JuMP.@constraint(m, x <= 2.9)

        JuMP.optimize!(m)
        status = MOI.get(m, MOI.TerminationStatus())

        @test status in [MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE]
    end

    testname = "Continuous problem"
    @testset "$testname" begin
        m = JuMP.Model(solver)

        JuMP.@variable(m, x >= 0, start = 1)
        JuMP.@variable(m, y >= 0, start = 1)

        JuMP.@objective(m, Min, -3x - y)

        JuMP.@constraint(m, 3x + 2y + 10 <= 20)
        JuMP.@constraint(m, x >= 1)

        JuMP.@NLconstraint(m, x^2 <= 5)
        JuMP.@NLconstraint(m, exp(y) + x <= 7)

        JuMP.optimize!(m)
        status = MOI.get(m, MOI.TerminationStatus())

        @test status == MOI.LOCALLY_SOLVED
        @test isapprox(JuMP.objective_value(m), -8.26928, atol = TOL)
        @test isapprox(JuMP.objective_bound(m), -8.26928, atol = TOL)
        @test isapprox(JuMP.value(x), 2.23607, atol = TOL)
        @test isapprox(JuMP.value(y), 1.56107, atol = TOL)
    end

    testname = "Maximization"
    @testset "$testname" begin
        m = JuMP.Model(solver)

        JuMP.@variable(m, x >= 0, start = 1, Int)
        JuMP.@variable(m, y >= 0, start = 1)

        JuMP.@objective(m, Max, 3x + y)

        JuMP.@constraint(m, 3x + 2y + 10 <= 20)
        JuMP.@NLconstraint(m, x^2 <= 9)

        JuMP.optimize!(m)
        status = MOI.get(m, MOI.TerminationStatus())

        @test status == MOI.LOCALLY_SOLVED
        @test isapprox(JuMP.objective_value(m), 9.5, atol = TOL)
    end

    testname = "Nonlinear objective"
    @testset "$testname" begin
        m = JuMP.Model(solver)

        JuMP.@variable(m, x >= 0, start = 1, Int)
        JuMP.@variable(m, y >= 0, start = 1)

        JuMP.@objective(m, Max, -x^2 - y)

        JuMP.@constraint(m, x + 2y >= 4)
        JuMP.@NLconstraint(m, x^2 <= 9)

        JuMP.optimize!(m)
        status = MOI.get(m, MOI.TerminationStatus())

        @test status == MOI.LOCALLY_SOLVED
        @test isapprox(JuMP.objective_value(m), -2.0, atol = TOL)
        @test isapprox(JuMP.objective_bound(m), -2.0, atol = TOL)
    end
end

# log_level model tests
function run_log_level(
    mip_solver_drives::Bool,
    mip_solver,
    cont_solver,
    log_level::Int,
    TOL::Real,
)
    solver = JuMP.optimizer_with_attributes(
        Pavito.Optimizer,
        "timeout" => 120.0,
        "mip_solver_drives" => mip_solver_drives,
        "mip_solver" => mip_solver,
        "cont_solver" => cont_solver,
        "log_level" => log_level,
    )

    testname = "QP optimal 2"
    @testset "$testname" begin
        m = JuMP.Model(solver)

        JuMP.@variable(m, x >= 0, Int)

        JuMP.@objective(m, Max, x)

        JuMP.@constraint(m, x^2 <= 5)
        JuMP.@constraint(m, x >= 0.5)

        JuMP.optimize!(m)
        status = MOI.get(m, MOI.TerminationStatus())

        @test status == MOI.LOCALLY_SOLVED
    end

    testname = "Infeasible 3"
    @testset "$testname" begin
        m = JuMP.Model(solver)

        JuMP.@variable(m, x >= 0, start = 1, Int)
        JuMP.@variable(m, y >= 0, start = 1)

        JuMP.@objective(m, Min, -3x - y)

        JuMP.@constraint(m, 3x + 2y + 10 <= 20)
        JuMP.@constraint(m, 6x + 5y >= 30)
        JuMP.@NLconstraint(m, x^2 >= 8)
        JuMP.@NLconstraint(m, exp(y) + x <= 7)

        JuMP.optimize!(m)
        status = MOI.get(m, MOI.TerminationStatus())

        @test status in [MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE]
    end
end
