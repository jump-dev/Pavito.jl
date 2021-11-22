#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

module TestJuMP

import JuMP
import MINLPTests
import Pavito
using Test

const MOI = JuMP.MOI

function runtests(mip_solver, cont_solver)
    @testset "$(msd)" for msd in (true, false)
        @testset "MINLPTests" begin
            run_minlptests(msd, mip_solver, cont_solver, 0, 1e-3)
        end
        @testset "QP-NLP" begin
            run_qp_nlp_tests(msd, mip_solver, cont_solver, 0, 1e-3)
        end
        @testset "log_level=$(log_level)" for log_level in 0:2
            run_log_level_tests(msd, mip_solver, cont_solver, log_level, 1e-3)
        end
    end
    return
end

function run_minlptests(
    mip_solver_drives::Bool,
    mip_solver,
    cont_solver,
    log_level::Int,
    TOL::Real,
)
    solver = MOI.OptimizerWithAttributes(
        Pavito.Optimizer,
        "timeout" => 120.0,
        "mip_solver_drives" => mip_solver_drives,
        "mip_solver" => mip_solver,
        "cont_solver" => cont_solver,
        "log_level" => log_level,
    )
    MINLPTests.test_nlp_mi(
        solver,
        exclude = String[
            # TODO fix failures:
            "003_010",
            "003_011",
            "003_012",
            "003_013",
            "003_014",
            "003_015",
            "003_016",
            "007_010",
            "007_020",
            # Excluded tests
            "006_010",  # User-defined function
        ],
        objective_tol = TOL,
        primal_tol = TOL,
        dual_tol = NaN,
    )
    return
end

function run_qp_nlp_tests(
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
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "_test_qp_") ||
           startswith("$(name)", "_test_nlp_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)(solver, TOL)
            end
        end
    end
    return
end

function run_log_level_tests(
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
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "_test_loglevel_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)(solver, TOL)
            end
        end
    end
    return
end

###
### Individual tests go below here.
###

function _test_qp_optimal(solver, TOL)
    m = JuMP.Model(solver)
    JuMP.@variable(m, x >= 0, Int)
    JuMP.@variable(m, y >= 0)
    JuMP.@variable(m, 0 <= u <= 10, Int)
    JuMP.@variable(m, w == 1)
    JuMP.@objective(m, Min, -3x - y)
    JuMP.@constraint(m, 3x + 10 <= 20)
    JuMP.@constraint(m, y^2 <= u * w)
    JuMP.optimize!(m)
    status = JuMP.termination_status(m)
    @test status == MOI.LOCALLY_SOLVED
    @test isapprox(JuMP.objective_value(m), -12.162277, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), -12.162277, atol = TOL)
    @test isapprox(JuMP.value(x), 3, atol = TOL)
    @test isapprox(JuMP.value(y), 3.162277, atol = TOL)
    return
end

function _test_qp_maximize(solver, TOL)
    m = JuMP.Model(solver)
    JuMP.@variable(m, x >= 0, Int)
    JuMP.@variable(m, y >= 0)
    JuMP.@variable(m, 0 <= u <= 10, Int)
    JuMP.@variable(m, w == 1)
    JuMP.@objective(m, Max, 3x + y)
    JuMP.@constraint(m, 3x + 2y + 10 <= 20)
    JuMP.@constraint(m, x^2 <= u * w)
    JuMP.optimize!(m)
    status = JuMP.termination_status(m)
    @test status == MOI.LOCALLY_SOLVED
    @test isapprox(JuMP.objective_value(m), 9.5, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), 9.5, atol = TOL)
    @test isapprox(JuMP.value(x), 3, atol = TOL)
    @test isapprox(JuMP.value(y), 0.5, atol = TOL)
    return
end

function _test_qp_infeasible(solver, TOL)
    m = JuMP.Model(solver)
    JuMP.@variable(m, x >= 0, Int)
    JuMP.@objective(m, Max, x)
    JuMP.@constraint(m, x^2 <= 3.9)
    JuMP.@constraint(m, x >= 1.1)
    JuMP.optimize!(m)
    status = JuMP.termination_status(m)
    @test status in (MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE)
    return
end

function _test_nlp_nonconvex_error(solver, TOL)
    m = JuMP.Model(solver)
    JuMP.@variable(m, x >= 0, start = 1, Int)
    JuMP.@variable(m, y >= 0, start = 1)
    JuMP.@objective(m, Min, -3x - y)
    JuMP.@constraint(m, 3x + 2y + 10 <= 20)
    JuMP.@NLconstraint(m, 8 <= x^2 <= 10)
    @test_throws ErrorException JuMP.optimize!(m)
    return
end

function _test_nlp_optimal(solver, TOL)
    m = JuMP.Model(solver)
    JuMP.@variable(m, x >= 0, start = 1, Int)
    JuMP.@variable(m, y >= 0, start = 1)
    JuMP.@objective(m, Min, -3x - y)
    JuMP.@constraint(m, 3x + 2y + 10 <= 20)
    JuMP.@constraint(m, x >= 1)
    JuMP.@NLconstraint(m, x^2 <= 5)
    JuMP.@NLconstraint(m, exp(y) + x <= 7)
    JuMP.optimize!(m)
    status = JuMP.termination_status(m)
    @test status == MOI.LOCALLY_SOLVED
    @test isapprox(JuMP.value(x), 2.0)
    return
end

function _test_nlp_infeasible_1(solver, TOL)
    m = JuMP.Model(solver)
    JuMP.@variable(m, x >= 0, start = 1, Int)
    JuMP.@variable(m, y >= 0, start = 1)
    JuMP.@objective(m, Min, -3x - y)
    JuMP.@constraint(m, 3x + 2y + 10 <= 20)
    JuMP.@NLconstraint(m, x^2 >= 9)
    JuMP.@NLconstraint(m, exp(y) + x <= 2)
    JuMP.optimize!(m)
    status = JuMP.termination_status(m)
    @test status in (MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE)
    return
end

function _test_nlp_infeasible_2(solver, TOL)
    m = JuMP.Model(solver)
    JuMP.@variable(m, x >= 0, start = 1, Int)
    JuMP.@objective(m, Max, x)
    JuMP.@NLconstraint(m, log(x) >= 0.75)
    JuMP.@constraint(m, x <= 2.9)
    JuMP.optimize!(m)
    status = JuMP.termination_status(m)
    @test status in (MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE)
    return
end

function _test_nlp_continuous(solver, TOL)
    m = JuMP.Model(solver)
    JuMP.@variable(m, x >= 0, start = 1)
    JuMP.@variable(m, y >= 0, start = 1)
    JuMP.@objective(m, Min, -3x - y)
    JuMP.@constraint(m, 3x + 2y + 10 <= 20)
    JuMP.@constraint(m, x >= 1)
    JuMP.@NLconstraint(m, x^2 <= 5)
    JuMP.@NLconstraint(m, exp(y) + x <= 7)
    JuMP.optimize!(m)
    status = JuMP.termination_status(m)
    @test status == MOI.LOCALLY_SOLVED
    @test isapprox(JuMP.objective_value(m), -8.26928, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), -8.26928, atol = TOL)
    @test isapprox(JuMP.value(x), 2.23607, atol = TOL)
    @test isapprox(JuMP.value(y), 1.56107, atol = TOL)
    return
end

function _test_nlp_maximization(solver, TOL)
    m = JuMP.Model(solver)
    JuMP.@variable(m, x >= 0, start = 1, Int)
    JuMP.@variable(m, y >= 0, start = 1)
    JuMP.@objective(m, Max, 3x + y)
    JuMP.@constraint(m, 3x + 2y + 10 <= 20)
    JuMP.@NLconstraint(m, x^2 <= 9)
    JuMP.optimize!(m)
    status = JuMP.termination_status(m)
    @test status == MOI.LOCALLY_SOLVED
    @test isapprox(JuMP.objective_value(m), 9.5, atol = TOL)
    return
end

function _test_nlp_nonlinear_objective(solver, TOL)
    m = JuMP.Model(solver)
    JuMP.@variable(m, x >= 0, start = 1, Int)
    JuMP.@variable(m, y >= 0, start = 1)
    JuMP.@objective(m, Max, -x^2 - y)
    JuMP.@constraint(m, x + 2y >= 4)
    JuMP.@NLconstraint(m, x^2 <= 9)
    JuMP.optimize!(m)
    status = JuMP.termination_status(m)
    @test status == MOI.LOCALLY_SOLVED
    @test isapprox(JuMP.objective_value(m), -2.0, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), -2.0, atol = TOL)
    return
end

function _test_loglevel_optimal(solver, TOL)
    m = JuMP.Model(solver)
    JuMP.@variable(m, x >= 0, Int)
    JuMP.@objective(m, Max, x)
    JuMP.@constraint(m, x^2 <= 5)
    JuMP.@constraint(m, x >= 0.5)
    JuMP.optimize!(m)
    status = JuMP.termination_status(m)
    @test status == MOI.LOCALLY_SOLVED
    return
end

function test_loglevel_infeasible(solver, TOL)
    m = JuMP.Model(solver)
    JuMP.@variable(m, x >= 0, start = 1, Int)
    JuMP.@variable(m, y >= 0, start = 1)
    JuMP.@objective(m, Min, -3x - y)
    JuMP.@constraint(m, 3x + 2y + 10 <= 20)
    JuMP.@constraint(m, 6x + 5y >= 30)
    JuMP.@NLconstraint(m, x^2 >= 8)
    JuMP.@NLconstraint(m, exp(y) + x <= 7)
    JuMP.optimize!(m)
    status = JuMP.termination_status(m)
    @test status in [MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE]
    return
end

end
