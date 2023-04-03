# Pavito.jl

[![Build Status](https://github.com/jump-dev/Pavito.jl/workflows/CI/badge.svg)](https://github.com/jump-dev/Pavito.jl/actions)
[![Coverage](https://codecov.io/gh/jump-dev/Pavito.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jump-dev/Pavito.jl)

[Pavito.jl](https://github.com/jump-dev/Pavito.jl) is a mixed-integer convex
programming (MICP) solver package written in [Julia](http://julialang.org/).

MICP problems are convex, except for restrictions that some variables take
binary or integer values.

Pavito solves MICP problems by constructing sequential polyhedral
outer-approximations of the convex feasible set, similar to [Bonmin](https://projects.coin-or.org/Bonmin).

Pavito accesses state-of-the-art MILP solvers and continuous, derivative-based
nonlinear programming (NLP) solvers through [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl).

For algorithms that use a conic solver instead of an NLP solver, use
[Pajarito](https://github.com/jump-dev/Pajarito.jl). Pajarito is a robust
mixed-integer conic solver that can handle such established problem classes as
mixed-integer second-order cone programming (MISOCP) and mixed-integer
semidefinite programming (MISDP).

## License

`Pavito.jl` is licensed under the [MPL 2.0 license](https://github.com/jump-dev/Pavito.jl/blob/master/LICENSE.md).

## Installation

Install Pavito using `Pkg.add`:
```julia
import Pkg
Pkg.add("Pavito")
```

## Use with JuMP

To use Pavito with [JuMP](https://github.com/jump-dev/JuMP.jl), use
`Pavito.Optimizer`:
```julia
using JuMP, Pavito
import GLPK, Ipopt
model = Model(
    optimizer_with_attributes(
        Pavito.Optimizer,
        "mip_solver" => optimizer_with_attributes(GLPK.Optimizer),
        "cont_solver" =>
            optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0),
    ),
)
```

The algorithm implemented by Pavito itself is relatively simple; most of the
hard work is performed by the MILP solver passed as `mip_solver` and the NLP
solver passed as `cont_solver`.

**The performance of Pavito depends on these two types of solvers.**

For better performance, you should use a commercial MILP solver such as CPLEX
or Gurobi.

## Options

The following optimizer attributes can set to a `Pavito.Optimizer` to modify its
behavior:

  * `log_level::Int` Verbosity flag: 0 for quiet, higher for basic solve info
  * `timeout::Float64` Time limit for algorithm (in seconds)
  * `rel_gap::Float64` Relative optimality gap termination condition
  * `mip_solver_drives::Bool` Let MILP solver manage convergence ("branch and
    cut")
  * `mip_solver::MOI.OptimizerWithAttributes` MILP solver
  * `cont_solver::MOI.OptimizerWithAttributes` Continuous NLP solver

**Pavito is not yet numerically robust and may require tuning of parameters to
improve convergence.**

If the default parameters don't work for you, please let us know by opening an
issue.

For improved Pavito performance, MILP solver integrality tolerance and
feasibility tolerances should typically be tightened, for example to `1e-8`.

## Bug reports and support

Please report any issues via the [GitHub issue tracker](https://github.com/jump-dev/Pavito.jl/issues).
All types of issues are welcome and encouraged; this includes bug reports,
documentation typos, feature requests, etc. The [Optimization (Mathematical)](https://discourse.julialang.org/c/domain/opt) category on Discourse is appropriate for general
discussion.
