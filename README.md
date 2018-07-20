[![Build Status](https://travis-ci.org/JuliaOpt/Pavito.jl.svg?branch=master)](https://travis-ci.org/JuliaOpt/Pavito.jl) [![codecov](https://codecov.io/gh/JuliaOpt/Pavito.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaOpt/Pavito.jl)

# Pavito

Pavito is a **mixed-integer convex programming** (MICP) solver package written in [Julia](http://julialang.org/). MICP problems are convex except for restrictions that some variables take binary or integer values.

Pavito solves MICP problems by constructing sequential polyhedral outer-approximations of the convex feasible set, similar to [Bonmin](https://projects.coin-or.org/Bonmin). Pavito accesses state-of-the-art MILP solvers and continuous, derivative-based nonlinear programming (NLP) solvers through the MathProgBase interface.

For algorithms that use a conic solver instead of an NLP solver, use [Pajarito](https://github.com/JuliaOpt/Pajarito.jl). Pajarito is a robust mixed-integer conic solver that can handle such established problem classes as mixed-integer second-order cone programming (MISOCP) and mixed-integer semidefinite programming (MISDP).

## Installation

Pavito can be installed through the Julia package manager:
```
julia> Pkg.add("Pavito")
```

## Usage

There are several convenient ways to model MICPs in Julia and access Pavito:

|             | [JuMP][JuMP-url]  | [Convex.jl][convex-url]  | [MathProgBase][mpb-url]  |
|-------------|-------------------|--------------------------|--------------------------|
| NLP model   | [X][JuMP-nlp-url] |                          | [X][mpb-nlp-url]         |
| Conic model | X                 | X                        | [X][mpb-conic-url]       |

[mpb-nlp-url]: http://mathprogbasejl.readthedocs.io/en/latest/nlp.html
[mpb-conic-url]: http://mathprogbasejl.readthedocs.io/en/latest/conic.html
[JuMP-url]: https://github.com/JuliaOpt/JuMP.jl
[JuMP-nlp-url]: http://jump.readthedocs.io/en/latest/nlp.html
[convex-url]: https://github.com/JuliaOpt/Convex.jl
[mpb-url]: https://github.com/JuliaOpt/MathProgBase.jl

JuMP and Convex.jl are algebraic modeling interfaces, while MathProgBase is a lower-level interface for providing input in raw callback or matrix form. Convex.jl is perhaps the most user-friendly way to provide input in conic form, since it transparently handles conversion of algebraic expressions. JuMP supports general nonlinear smooth functions, e.g. by using `@NLconstraint`. JuMP also supports conic modeling, but requires cones to be explicitly specified, e.g. by using `norm(x) <= t` for second-order cone constraints. Pavito may be accessed through MathProgBase from outside Julia by using the experimental [cmpb](https://github.com/mlubin/cmpb) interface which provides a C API to the low-level conic input format. The [ConicBenchmarkUtilities](https://github.com/mlubin/ConicBenchmarkUtilities.jl) package provides utilities to read files in the [CBF](http://cblib.zib.de/) format.

## MIP and continuous solvers

The algorithm implemented by Pavito itself is relatively simple, and most of the hard work is performed by the MILP solver and the NLP solver. **The performance of Pavito depends on these two types of solvers.**

The mixed-integer solver is specified by using the `mip_solver` option to `PavitoSolver`, e.g. `PavitoSolver(mip_solver=CplexSolver())`. You must first load the Julia package which provides the mixed-integer solver, e.g. `using CPLEX`. The continuous derivative-based nonlinear solver (e.g. [Ipopt](https://projects.coin-or.org/Ipopt) or [KNITRO](http://www.ziena.com/knitro.htm)) is specified by using the `cont_solver` option, e.g. `PavitoSolver(cont_solver=IpoptSolver())`.

MIP and continuous solver parameters must be specified through their corresponding Julia interfaces. For example, to turn off the output of Ipopt solver, use `cont_solver=IpoptSolver(print_level=0)`.

## Pavito solver options

The following options can be passed to `PavitoSolver()` to modify its behavior:

  * `log_level::Int` Verbosity flag: 0 for quiet, higher for basic solve info
  * `timeout::Float64` Time limit for algorithm (in seconds)
  * `rel_gap::Float64` Relative optimality gap termination condition
  * `mip_solver_drives::Bool` Let MILP solver manage convergence ("branch and cut")
  * `mip_solver::MathProgBase.AbstractMathProgSolver` MILP solver
  * `cont_solver::MathProgBase.AbstractMathProgSolver` Continuous NLP solver

**Pavito is not yet numerically robust and may require tuning of parameters to improve convergence.** If the default parameters don't work for you, please let us know. For improved Pavito performance, MILP solver integrality tolerance and feasibility tolerances should typically be tightened, for example to `1e-8`.

## Bug reports and support

Please report any issues via the Github **[issue tracker]**. All types of issues are welcome and encouraged; this includes bug reports, documentation typos, feature requests, etc. The **[Optimization (Mathematical)]** category on Discourse is appropriate for general discussion.

[issue tracker]: https://github.com/JuliaOpt/Pavito.jl/issues
[Optimization (Mathematical)]: https://discourse.julialang.org/c/domain/opt
