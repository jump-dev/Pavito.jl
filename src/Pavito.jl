#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 This package contains the mixed-integer convex programming (MICP) solver Pavito
 See readme for details
=========================================================#

__precompile__()

module Pavito
    import MathProgBase
    using JuMP
    using ConicNonlinearBridge

    using Compat.Printf
    using Compat.SparseArrays
    using Compat.LinearAlgebra

    import Compat: undef
    import Compat: @warn
    import Compat: stdout
    import Compat: stderr

    include("solver.jl")
    include("algorithm.jl")
end
