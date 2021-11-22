#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 This package contains the mixed-integer convex programming (MICP) solver Pavito
 See README.md for details
=========================================================#

module Pavito
using Printf

import MathOptInterface
const MOI = MathOptInterface

const VI = MOI.VariableIndex
const SQF = MOI.ScalarQuadraticFunction{Float64}
const LT = MOI.LessThan{Float64}
const GT = MOI.GreaterThan{Float64}

include("infeasible_nlp.jl")
include("optimize.jl")
include("cut_utils.jl")
include("MOI_wrapper.jl")
end
