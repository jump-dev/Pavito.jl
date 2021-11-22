#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 wrapped NLP solver for infeasible subproblem case
=========================================================#

struct InfeasibleNLPEvaluator <: MOI.AbstractNLPEvaluator
    d::MOI.AbstractNLPEvaluator
    num_variables::Int
    minus::BitVector
end

function MOI.initialize(
    d::InfeasibleNLPEvaluator,
    requested_features::Vector{Symbol},
)
    return MOI.initialize(d.d, requested_features)
end

function MOI.features_available(d::InfeasibleNLPEvaluator)
    return intersect([:Grad, :Jac, :Hess], MOI.features_available(d.d))
end

function MOI.eval_constraint(d::InfeasibleNLPEvaluator, g, x)
    MOI.eval_constraint(d.d, g, x[1:d.num_variables])
    for i in eachindex(d.minus)
        g[i] -= sign(d.minus[i]) * x[d.num_variables+i]
    end
    return
end

function MOI.jacobian_structure(d::InfeasibleNLPEvaluator)
    IJ_new = copy(MOI.jacobian_structure(d.d))
    for i in eachindex(d.minus)
        push!(IJ_new, (i, d.num_variables + i))
    end
    return IJ_new
end

function MOI.eval_constraint_jacobian(d::InfeasibleNLPEvaluator, J, x)
    MOI.eval_constraint_jacobian(d.d, J, x[1:d.num_variables])
    k = length(J) - length(d.minus)
    for i in eachindex(d.minus)
        J[k+i] = (d.minus[i] ? -1.0 : 1.0)
    end
    return
end

# Hessian: add linear terms and remove the objective so the hessian of the
# objective is zero and the hessian of the constraints is unaffected;
# also set `σ = 0.0` to absorb the contribution of the hessian of the objective

function MOI.hessian_lagrangian_structure(d::InfeasibleNLPEvaluator)
    return MOI.hessian_lagrangian_structure(d.d)
end

function MOI.eval_hessian_lagrangian(d::InfeasibleNLPEvaluator, H, x, σ, μ)
    return MOI.eval_hessian_lagrangian(d.d, H, x[1:d.num_variables], 0.0, μ)
end
