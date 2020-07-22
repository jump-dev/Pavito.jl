#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

struct InfeasibleNLPEvaluator <: MOI.AbstractNLPEvaluator
    d::MOI.AbstractNLPEvaluator
    num_variables::Int
    minus::BitVector
end

MOI.initialize(d::InfeasibleNLPEvaluator, requested_features::Vector{Symbol}) = MOI.initialize(d.d, requested_features)
MOI.features_available(d::InfeasibleNLPEvaluator) = intersect([:Grad, :Jac, :Hess], MOI.features_available(d.d))

function MOI.eval_constraint(d::InfeasibleNLPEvaluator, g, x)
    MOI.eval_constraint(d.d, g, x[1:d.num_variables])
    for i in eachindex(d.minus)
        if d.minus[i]
            g[i] -= x[d.num_variables + i]
        else
            g[i] += x[d.num_variables + i]
        end
    end
    return
end

function MOI.jacobian_structure(d::InfeasibleNLPEvaluator)
    IJ = MOI.jacobian_structure(d.d)
    IJ_new = copy(IJ)
    for i in eachindex(d.minus)
        push!(IJ_new, (i, d.num_variables + i))
    end
    return IJ_new
end
function MOI.eval_constraint_jacobian(d::InfeasibleNLPEvaluator, J, x)
    MOI.eval_constraint_jacobian(d.d, J, x[1:d.num_variables])
    k = length(J) - length(d.minus)
    for i in eachindex(d.minus)
        if d.minus[i]
            J[k + i] = -1.0
        else
            J[k + i] = 1.0
        end
    end
    return
end

# Hessian
# We add linear terms and remove the objective so the hessian of the objective is zero
# and the hessian of the constraints is unaffected.
# We set `σ = 0.0` to absorb the contribution of the the hessian of the objective.
MOI.hessian_lagrangian_structure(d::InfeasibleNLPEvaluator) = MOI.hessian_lagrangian_structure(d.d)
MOI.eval_hessian_lagrangian(d::InfeasibleNLPEvaluator, H, x, σ, μ) = MOI.eval_hessian_lagrangian(d.d, H, x[1:d.num_variables], 0.0, μ)
