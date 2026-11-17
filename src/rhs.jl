# Evaluating and differentiating the right-hand side of a SciML problem, in place or not.

# f(u) of the problem at parameters p and time t, as a new vector
function rhs(prob, u::AbstractVector, p, t)
    if isinplace(prob)
        du = similar(u, promote_type(eltype(u), Float64))
        prob.f(du, u, p, t)
        return du
    end
    return collect(prob.f(u, p, t))
end

# Jacobian of f at u by automatic differentiation (analytic `jac` used when the problem has one)
function rhs_jacobian(prob, u::AbstractVector, p, t)
    jac = hasproperty(prob.f, :jac) ? getproperty(prob.f, :jac) : nothing
    if jac !== nothing
        if isinplace(prob)
            J = zeros(length(u), length(u))
            jac(J, u, p, t)
            return J
        end
        return Matrix{Float64}(jac(u, p, t))
    end
    return Matrix{Float64}(ForwardDiff.jacobian(x -> rhs(prob, x, p, t), Vector{Float64}(u)))
end

residual_norm(prob, u, p, t) = norm(rhs(prob, u, p, t), Inf)
