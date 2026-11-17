# Jacobians and fixed points of arbitrary maps u ↦ F(u): one time step of a discrete-time model,
# the period map of a forced ODE, a year of a daily-step simulation, …

# ∂F/∂u at u as a matrix
function map_derivative(F, u::Vector{Float64}, method::Symbol, h::Real)
    if method === :forwarddiff
        return Matrix{Float64}(ForwardDiff.jacobian(x -> collect(F(x)), u))
    elseif method === :finite_differences
        n = length(u)
        J = zeros(n, n)
        for j in 1:n
            d = h * max(abs(u[j]), 1.0)
            up = copy(u); up[j] += d
            dn = copy(u); dn[j] -= d
            J[:, j] = (collect(F(up)) .- collect(F(dn))) ./ (2d)
        end
        return J
    end
    throw(ArgumentError("method must be :forwarddiff or :finite_differences, got :$method"))
end

function check_square(J, u)
    size(J) == (length(u), length(u)) ||
        throw(DimensionMismatch("F must map a state of length $(length(u)) to one of the same length; " *
                                "its Jacobian is $(size(J, 1))×$(size(J, 2))"))
    return J
end

"""
    map_jacobian(F, u; method = :forwarddiff, h = 1e-6) -> DiscreteJacobian

The Jacobian of the map `u ↦ F(u)` at `u`, as ResilienceCore's `DiscreteJacobian` with state
`u`. `F` is any function from a state vector to a state vector of the same length: one step of a
discrete-time model, the period map of a forced ODE ([`period_map_jacobian`](@ref) is this
function applied to the flow over one period), or a year of a daily-step simulation. At a fixed
point (`F(u) = u`) its eigenvalues are the multipliers that decide local stability, and
ResilienceCore's `resilience` is the return rate per application of `F`, `−log ρ(∂F/∂u)`.

- `method = :forwarddiff` differentiates `F` with dual numbers. `F` must accept them, which
  rules out code that writes into preallocated `Float64` arrays or calls non-Julia libraries.
- `method = :finite_differences` uses central differences with step `h · max(|u_j|, 1)`.

`u` need not be a fixed point; see [`fixed_point`](@ref) to find one.
"""
function map_jacobian(F, u::AbstractVector; method::Symbol = :forwarddiff, h::Real = 1e-6)
    x = Vector{Float64}(u)
    return DiscreteJacobian(check_square(map_derivative(F, x, method, h), x), x)
end

"""
    fixed_point(F, u0; method = :forwarddiff, h = 1e-6, tol = 1e-9, maxiters = 50)
        -> (u, multipliers, jacobian, converged, iterations)

A fixed point `u` of the map `F` (`F(u) = u`), found by Newton's method from `u0`: each step
solves `(F'(u) − I) δ = −(F(u) − u)`, with `F'` from [`map_jacobian`](@ref) (`method` and `h` as
there). Newton converges to the fixed point near `u0` whether it is stable or not.

Returns the point, its multipliers (the eigenvalues of `F'(u)`), `F'(u)` as a `DiscreteJacobian`,
whether `‖F(u) − u‖∞ ≤ tol · max(1, ‖u‖∞)` was reached, and the number of Newton steps taken.
The fixed point is locally stable when every multiplier has modulus below one
(`isstable(result.jacobian)`).
"""
function fixed_point(F, u0::AbstractVector; method::Symbol = :forwarddiff, h::Real = 1e-6, tol::Real = 1e-9,
                     maxiters::Integer = 50)
    u = Vector{Float64}(u0)
    done(u, g) = norm(g, Inf) <= tol * max(1.0, norm(u, Inf))
    iterations = 0
    for k in 1:maxiters
        g = collect(F(u)) .- u
        done(u, g) && break
        Φ = check_square(map_derivative(F, u, method, h), u)
        δ = try
            (Φ - I) \ (-g)
        catch
            break
        end
        all(isfinite, δ) || break
        u = u .+ δ
        iterations = k
    end
    g = collect(F(u)) .- u
    Φ = check_square(map_derivative(F, u, method, h), u)
    return (u = u, multipliers = eigvals(Φ), jacobian = DiscreteJacobian(Φ, u), converged = done(u, g),
            iterations = iterations)
end
