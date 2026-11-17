# Period maps of forced (seasonal) systems: their Jacobian, periodic orbits and Floquet multipliers.

# the state after integrating from u over (t0, t0 + T)
function flow(prob, u, T, t0, p, alg, solve_kwargs)
    pr = remake(prob; u0 = u, p = p, tspan = (t0, t0 + T))
    sol = SciMLBase.solve(pr, alg; save_everystep = false, save_start = false, solve_kwargs...)
    SciMLBase.successful_retcode(sol) ||
        throw(ErrorException("the integration over one period failed with retcode $(sol.retcode)"))
    return collect(sol.u[end])
end

function map_jacobian(prob, u, T, t0, p, alg, method, h, solve_kwargs)
    if method === :forwarddiff
        return Matrix{Float64}(ForwardDiff.jacobian(x -> flow(prob, x, T, t0, p, alg, solve_kwargs), u))
    elseif method === :finite_differences
        n = length(u)
        J = zeros(n, n)
        for j in 1:n
            d = h * max(abs(u[j]), 1.0)
            up = copy(u); up[j] += d
            dn = copy(u); dn[j] -= d
            J[:, j] = (flow(prob, up, T, t0, p, alg, solve_kwargs) .- flow(prob, dn, T, t0, p, alg, solve_kwargs)) ./ (2d)
        end
        return J
    end
    throw(ArgumentError("method must be :forwarddiff or :finite_differences, got :$method"))
end

"""
    period_map_jacobian(prob::ODEProblem, u0, T; alg, t0 = first(prob.tspan), p = prob.p,
                        method = :forwarddiff, h = 1e-6, solve_kwargs...) -> DiscreteJacobian

The Jacobian of the period map `Φ_T: u ↦ u(t0 + T)` of `prob` started from `u0` at `t0`, as a
`DiscreteJacobian` (the monodromy matrix of the variational equation). At a point of a
`T`-periodic orbit its eigenvalues are the Floquet multipliers; the orbit is stable when all have
modulus below one, and ResilienceCore's `resilience` of the result is the return rate per
period, `−log ρ(Φ)`.

- `alg` is the ODE algorithm (required), e.g. `Tsit5()`; `solve_kwargs` (such as `abstol`,
  `reltol`) go to `solve`.
- `method = :forwarddiff` differentiates through the solver (forward-mode sensitivities of the
  integration). This is accurate to solver tolerance and works with the OrdinaryDiffEq solvers
  when `f` accepts dual numbers. `method = :finite_differences` uses central differences with step
  `h · max(|u_j|, 1)`; use it when `f` cannot take dual numbers.

`u0` is stored as the operator's state. It need not be periodic, but the multipliers mean
Floquet multipliers only at a periodic point; see [`periodic_orbit`](@ref).
"""
function period_map_jacobian(prob::ODEProblem, u0::AbstractVector, T::Real; alg, t0 = first(prob.tspan),
                             p = prob.p, method::Symbol = :forwarddiff, h::Real = 1e-6, solve_kwargs...)
    u = Vector{Float64}(u0)
    J = map_jacobian(prob, u, T, t0, p, alg, method, h, solve_kwargs)
    return DiscreteJacobian(J, u)
end

"""
    periodic_orbit(prob::ODEProblem, u0, T; alg, t0 = first(prob.tspan), p = prob.p,
                   method = :forwarddiff, tol = 1e-9, maxiters = 50, solve_kwargs...)
        -> (u, multipliers, jacobian, converged, iterations)

A point `u` of a `T`-periodic orbit of `prob` (a fixed point of the period map, `Φ_T(u) = u`),
found by Newton shooting from `u0`: each step solves `(Φ'(u) − I) δ = −(Φ_T(u) − u)`, with `Φ'`
from [`period_map_jacobian`](@ref). For a model with seasonal forcing of period `T` this is the
seasonal attractor or saddle near `u0`.

Returns the point, its Floquet multipliers (eigenvalues of `Φ'(u)`), the period-map Jacobian as a
`DiscreteJacobian`, and whether `‖Φ_T(u) − u‖∞ ≤ tol · max(1, ‖u‖∞)` was reached.
"""
function periodic_orbit(prob::ODEProblem, u0::AbstractVector, T::Real; alg, t0 = first(prob.tspan), p = prob.p,
                        method::Symbol = :forwarddiff, h::Real = 1e-6, tol::Real = 1e-9, maxiters::Integer = 50,
                        solve_kwargs...)
    u = Vector{Float64}(u0)
    n = length(u)
    done(u, g) = norm(g, Inf) <= tol * max(1.0, norm(u, Inf))
    iterations = 0
    for k in 1:maxiters
        g = flow(prob, u, T, t0, p, alg, solve_kwargs) .- u
        done(u, g) && break
        Φ = map_jacobian(prob, u, T, t0, p, alg, method, h, solve_kwargs)
        δ = (Φ - I) \ (-g)
        all(isfinite, δ) || break
        u = u .+ δ
        iterations = k
    end
    g = flow(prob, u, T, t0, p, alg, solve_kwargs) .- u
    Φ = map_jacobian(prob, u, T, t0, p, alg, method, h, solve_kwargs)
    return (u = u, multipliers = eigvals(Φ), jacobian = DiscreteJacobian(Φ, u), converged = done(u, g),
            iterations = iterations)
end
