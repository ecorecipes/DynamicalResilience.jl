# Period maps of forced (seasonal) systems: their Jacobian, periodic orbits and Floquet multipliers.

# the state after integrating from u over (t0, t0 + T)
function flow(prob, u, T, t0, p, alg, solve_kwargs)
    pr = remake(prob; u0 = u, p = p, tspan = (t0, t0 + T))
    sol = SciMLBase.solve(pr, alg; save_everystep = false, save_start = false, solve_kwargs...)
    SciMLBase.successful_retcode(sol) ||
        throw(ErrorException("the integration over one period failed with retcode $(sol.retcode)"))
    return collect(sol.u[end])
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

This is [`map_jacobian`](@ref) applied to the period map. `u0` is stored as the operator's
state. It need not be periodic, but the multipliers mean Floquet multipliers only at a periodic
point; see [`periodic_orbit`](@ref).
"""
function period_map_jacobian(prob::ODEProblem, u0::AbstractVector, T::Real; alg, t0 = first(prob.tspan),
                             p = prob.p, method::Symbol = :forwarddiff, h::Real = 1e-6, solve_kwargs...)
    return map_jacobian(x -> flow(prob, x, T, t0, p, alg, solve_kwargs), u0; method = method, h = h)
end

"""
    periodic_orbit(prob::ODEProblem, u0, T; alg, t0 = first(prob.tspan), p = prob.p,
                   method = :forwarddiff, tol = 1e-9, maxiters = 50, solve_kwargs...)
        -> (u, multipliers, jacobian, converged, iterations)

A point `u` of a `T`-periodic orbit of `prob` (a fixed point of the period map, `Φ_T(u) = u`),
found by Newton shooting from `u0`: each step solves `(Φ'(u) − I) δ = −(Φ_T(u) − u)`, with `Φ'`
from [`period_map_jacobian`](@ref); this is [`fixed_point`](@ref) applied to the period map.
For a model with seasonal forcing of period `T` this is the seasonal attractor or saddle near
`u0`.

Returns the point, its Floquet multipliers (eigenvalues of `Φ'(u)`), the period-map Jacobian as a
`DiscreteJacobian`, and whether `‖Φ_T(u) − u‖∞ ≤ tol · max(1, ‖u‖∞)` was reached.
"""
function periodic_orbit(prob::ODEProblem, u0::AbstractVector, T::Real; alg, t0 = first(prob.tspan), p = prob.p,
                        method::Symbol = :forwarddiff, h::Real = 1e-6, tol::Real = 1e-9, maxiters::Integer = 50,
                        solve_kwargs...)
    return fixed_point(x -> flow(prob, x, T, t0, p, alg, solve_kwargs), u0; method = method, h = h, tol = tol,
                       maxiters = maxiters)
end
