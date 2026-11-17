# Equilibria of ODE problems.

"""
    steady_state(prob::ODEProblem, u0 = prob.u0; p = prob.p, t = first(prob.tspan),
                 method = :newton, alg = nothing, tol = 1e-10, maxiters = 100, tmax = 1e4,
                 solve_kwargs...)
        -> (u, converged, residual, iterations)

An equilibrium `u` of `prob` (`f(u, p, t) = 0`), started from `u0`.

- `method = :newton`: damped Newton iteration on `f`, with the Jacobian from the problem's
  analytic `jac` if it has one and by ForwardDiff otherwise, halving the step until the residual
  falls. It converges to the equilibrium near `u0` whether or not that equilibrium is stable,
  which is what continuation needs ([`resilience_along`](@ref)).
- `method = :integrate`: integrate from `u0` over `(t, t + tmax)` with the ODE algorithm `alg`
  (required; for example `Tsit5()` from OrdinaryDiffEqTsit5), then refine the end state by
  Newton. It finds an attracting equilibrium, the one a simulation would reach.

`converged` is true when the residual `‖f(u)‖∞` is at most `tol · max(1, ‖u‖∞)`. `solve_kwargs`
are passed to `solve` for `:integrate`.
"""
function steady_state(prob::ODEProblem, u0::AbstractVector = prob.u0; p = prob.p, t = first(prob.tspan),
                      method::Symbol = :newton, alg = nothing, tol::Real = 1e-10, maxiters::Integer = 100,
                      tmax::Real = 1e4, solve_kwargs...)
    if method === :integrate
        alg === nothing && throw(ArgumentError("method = :integrate needs an ODE algorithm: pass `alg`, " *
                                               "for example `alg = Tsit5()`"))
        sol = SciMLBase.solve(remake(prob; u0 = Vector{Float64}(u0), p = p, tspan = (t, t + tmax)), alg;
                              save_everystep = false, solve_kwargs...)
        start = Vector{Float64}(sol.u[end])
        res = newton(prob, start, p, t, tol, maxiters)
        return (u = res.u, converged = res.converged, residual = res.residual,
                iterations = res.iterations)
    elseif method === :newton
        return newton(prob, Vector{Float64}(u0), p, t, tol, maxiters)
    end
    throw(ArgumentError("method must be :newton or :integrate, got :$method"))
end

function newton(prob, u::Vector{Float64}, p, t, tol, maxiters)
    converged(u, r) = r <= tol * max(1.0, norm(u, Inf))
    f = rhs(prob, u, p, t)
    r = norm(f, Inf)
    for k in 1:maxiters
        converged(u, r) && return (u = u, converged = true, residual = r, iterations = k - 1)
        J = rhs_jacobian(prob, u, p, t)
        step = try
            J \ f
        catch
            return (u = u, converged = false, residual = r, iterations = k)
        end
        all(isfinite, step) || return (u = u, converged = false, residual = r, iterations = k)
        λ = 1.0
        accepted = false
        for _ in 1:30                                   # backtracking: halve until the residual falls
            trial = u .- λ .* step
            ft = rhs(prob, trial, p, t)
            rt = norm(ft, Inf)
            if all(isfinite, ft) && rt < r
                u, f, r, accepted = trial, ft, rt, true
                break
            end
            λ /= 2
        end
        accepted || return (u = u, converged = converged(u, r), residual = r, iterations = k)
    end
    return (u = u, converged = converged(u, r), residual = r, iterations = maxiters)
end

"""
    steady_state(prob::DiscreteProblem, u0 = prob.u0; p = prob.p, t = first(prob.tspan),
                 method = :newton, tol = 1e-10, maxiters = 100, iterations = 10_000, h = 1e-6)
        -> (u, converged, residual, iterations)

A fixed point `u` of the discrete-time map of `prob` (`f(u, p, t) = u`), started from `u0`.

- `method = :newton`: [`fixed_point`](@ref) on `u ↦ f(u, p, t)`, which finds the fixed point near
  `u0` whether or not it is stable.
- `method = :iterate`: apply the map up to `iterations` times from `u0`, then refine by Newton;
  it finds an attracting fixed point, the one a simulation would reach.

The returned fields match the ODE method; `residual` is `‖f(u) − u‖∞`.
"""
function steady_state(prob::SciMLBase.DiscreteProblem, u0::AbstractVector = prob.u0; p = prob.p,
                      t = first(prob.tspan), method::Symbol = :newton, tol::Real = 1e-10, maxiters::Integer = 100,
                      iterations::Integer = 10_000, h::Real = 1e-6)
    F(x) = rhs(prob, x, p, t)
    start = Vector{Float64}(u0)
    if method === :iterate
        for _ in 1:iterations
            next = Vector{Float64}(F(start))
            all(isfinite, next) || break
            done = norm(next .- start, Inf) <= tol * max(1.0, norm(start, Inf))
            start = next
            done && break
        end
    elseif method !== :newton
        throw(ArgumentError("method must be :newton or :iterate, got :$method"))
    end
    fp = fixed_point(F, start; tol = tol, maxiters = maxiters, h = h)
    residual = norm(F(fp.u) .- fp.u, Inf)
    return (u = fp.u, converged = fp.converged, residual = residual, iterations = fp.iterations)
end
