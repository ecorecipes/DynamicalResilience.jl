# Simulation-based resilience on SciML's integrator interface: recovery after a pulse, basin
# stability and early-warning indicators. Every function takes `(prob, alg)` as `solve` does, so
# it works for ODE, SDE, discrete-time and jump problems and for any model package whose problems
# implement the integrator contract below.
#
# The contract, as used here:
# - `init(prob, alg; kwargs...)` (with `seed` passed through when given);
# - `step!(integ, dt, true)`: advance exactly `dt`;
# - `reinit!(integ)`: restart from the problem's initial state (and, for integrators that keep
#   their own random streams, from the same streams, so the run replays exactly);
# - `integ.u`, `integ.t`;
# - to apply a pulse: `set_u!(integ, u)`, or else in-place mutation of `integ.u` followed by
#   `derivative_discontinuity!(integ, true)` (or the deprecated `u_modified!`);
# - optionally `check_error(integ)`;
# - `integ.p` assignment, only for `ramped_series` with a `ramp`;
# - for `basin_stability`: `solve` on an `EnsembleProblem` whose `prob_func(prob, ctx)` uses `remake`.

# Whether `f` has a method for these arguments other than SciMLBase's generic fallback, which only
# throws "not implemented".
function implemented(f, args...)
    hasmethod(f, Tuple{map(typeof, args)...}) || return false
    return parentmodule(which(f, Tuple{map(typeof, args)...})) !== SciMLBase
end

function mark_modified!(integ)
    if implemented(SciMLBase.derivative_discontinuity!, integ, true)
        SciMLBase.derivative_discontinuity!(integ, true)
    elseif implemented(SciMLBase.u_modified!, integ, true)
        SciMLBase.u_modified!(integ, true)
    end
    return integ
end

# Put the state `new` into the integrator; `new === integ.u` means the perturbation mutated it.
function apply_state!(integ, new)
    if new === integ.u
        mark_modified!(integ)
    elseif implemented(SciMLBase.set_u!, integ, new)
        SciMLBase.set_u!(integ, new)
    elseif integ.u isa AbstractArray
        copyto!(integ.u, new)
        mark_modified!(integ)
    else
        throw(ArgumentError("cannot apply the perturbed state: $(typeof(integ)) implements neither " *
                            "`set_u!` nor a mutable array state; make `perturb` mutate `u` in place"))
    end
    return integ
end

ok(integ) = !implemented(SciMLBase.check_error, integ) ||
            SciMLBase.successful_retcode(SciMLBase.check_error(integ))

# Step to time `t` exactly (no-op if already there).
function advance_to!(integ, t)
    gap = t - integ.t
    gap > 0 && SciMLBase.step!(integ, gap, true)
    return ok(integ)
end

snapshot(x) = x isa AbstractArray ? copy(x) : x

# Only the current state is read, so SciML's own differential-equation integrators are started
# without saving every step (which OrdinaryDiffEq also requires before `set_u!`); other problem
# types get no extra keyword, since their `init` may not accept it.
init_defaults(prob) = prob isa SciMLBase.AbstractDEProblem ? (save_everystep = false,) : (;)

function init_run(prob, alg, seed; kwargs...)
    opts = merge(init_defaults(prob), values(kwargs))
    return seed === nothing ? SciMLBase.init(prob, alg; opts...) : SciMLBase.init(prob, alg; seed = seed, opts...)
end

"""
    recovery(prob, alg; perturb, observe = identity, t_pulse = first(prob.tspan),
             horizon = last(prob.tspan) - t_pulse, interval, distance = (a, b) -> norm(a .- b),
             seed = nothing, perturbed_seed = seed, rng = Random.default_rng(), tol = 0.01,
             window = (0.1, 1e-4), kwargs...)
        -> (t, distance, return_time, return_rate, retcode)

Recovery after a pulse, from paired simulations of `prob` with the algorithm `alg`.

The control run is integrated to `t_pulse` and then recorded every `interval` until
`t_pulse + horizon`. The perturbed run repeats it up to `t_pulse`, where the state is replaced
by `perturb(u, rng)` (which may return a new state or mutate `u` in place and return it), and is
recorded on the same grid. `t` is time since the pulse and `distance[k]` is
`distance(observe(perturbed), observe(control))` at `t[k]`; `distance[1]` is the size of the
pulse as seen through `observe`.

The two runs share their random numbers, so the distance measures the response to the pulse
rather than to different noise:

- **without `seed`** the perturbed run is the control replayed with `reinit!`. That is exact for
  deterministic problems and for integrators that restore their random streams on `reinit!`;
- **with `seed`** each run is a fresh `init(prob, alg; seed)`, the perturbed one with
  `perturbed_seed` (equal to `seed` by default; pass a different value for independent noise).
  For SDE problems the noise is then shared exactly only with fixed steps (`adaptive = false`,
  `dt = …` in `kwargs`), because adaptive steps sample the noise at state-dependent times.

`return_time` is the first `t` at which the distance falls to `tol` times its initial value
(`Inf` if it does not within the horizon). `return_rate` is minus the slope of a least-squares
fit of `log(distance)` against `t` over the points where the distance lies between
`window[2]` and `window[1]` times its initial value, so it measures the asymptotic return once
fast modes have died out; it is `NaN` if fewer than three points fall in that window. For a
small pulse to a stable equilibrium it approaches `resilience(linearise(prob, u*))`, and per
step for a map, `-log ρ(J)`.

`kwargs` are passed to `init` (so a fixed solver step is `dt = …`, separate from the recording
`interval`). `retcode` is `:Success`, or `:Failure` if either run's
`check_error` reported a failure (the series then stop there).
"""
function recovery(prob, alg; perturb, observe = identity, t_pulse = first(prob.tspan),
                  horizon = last(prob.tspan) - t_pulse, interval, distance = (a, b) -> norm(a .- b),
                  seed = nothing, perturbed_seed = seed, rng::AbstractRNG = Random.default_rng(),
                  tol::Real = 0.01, window::Tuple{Real, Real} = (0.1, 1e-4), kwargs...)
    n = floor(Int, horizon / interval + 1e-9)
    n ≥ 1 || throw(ArgumentError("horizon must be at least one recording interval"))
    grid = t_pulse .+ interval .* (0:n)

    control = Vector{Any}(undef, 0)
    integ = init_run(prob, alg, seed; kwargs...)
    good = advance_to!(integ, t_pulse)
    good && push!(control, snapshot(observe(integ.u)))
    for k in 1:n
        good || break
        good = advance_to!(integ, grid[k + 1])
        good && push!(control, snapshot(observe(integ.u)))
    end

    if seed === nothing
        SciMLBase.reinit!(integ)
    else
        integ = init_run(prob, alg, perturbed_seed; kwargs...)
    end
    d = Float64[]
    good2 = advance_to!(integ, t_pulse)
    if good2
        apply_state!(integ, perturb(integ.u, rng))
        isempty(control) || push!(d, distance(observe(integ.u), control[1]))
    end
    for k in 1:n
        (good2 && k + 1 ≤ length(control)) || break
        good2 = advance_to!(integ, grid[k + 1])
        good2 && push!(d, distance(observe(integ.u), control[k + 1]))
    end

    t = collect(grid[1:length(d)]) .- t_pulse
    return (t = t, distance = d, return_time = return_time(t, d, tol), return_rate = return_rate(t, d, window),
            retcode = good && good2 ? :Success : :Failure)
end

function return_time(t, d, tol)
    isempty(d) && return NaN
    k = findfirst(x -> x ≤ tol * d[1], d)
    return k === nothing ? Inf : t[k]
end

function return_rate(t, d, window)
    isempty(d) && return NaN
    hi, lo = window
    keep = [i for i in eachindex(d) if lo * d[1] ≤ d[i] ≤ hi * d[1] && d[i] > 0]
    length(keep) < 3 && return NaN
    x, y = t[keep], log.(d[keep])
    x̄, ȳ = sum(x) / length(x), sum(y) / length(y)
    return -sum((x .- x̄) .* (y .- ȳ)) / sum(abs2, x .- x̄)
end

"""
    basin_stability(prob, alg; sample_perturbation, returned, observe = identity,
                    trajectories = 1000, ensemble = EnsembleThreads(), tspan = prob.tspan,
                    seed = nothing, kwargs...)
        -> (fraction, se, returned, trajectories, failed)

The fraction of random perturbations from which the system returns (Menck et al. 2013).

Each trajectory starts from `sample_perturbation(copy(prob.u0), rng)`, with `prob.u0` normally
the attractor and `rng` the trajectory's own random number generator (`ctx.rng` of an
`EnsembleProblem`, reproducible when `seed` is given). It is solved over `tspan` with `alg` and
counts as having returned if `returned(observe(final state))` is true. `fraction` is
`returned / (trajectories - failed)` and `se` its binomial standard error; trajectories whose
solve failed are counted in `failed` and excluded, never silently treated as not returning.
`kwargs` go to `solve` (for example `save_everystep = false`).
"""
function basin_stability(prob, alg; sample_perturbation, returned, observe = identity, trajectories::Integer = 1000,
                         ensemble = SciMLBase.EnsembleThreads(), tspan = prob.tspan, seed = nothing, kwargs...)
    base = prob.u0
    pf(p, ctx) = SciMLBase.remake(p; u0 = sample_perturbation(copy(base), ctx.rng), tspan = tspan)
    function of(sol, ctx)
        SciMLBase.successful_retcode(sol) || return (missing, false)
        return (Bool(returned(observe(sol.u[end]))), false)
    end
    ens = SciMLBase.EnsembleProblem(prob; prob_func = pf, output_func = of)
    sim = seed === nothing ?
          SciMLBase.solve(ens, alg, ensemble; trajectories = trajectories, kwargs...) :
          SciMLBase.solve(ens, alg, ensemble; trajectories = trajectories, seed = seed, kwargs...)
    outcomes = sim.u
    failed = count(ismissing, outcomes)
    k = count(x -> x === true, outcomes)
    m = trajectories - failed
    p = m == 0 ? NaN : k / m
    return (fraction = p, se = m == 0 ? NaN : sqrt(p * (1 - p) / m), returned = k, trajectories = trajectories,
            failed = failed)
end

"""
    ramped_series(prob, alg; saveat, ramp = nothing, observe = identity, seed = nothing, kwargs...)
        -> (t, x)

A time series from one run of `prob`, recorded at the times `saveat`, for
[`early_warning`](@ref). With `ramp = (p0, t) -> p`, the parameter is set to `ramp(prob.p, t)`
at each recording time (piecewise constant between them, so the ramp should be slow on that
grid); the integrator must then allow `integ.p` to be assigned, as OrdinaryDiffEq and
StochasticDiffEq integrators do. With `ramp = nothing` the forcing must already be in the model.
`x[k] = observe(u(t[k]))`.
"""
function ramped_series(prob, alg; saveat, ramp = nothing, observe = identity, seed = nothing, kwargs...)
    times = collect(saveat)
    integ = init_run(prob, alg, seed; kwargs...)
    p0 = prob.p
    xs = Any[]
    ts = eltype(times)[]
    for τ in times
        if ramp !== nothing
            integ.p = ramp(p0, integ.t)
            mark_modified!(integ)
        end
        advance_to!(integ, τ) || break
        push!(ts, τ)
        push!(xs, snapshot(observe(integ.u)))
    end
    return (t = ts, x = [x for x in xs])
end

"""
    early_warning(x::AbstractVector; t = 1:length(x), window = 0.5, detrend = :gaussian,
                  bandwidth = 0.1)
        -> (t, variance, ac1, tau_variance, tau_ac1, residuals)

Early-warning indicators of critical slowing down (Scheffer et al. 2009; Dakos et al. 2012):
rolling variance and lag-1 autocorrelation of the detrended series, and Kendall's τ of each
against time. A rise in both, with τ well above zero, is the signature of a system losing
resilience as it approaches a bifurcation.

- `detrend`: `:gaussian` subtracts a Gaussian kernel smoother with standard deviation
  `bandwidth` (as a fraction of the series length if below 1, otherwise in samples);
  `:linear` subtracts a least-squares line; `:none` leaves the series.
- `window`: the rolling window, as a fraction of the series length if below 1, otherwise in
  samples. Indicator `k` uses the window ending at sample `window + k - 1`, and its time is
  that sample's `t`.
"""
function early_warning(x::AbstractVector{<:Real}; t = 1:length(x), window::Real = 0.5, detrend::Symbol = :gaussian,
                       bandwidth::Real = 0.1)
    n = length(x)
    length(t) == n || throw(ArgumentError("t and x must have the same length"))
    w = window < 1 ? max(3, round(Int, window * n)) : Int(window)
    3 ≤ w ≤ n || throw(ArgumentError("the window must hold between 3 and $n samples, got $w"))
    r = residuals(Float64.(x), Float64.(collect(t)), detrend, bandwidth)
    m = n - w + 1
    v, a = zeros(m), zeros(m)
    for k in 1:m
        s = view(r, k:(k + w - 1))
        μ = sum(s) / w
        c = s .- μ
        v[k] = sum(abs2, c) / (w - 1)
        a[k] = sum(c[i] * c[i + 1] for i in 1:(w - 1)) / sum(abs2, c)
    end
    te = collect(t)[w:n]
    return (t = te, variance = v, ac1 = a, tau_variance = kendall_tau(te, v), tau_ac1 = kendall_tau(te, a),
            residuals = r)
end

function residuals(x, t, detrend, bandwidth)
    n = length(x)
    detrend === :none && return copy(x)
    if detrend === :linear
        t̄, x̄ = sum(t) / n, sum(x) / n
        b = sum((t .- t̄) .* (x .- x̄)) / sum(abs2, t .- t̄)
        return x .- (x̄ .+ b .* (t .- t̄))
    elseif detrend === :gaussian
        bw = bandwidth < 1 ? bandwidth * n : float(bandwidth)
        reach = ceil(Int, 4bw)
        smooth = similar(x)
        for i in 1:n
            lo, hi = max(1, i - reach), min(n, i + reach)
            wts = [exp(-((j - i) / bw)^2 / 2) for j in lo:hi]
            smooth[i] = sum(wts .* view(x, lo:hi)) / sum(wts)
        end
        return x .- smooth
    end
    throw(ArgumentError("detrend must be :gaussian, :linear or :none, got :$detrend"))
end

# Kendall's τ-b
function kendall_tau(x, y)
    n = length(x)
    conc = disc = tx = ty = 0
    for i in 1:(n - 1), j in (i + 1):n
        sx, sy = sign(x[j] - x[i]), sign(y[j] - y[i])
        if sx == 0 && sy == 0
            continue
        elseif sx == 0
            tx += 1
        elseif sy == 0
            ty += 1
        elseif sx == sy
            conc += 1
        else
            disc += 1
        end
    end
    denom = sqrt(float(conc + disc + tx) * float(conc + disc + ty))
    return denom == 0 ? NaN : (conc - disc) / denom
end
