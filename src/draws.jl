# Resilience metrics over draws of a model's parameters (bootstrap replicates, posterior samples, …).

"""
    metrics_over(build, draws; u0 = nothing, metrics = (resilience, reactivity),
                 quantiles = (0.025, 0.5, 0.975), warm_start = true, t = nothing,
                 steady_kwargs...)
        -> (equilibrium, converged, stable, metrics, quantiles, probabilities,
            n, n_converged, n_stable, n_unconverged, n_unstable)

Uncertainty in resilience metrics from uncertainty in a model's parameters. For each draw `d` in
`draws` (any iterable, for example `eachrow(coefs)` for a matrix of bootstrap coefficients with
one replicate per row), `build(d)` returns a SciML problem (`ODEProblem` or `DiscreteProblem`);
its equilibrium is found with [`steady_state`](@ref), linearised with ResilienceCore's
`linearise`, and each function in `metrics` is applied to the resulting operator.

- **Starting points.** The first draw starts from `u0` (`prob.u0` if `u0` is `nothing`). With
  `warm_start = true` each later draw starts from the last equilibrium found, which is faster
  and keeps neighbouring draws on the same branch; if that fails, the draw is retried from
  `u0`. `steady_kwargs` (such as `method`, `alg`, `tol`) go to `steady_state`, and `t` (default
  `first(prob.tspan)`) to both `steady_state` and `linearise`.
- **Nothing is dropped.** Per-draw vectors cover every draw. Metrics are `NaN` where no
  equilibrium converged and are computed for unstable equilibria too (where `resilience` is
  negative). The counts `n_unconverged` and `n_unstable` say how many draws are in each case.
- **Summaries.** `quantiles` is a `NamedTuple` of metric quantile vectors at `probabilities`,
  taken over the draws with a converged, *stable* equilibrium only. If no draw is stable, they
  are `NaN`.
"""
function metrics_over(build, draws; u0 = nothing, metrics = (resilience, reactivity),
                      quantiles = (0.025, 0.5, 0.975), warm_start::Bool = true, t = nothing, steady_kwargs...)
    ds = collect(draws)
    n = length(ds)
    names = Tuple(Symbol(nameof(m)) for m in metrics)
    values = [fill(NaN, n) for _ in metrics]
    equilibria = Vector{Vector{Float64}}(undef, n)
    converged = falses(n)
    stable = falses(n)
    previous = nothing
    for (k, d) in enumerate(ds)
        prob = build(d)
        tk = t === nothing ? first(prob.tspan) : t
        cold = Vector{Float64}(u0 === nothing ? prob.u0 : u0)
        start = warm_start && previous !== nothing && length(previous) == length(cold) ? previous : cold
        ss = steady_state(prob, start; t = tk, steady_kwargs...)
        if !ss.converged && start !== cold
            ss = steady_state(prob, cold; t = tk, steady_kwargs...)
        end
        equilibria[k] = ss.u
        ss.converged || continue
        converged[k] = true
        previous = ss.u
        J = linearise(prob, ss.u; t = tk, check = false)
        stable[k] = isstable(J)
        for (i, m) in enumerate(metrics)
            values[i][k] = m(J)
        end
    end
    probabilities = collect(Float64, quantiles)
    summary = map(values) do v
        kept = v[stable]
        kept = kept[isfinite.(kept)]
        isempty(kept) ? fill(NaN, length(probabilities)) : quantile(kept, probabilities)
    end
    nc, ns = count(converged), count(stable)
    return (equilibrium = equilibria, converged = converged, stable = stable,
            metrics = NamedTuple{names}(Tuple(values)), quantiles = NamedTuple{names}(Tuple(summary)),
            probabilities = probabilities, n = n, n_converged = nc, n_stable = ns, n_unconverged = n - nc,
            n_unstable = nc - ns)
end
