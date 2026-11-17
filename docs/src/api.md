# API Reference

```@meta
CurrentModule = DynamicalResilience
```

```@docs
DynamicalResilience
```

## Steady states

```@docs
steady_state
```

## The basic reproduction number

```@docs
reproduction_number
```

## Maps and fixed points

One step of a discrete-time model, the period map of a forced ODE or a year of a daily-step
simulation are all maps `u ↦ F(u)`.

```@docs
map_jacobian
fixed_point
```

## Forced and seasonal systems

```@docs
period_map_jacobian
periodic_orbit
```

## Thresholds

```@docs
resilience_along
leading_eigenvalue
```

## Uncertainty over parameter draws

```@docs
metrics_over
```

## Simulation-based resilience

These take `(prob, alg)` as `solve` does and drive SciML's integrator interface, so they apply to
ODE, SDE, discrete-time and jump problems and to any model package whose problems implement the
same contract. Each function uses only these calls on the integrator:

- `init(prob, alg; kwargs...)`, with `seed` passed through when given;
- `step!(integ, dt, true)`, to advance exactly `dt`;
- `reinit!(integ)`, to replay the control run;
- `integ.u` and `integ.t`;
- `set_u!(integ, u)`, or else in-place mutation of `integ.u` followed by
  `derivative_discontinuity!(integ, true)` (or the deprecated `u_modified!`), to apply a pulse;
- `check_error(integ)`, if implemented;
- assignment to `integ.p`, only for [`ramped_series`](@ref) with a `ramp`.

`basin_stability` solves an `EnsembleProblem` with `prob_func(prob, ctx)` and `remake`.

```@docs
recovery
basin_stability
ramped_series
early_warning
```

## Re-exported from ResilienceCore

`linearise`, `resilience`, `reactivity`, `isstable`, `ContinuousJacobian` and
`DiscreteJacobian` are re-exported from
[ResilienceCore.jl](https://ecorecipes.github.io/ResilienceCore.jl/dev/), where they are
documented, together with the other metrics that apply to the operators returned here.
