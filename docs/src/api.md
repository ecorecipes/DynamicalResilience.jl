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

## Re-exported from ResilienceCore

`linearise`, `resilience`, `reactivity`, `isstable`, `ContinuousJacobian` and
`DiscreteJacobian` are re-exported from
[ResilienceCore.jl](https://ecorecipes.github.io/ResilienceCore.jl/dev/), where they are
documented, together with the other metrics that apply to the operators returned here.
