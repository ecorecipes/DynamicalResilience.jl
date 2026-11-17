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

## Re-exported from ResilienceCore

`linearise`, `resilience`, `reactivity`, `isstable`, `ContinuousJacobian` and
`DiscreteJacobian` are re-exported from
[ResilienceCore.jl](https://ecorecipes.github.io/ResilienceCore.jl/dev/), where they are
documented, together with the other metrics that apply to the operators returned here.
