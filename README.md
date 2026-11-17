# DynamicalResilience.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://ecorecipes.github.io/DynamicalResilience.jl/dev/)

Resilience of dynamical systems written as SciML problems, built on
[ResilienceCore.jl](https://github.com/ecorecipes/ResilienceCore.jl). ResilienceCore says what a
linearised operator means and which metrics apply to it; this package finds the states to
linearise at and follows them as conditions change.

- `steady_state(prob, u0)`: an equilibrium, by damped Newton (stable or not) or by integrating
  with a given algorithm and refining.
- `reproduction_number(prob, dfe; infected, new_infections)`: R₀ by the next-generation method
  (van den Driessche & Watmough 2002), with F, V and K = FV⁻¹.
- `period_map_jacobian(prob, u0, T; alg)` and `periodic_orbit(prob, u0, T; alg)`: the monodromy
  matrix of a forced or seasonal system and its periodic orbit by shooting, as a
  `DiscreteJacobian` whose eigenvalues are the Floquet multipliers.
- `resilience_along(prob, set_parameter, values)`: resilience metrics along a parameter, with
  warm-started continuation, and whether stability is lost at a fold or a Hopf bifurcation.

```julia
using DynamicalResilience, SciMLBase

f(u, μ, t) = [μ - u[1]^2]                       # a fold at μ = 0
prob = ODEProblem(f, [1.0], (0.0, 1.0), 1.0)
path = resilience_along(prob, (p, μ) -> μ, 1.0:-0.07:-0.2; u0 = [1.0])
path.metrics.resilience                         # 2√μ, falling to zero: critical slowing down
path.transition                                 # (kind = :fold, …)
```

No ODE solver is a dependency: functions that integrate take the algorithm (`alg`), for example
`Tsit5()` from OrdinaryDiffEqTsit5. Case-study packages supply only model glue, such as named
equilibria and which terms are new infections.

## Status

Not registered; ResilienceCore and StructuredPopulationCore are fetched by URL through
`[sources]` (Julia ≥ 1.11). Tests: `julia --project -e 'using Pkg; Pkg.test()'`.
