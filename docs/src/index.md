# DynamicalResilience.jl

Resilience of dynamical systems written as SciML problems. Where
[ResilienceCore.jl](https://ecorecipes.github.io/ResilienceCore.jl/dev/) defines what a
linearised operator means and which metrics apply to it, this package finds the points to
linearise at and follows them as conditions change:

| question | function | returns |
|---|---|---|
| where is the equilibrium? | [`steady_state`](@ref) (Newton, or integrate then refine) | the state and whether it converged |
| will a pathogen invade? | [`reproduction_number`](@ref) (next-generation method) | R₀, F, V and K = FV⁻¹ |
| is a seasonal cycle stable? | [`period_map_jacobian`](@ref), [`periodic_orbit`](@ref) | a `DiscreteJacobian`: the Floquet multipliers |
| how far is the threshold? | [`resilience_along`](@ref) | metrics along a parameter, and whether stability is lost at a fold or a Hopf bifurcation |

No ODE solver is a dependency. Functions that integrate take the algorithm as `alg`, for
example `Tsit5()` from OrdinaryDiffEqTsit5.

## Example: approaching a fold

As a parameter brings a stable equilibrium towards a fold (saddle-node) bifurcation, its
resilience, the return rate after a small perturbation, falls to zero like a square root. This
is critical slowing down, the basis of early-warning indicators.

```julia
using DynamicalResilience, SciMLBase

f(u, μ, t) = [μ - u[1]^2]                     # stable branch √μ, resilience 2√μ
prob = ODEProblem(f, [1.0], (0.0, 1.0), 1.0)
path = resilience_along(prob, (p, μ) -> μ, 1.0:-0.05:0.01; u0 = [1.0])
path.metrics.resilience                       # 2√μ: 2.0, 1.95, …, 0.45 at μ = 0.05
```

With a Hopf bifurcation instead, the leading eigenvalues are a complex pair and
`path.transition.kind == :hopf`.

## Example: R₀

```julia
sir(u, p, t) = [p.μ - p.β*u[1]*u[2] - p.μ*u[1], p.β*u[1]*u[2] - (p.γ + p.μ)*u[2]]
prob = ODEProblem(sir, [1.0, 0.0], (0.0, 1.0), (β = 2.0, γ = 0.5, μ = 0.02))
reproduction_number(prob, [1.0, 0.0]; infected = [2],
                    new_infections = (u, p, t) -> [p.β*u[1]*u[2]]).R0    # β/(γ + μ)
```

Case-study packages (for example Chytridiomycosis.jl) supply only model glue: which named
equilibria a model has and which terms are new infections.
