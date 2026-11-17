"""
    DynamicalResilience

Resilience of dynamical systems written as SciML problems: steady states, the basic
reproduction number by the next-generation method, period maps and Floquet multipliers of
forced systems and of any map, resilience followed along a parameter to a fold or a Hopf
bifurcation, the distribution of resilience metrics over parameter draws, and simulation-based
resilience (recovery after a pulse, basin stability, early-warning indicators) on SciML's
integrator interface.

Results are the operator types of [ResilienceCore.jl](https://github.com/ecorecipes/ResilienceCore.jl)
(`ContinuousJacobian`, `DiscreteJacobian`), so every ResilienceCore metric applies to them.
No ODE solver is a dependency: functions that integrate take the algorithm as `alg`.
"""
module DynamicalResilience

using ForwardDiff: ForwardDiff
using LinearAlgebra: I, eigvals, norm
using ResilienceCore: ResilienceCore, ContinuousJacobian, DiscreteJacobian, linearise, resilience, reactivity,
                      isstable
using Random: Random, AbstractRNG
using SciMLBase: SciMLBase, ODEProblem, isinplace, remake
using Statistics: quantile

export steady_state, reproduction_number, period_map_jacobian, periodic_orbit, resilience_along,
       leading_eigenvalue, map_jacobian, fixed_point, metrics_over
export recovery, basin_stability, early_warning, ramped_series
# re-exported from ResilienceCore, so that `using DynamicalResilience` is enough for the common case
export linearise, resilience, reactivity, isstable, ContinuousJacobian, DiscreteJacobian

include("rhs.jl")
include("maps.jl")
include("steady.jl")
include("reproduction.jl")
include("periodic.jl")
include("continuation.jl")
include("draws.jl")
include("simulation.jl")

end
