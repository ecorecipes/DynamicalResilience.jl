"""
    DynamicalResilience

Resilience of dynamical systems written as SciML problems: steady states, the basic
reproduction number by the next-generation method, period maps and Floquet multipliers of
forced systems, and resilience followed along a parameter to a fold or a Hopf bifurcation.

Results are the operator types of [ResilienceCore.jl](https://github.com/ecorecipes/ResilienceCore.jl)
(`ContinuousJacobian`, `DiscreteJacobian`), so every ResilienceCore metric applies to them.
No ODE solver is a dependency: functions that integrate take the algorithm as `alg`.
"""
module DynamicalResilience

using ForwardDiff: ForwardDiff
using LinearAlgebra: I, eigvals, norm
using ResilienceCore: ResilienceCore, ContinuousJacobian, DiscreteJacobian, linearise, resilience, reactivity,
                      isstable
using SciMLBase: SciMLBase, ODEProblem, isinplace, remake

export steady_state, reproduction_number, period_map_jacobian, periodic_orbit, resilience_along,
       leading_eigenvalue
# re-exported from ResilienceCore, so that `using DynamicalResilience` is enough for the common case
export linearise, resilience, reactivity, isstable, ContinuousJacobian, DiscreteJacobian

include("rhs.jl")
include("steady.jl")
include("reproduction.jl")
include("periodic.jl")
include("continuation.jl")

end
