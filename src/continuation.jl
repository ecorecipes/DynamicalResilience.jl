# Resilience followed along a parameter, and where stability is lost.

"""
    leading_eigenvalue(J::ContinuousJacobian) -> Complex

The eigenvalue of `J` with the largest real part (of a complex pair, the one with positive
imaginary part). Its real part is minus ResilienceCore's `resilience`.
"""
function leading_eigenvalue(J::ContinuousJacobian)
    λ = eigvals(J.J)
    m = maximum(real, λ)
    top = filter(z -> real(z) >= m - 1e-12 * max(1.0, abs(m)), λ)
    return complex(top[argmax(imag.(top))])
end

"""
    resilience_along(prob::ODEProblem, set_parameter, values; u0 = prob.u0,
                     metrics = (resilience, reactivity), t = first(prob.tspan),
                     method = :newton, alg = nothing, tol = 1e-10, complex_tol = 1e-8,
                     steady_kwargs...)
        -> (parameter, equilibrium, converged, stable, leading, metrics, transition)

Follow an equilibrium of `prob` along a parameter and compute resilience metrics on the way: a
natural-parameter continuation with warm starts.

For each `v` in `values` (in the order given), the parameters are `set_parameter(prob.p, v)`, the
equilibrium is found with [`steady_state`](@ref) from the previous one (from `u0` at the start),
it is linearised with ResilienceCore's `linearise`, and each function in `metrics` is applied to
the resulting `ContinuousJacobian`. Newton's method follows unstable equilibria as well, so the
path continues through a Hopf bifurcation; at a fold the equilibrium disappears and Newton stops
converging (or jumps to the other branch, whose leading eigenvalue is real and positive).

Returns vectors over `values`: the equilibria, whether each converged, whether each is stable,
the [`leading_eigenvalue`](@ref), and a `NamedTuple` of metric vectors keyed by the metric
functions' names (`NaN` where the equilibrium was not found). `transition` describes the first
loss of stability after a stable stretch, or is `nothing`:

- `kind = :hopf` if the leading eigenvalue at the last stable point is a complex pair
  (`|imag| > complex_tol`): a pair crosses the imaginary axis and oscillations appear;
- `kind = :fold` if it is real: a real eigenvalue reaches zero, so resilience falls to zero as the
  equilibrium meets a saddle and disappears;
- `index`, the last stable point; `estimate`, the parameter where the leading real part crosses
  zero by linear interpolation between the last stable and the next point (or, if the next point
  did not converge, the midpoint of the two parameter values).

A fold is approached like a square root (`resilience ∝ √|v − v*|`), so the interpolated estimate
is biased towards the last stable point; refine `values` near the transition for accuracy.
"""
function resilience_along(prob::ODEProblem, set_parameter, values::AbstractVector; u0::AbstractVector = prob.u0,
                          metrics = (resilience, reactivity), t = first(prob.tspan), method::Symbol = :newton,
                          alg = nothing, tol::Real = 1e-10, complex_tol::Real = 1e-8, steady_kwargs...)
    n = length(values)
    names = Tuple(Symbol(nameof(m)) for m in metrics)
    out = [fill(NaN, n) for _ in metrics]
    equilibria = Vector{Vector{Float64}}(undef, n)
    converged = falses(n)
    stable = falses(n)
    leading = fill(complex(NaN, NaN), n)
    u = Vector{Float64}(u0)
    for (k, v) in enumerate(values)
        p = set_parameter(prob.p, v)
        ss = steady_state(prob, u; p = p, t = t, method = method, alg = alg, tol = tol, steady_kwargs...)
        equilibria[k] = ss.u
        ss.converged || continue
        converged[k] = true
        u = ss.u
        J = linearise(prob, u; p = p, t = t, check = false)
        stable[k] = isstable(J)
        leading[k] = leading_eigenvalue(J)
        for (i, m) in enumerate(metrics)
            out[i][k] = m(J)
        end
    end
    return (parameter = collect(values), equilibrium = equilibria, converged = converged, stable = stable,
            leading = leading, metrics = NamedTuple{names}(Tuple(out)),
            transition = find_transition(values, converged, stable, leading, complex_tol))
end

function find_transition(values, converged, stable, leading, complex_tol)
    for k in 1:(length(values) - 1)
        stable[k] || continue
        stable[k + 1] && continue
        λ = leading[k]
        kind = abs(imag(λ)) > complex_tol ? :hopf : :fold
        v0, v1 = float(values[k]), float(values[k + 1])
        estimate = if converged[k + 1]
            r0, r1 = real(λ), real(leading[k + 1])
            r1 == r0 ? (v0 + v1) / 2 : v0 + (0 - r0) * (v1 - v0) / (r1 - r0)
        else
            (v0 + v1) / 2
        end
        return (kind = kind, index = k, estimate = estimate)
    end
    return nothing
end
