# The basic reproduction number by the next-generation method (Diekmann, Heesterbeek & Metz 1990;
# van den Driessche & Watmough 2002).

"""
    reproduction_number(prob::ODEProblem, dfe; infected, new_infections, p = prob.p,
                        t = first(prob.tspan), check = true, tol = 1e-8)
        -> (R0, F, V, K)

The basic reproduction number of `prob` at its disease-free equilibrium `dfe`, by the
next-generation method of van den Driessche & Watmough (2002).

- `infected` are the indices of the infected compartments (or, for a macroparasite or
  pathogen-load model, of every state that carries infection).
- `new_infections(u, p, t)` returns the rate at which *new* infections enter each of them, a
  vector in the order of `infected`. Everything else in `f` restricted to `infected` (progression,
  recovery, death, and transfers between infected compartments) is the transition part.

With `F = ∂new_infections/∂x` and `V = F − ∂f_infected/∂x`, both over the infected states at `dfe`
and computed by ForwardDiff, `K = F V⁻¹` is the next-generation matrix and `R0 = ρ(K)`. The
disease-free equilibrium is locally stable when `R0 < 1` and unstable when `R0 > 1`, provided the
uninfected subsystem is stable at `dfe` (the conditions of the theorem). `check` verifies that
`dfe` is an equilibrium (`‖f(dfe)‖∞ ≤ tol · max(1, ‖dfe‖∞)`) and that `new_infections(dfe) = 0`.
"""
function reproduction_number(prob::ODEProblem, dfe::AbstractVector; infected::AbstractVector{<:Integer},
                             new_infections, p = prob.p, t = first(prob.tspan), check::Bool = true,
                             tol::Real = 1e-8)
    x = Vector{Float64}(dfe)
    idx = collect(Int, infected)
    isempty(idx) && throw(ArgumentError("`infected` must list at least one state"))
    all(i -> 1 <= i <= length(x), idx) || throw(ArgumentError("`infected` has indices outside 1:$(length(x))"))
    if check
        r = residual_norm(prob, x, p, t)
        bound = tol * max(1.0, norm(x, Inf))
        r <= bound || throw(ArgumentError("dfe is not an equilibrium: ‖f(dfe)‖∞ = $r > $bound"))
        n0 = norm(new_infections(x, p, t), Inf)
        n0 <= bound || throw(ArgumentError("new_infections(dfe) is not zero ($n0): dfe is not disease free"))
    end
    newinf = u -> collect(new_infections(u, p, t))
    length(newinf(x)) == length(idx) ||
        throw(ArgumentError("new_infections returns $(length(newinf(x))) rates for $(length(idx)) infected states"))
    Fall = ForwardDiff.jacobian(newinf, x)
    Jall = rhs_jacobian(prob, x, p, t)
    F = Matrix{Float64}(Fall[:, idx])
    V = F .- Jall[idx, idx]
    K = F / V
    return (R0 = maximum(abs, eigvals(K)), F = F, V = V, K = K)
end
