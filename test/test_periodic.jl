# du/dt = f(t) A u with f(t) = 1 + 0.5 sin(2πt): the period-1 monodromy is exp(A ∫f) = exp(A)
forced_linear(u, p, t) = (1 + 0.5 * sin(2π * t)) .* (p * u)

# seasonally forced logistic growth: du/dt = r(t) u (1 − u/K)
seasonal_logistic(u, p, t) = [p.r * (1 + 0.8 * sin(2π * t)) * u[1] * (1 - u[1] / p.K)]

@testset "period maps" begin
    A = [-0.5 0.3; 0.1 -0.8]
    prob = ODEProblem(forced_linear, [1.0, 1.0], (0.0, 1.0), A)
    for method in (:forwarddiff, :finite_differences)
        Φ = period_map_jacobian(prob, [0.3, -0.2], 1.0; alg = Tsit5(), method, abstol = 1e-12, reltol = 1e-12)
        @test Φ isa DiscreteJacobian
        @test Φ.J ≈ exp(A) rtol = 1e-7
    end
    # the origin is the periodic orbit; its multipliers are the eigenvalues of exp(A)
    orb = periodic_orbit(prob, [0.2, 0.1], 1.0; alg = Tsit5(), abstol = 1e-12, reltol = 1e-12)
    @test orb.converged
    @test norm(orb.u) < 1e-8
    @test sort(abs.(orb.multipliers)) ≈ sort(abs.(eigvals(exp(A)))) rtol = 1e-7
    @test resilience(orb.jacobian) ≈ -log(maximum(abs, eigvals(exp(A)))) rtol = 1e-6

    # the carrying capacity is a periodic orbit (u ≡ K), with multiplier exp(−∫r(t)dt) = exp(−r)
    p = (r = 1.2, K = 2.0)
    lprob = ODEProblem(seasonal_logistic, [0.5], (0.0, 1.0), p)
    orb = periodic_orbit(lprob, [1.5], 1.0; alg = Tsit5(), abstol = 1e-12, reltol = 1e-12)
    @test orb.converged
    @test orb.u ≈ [2.0] atol = 1e-8
    @test only(orb.multipliers) ≈ exp(-p.r) rtol = 1e-6
    @test isstable(orb.jacobian)
end
