# Lotka–Volterra competition with a known interior equilibrium
function lv!(du, u, p, t)
    r, A = p
    du .= u .* (r .+ A * u)
    return nothing
end
lv(u, p, t) = u .* (p[1] .+ p[2] * u)

@testset "steady_state" begin
    r = [1.0, 0.8]
    A = [-1.0 -0.5; -0.4 -1.0]
    ustar = -A \ r
    for prob in (ODEProblem(lv!, [0.3, 0.3], (0.0, 1.0), (r, A)), ODEProblem(lv, [0.3, 0.3], (0.0, 1.0), (r, A)))
        # Newton converges to the equilibrium near its start, stable or not: from (0.3, 0.3) it
        # reaches the origin, so start it near the interior point
        @test steady_state(prob).u ≈ [0.0, 0.0] atol = 1e-12
        ss = steady_state(prob, [0.7, 0.45])
        @test ss.converged
        @test ss.u ≈ ustar atol = 1e-10
        # integrating finds the attractor from the same start
        si = steady_state(prob; method = :integrate, alg = Tsit5(), tmax = 200.0)
        @test si.converged
        @test si.u ≈ ustar atol = 1e-10
        J = linearise(prob, ss.u)
        @test J isa ContinuousJacobian
        @test isstable(J)
    end
    # Newton also finds an unstable equilibrium: the saddle at r1 > 0 with species 2 absent is
    # reached from nearby
    prob = ODEProblem(lv, [0.9, 0.0], (0.0, 1.0), (r, A))
    @test steady_state(prob).u ≈ [1.0, 0.0] atol = 1e-10
    @test_throws ArgumentError steady_state(prob; method = :integrate)
    @test_throws ArgumentError steady_state(prob; method = :bogus)
end
