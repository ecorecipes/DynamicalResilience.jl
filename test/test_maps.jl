using SciMLBase: DiscreteProblem

# the logistic map x ↦ r x (1 − x): fixed point 1 − 1/r with multiplier 2 − r
logistic_map(u, p, t) = [p * u[1] * (1 - u[1])]
logistic_map!(du, u, p, t) = (du[1] = p * u[1] * (1 - u[1]); nothing)

@testset "maps" begin
    A = [0.5 0.2; -0.1 0.3]
    b = [1.0, 2.0]
    F(u) = A * u .+ b
    for method in (:forwarddiff, :finite_differences)
        J = map_jacobian(F, [0.3, -0.4]; method)
        @test J isa DiscreteJacobian
        @test J.J ≈ A rtol = 1e-8
    end
    fp = fixed_point(F, [0.0, 0.0])
    @test fp.converged
    @test fp.u ≈ (I - A) \ b
    @test sort(abs.(fp.multipliers)) ≈ sort(abs.(eigvals(A)))
    @test isstable(fp.jacobian)
    @test_throws DimensionMismatch map_jacobian(u -> [sum(u)], [1.0, 2.0])
    @test_throws ArgumentError map_jacobian(F, [0.0, 0.0]; method = :other)

    # Newton finds stable and unstable fixed points alike
    for r in (2.8, 3.2)
        fp = fixed_point(u -> [r * u[1] * (1 - u[1])], [0.5])
        @test fp.converged
        @test fp.u ≈ [1 - 1 / r]
        @test only(fp.multipliers) ≈ 2 - r
        @test isstable(fp.jacobian) == (abs(2 - r) < 1)
    end

    # steady_state on a DiscreteProblem, out of place and in place, by Newton and by iteration
    for f in (logistic_map, logistic_map!), method in (:newton, :iterate)
        prob = DiscreteProblem(f, [0.6], (0, 100), 2.8)     # from 0.3, plain Newton overshoots to 0
        ss = steady_state(prob; method)
        @test ss.converged
        @test ss.u ≈ [1 - 1 / 2.8]
        @test ss.residual < 1e-10
        @test linearise(prob, ss.u).J ≈ map_jacobian(u -> logistic_map(u, 2.8, 0), ss.u).J
    end
    @test_throws ArgumentError steady_state(DiscreteProblem(logistic_map, [0.3], (0, 1), 2.8); method = :other)
end
