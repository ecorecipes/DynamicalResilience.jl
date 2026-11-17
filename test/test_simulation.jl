using OrdinaryDiffEqFunctionMap: FunctionMap
using StochasticDiffEq: EM
using SciMLBase: DiscreteProblem, SDEProblem, EnsembleSerial
using Statistics: mean, var

@testset "simulation-based resilience" begin
    @testset "recovery of an ODE: return rate → resilience as the pulse shrinks" begin
        f(u, p, t) = [-u[1] + u[2]^2, -3u[2] + u[1]^2 + 0.5u[1]]          # J(0) = [-1 0; 0.5 -3]
        prob = ODEProblem(f, [0.0, 0.0], (0.0, 30.0))
        r = resilience(linearise(prob, [0.0, 0.0]))
        @test r ≈ 1.0
        err = map((0.3, 0.01, 0.001)) do ε
            rec = recovery(prob, Tsit5(); perturb = (u, rng) -> u .+ ε .* [1.0, 1.0], interval = 0.05, horizon = 20.0,
                           abstol = 1e-12, reltol = 1e-12)
            @test rec.retcode === :Success
            @test rec.distance[1] ≈ ε * sqrt(2)
            @test isfinite(rec.return_time)
            abs(rec.return_rate - r)
        end
        @test err[end] < 1e-3
        @test err[end] ≤ err[1]

        # replay by reinit!: a pulse that changes nothing leaves the runs identical, whether the
        # perturbation returns the state or mutates it in place
        same = recovery(prob, Tsit5(); perturb = (u, rng) -> u, interval = 0.5, horizon = 5.0)
        @test all(iszero, same.distance)
        @test isnan(same.return_rate)
        inplace = recovery(prob, Tsit5(); perturb = (u, rng) -> (u .+= 0.01; u), interval = 0.05, horizon = 20.0,
                           abstol = 1e-12, reltol = 1e-12)
        @test inplace.distance[1] ≈ 0.01 * sqrt(2)
        @test inplace.return_rate ≈ r rtol = 1e-3
    end

    @testset "recovery of a map: return rate → −log ρ(J) per step" begin
        g(u, p, t) = [0.5u[1] + 0.2u[2] + 0.1u[1]^2, 0.1u[1] + 0.7u[2]]
        prob = DiscreteProblem(g, [0.0, 0.0], (0.0, 200.0))
        J = map_jacobian(u -> g(u, nothing, 0), [0.0, 0.0])
        target = -log(maximum(abs, eigvals(J.J)))
        @test resilience(J) ≈ target
        kick(u, rng) = u .+ 1e-3 .* [1.0, -0.5]
        deep = recovery(prob, FunctionMap(); perturb = kick, interval = 1, horizon = 120, window = (1e-3, 1e-8))
        @test deep.return_rate ≈ target rtol = 1e-5
        # the default window still carries some of the faster mode
        @test recovery(prob, FunctionMap(); perturb = kick, interval = 1, horizon = 120).return_rate ≈ target rtol = 0.02
    end

    @testset "basin stability of ẋ = x − x³ from x* = 1" begin
        # x* = 1 is regained iff the perturbed state stays positive: Δ ~ U(−3, 1) gives 1/2
        prob = ODEProblem((u, p, t) -> [u[1] - u[1]^3], [1.0], (0.0, 30.0))
        bs(seed) = basin_stability(prob, Tsit5(); sample_perturbation = (u, rng) -> u .+ (4rand(rng) - 3),
                                   returned = u -> abs(u[1] - 1) < 0.01, trajectories = 1000, seed = seed,
                                   ensemble = EnsembleSerial())
        b = bs(1)
        @test b.failed == 0 && b.trajectories == 1000
        @test abs(b.fraction - 0.5) < 3 * b.se
        @test bs(1).returned == b.returned                              # reproducible with a seed
    end

    @testset "early-warning indicators" begin
        @test DynamicalResilience.kendall_tau(1:10, 1:10) ≈ 1
        @test DynamicalResilience.kendall_tau(1:10, 10:-1:1) ≈ -1
        lin = early_warning(2.0 .* (1:50) .+ 3; detrend = :linear, window = 10)
        @test maximum(abs, lin.residuals) < 1e-10
        @test_throws ArgumentError early_warning(randn(10); window = 2)
        @test_throws ArgumentError early_warning(randn(10); detrend = :loess)

        # dx = (r − x²) dt + σ dW with r ramped from 1 towards the fold at r = 0: critical slowing
        # down raises variance and lag-1 autocorrelation. Kendall's τ of a single series is noisy
        # (rolling windows overlap), so compare means over seeds with an unramped control.
        T = 1000.0
        prob = SDEProblem((u, p, t) -> [p - u[1]^2], (u, p, t) -> [0.02], [1.0], (0.0, T), 1.0)
        ramp = (p0, t) -> 1.0 - 0.98t / T
        series(seed, rp) = ramped_series(prob, EM(); saveat = 0:1.0:T, ramp = rp, observe = u -> u[1], seed = seed,
                                         dt = 0.01, adaptive = false)
        ramped = [early_warning(series(s, ramp).x) for s in 1:10]
        still = [early_warning(series(s, nothing).x) for s in 1:10]
        @test series(1, ramp).x[end] ≈ sqrt(0.02) rtol = 0.5               # tracking the moving equilibrium
        @test mean(e.tau_variance for e in ramped) > 0.4
        @test mean(e.tau_ac1 for e in ramped) > 0.2
        @test abs(mean(e.tau_variance for e in still)) < 0.3
        @test abs(mean(e.tau_ac1 for e in still)) < 0.3
        @test all(e.variance[end] > e.variance[1] for e in ramped)
        @test all(e.ac1[end] > e.ac1[1] for e in ramped)
    end

    @testset "common random numbers make recovery of a noisy system precise" begin
        prob = SDEProblem((u, p, t) -> [-u[1] - u[1]^3], (u, p, t) -> [0.3], [0.0], (0.0, 10.0))
        at2(s, s2) = recovery(prob, EM(); perturb = (u, rng) -> u .+ 0.5, interval = 0.1, horizon = 5.0, seed = s,
                              perturbed_seed = s2, adaptive = false, dt = 0.01).distance[21]
        shared = [at2(s, s) for s in 1:30]
        independent = [at2(s, s + 1000) for s in 1:30]
        @test var(shared) < 0.05 * var(independent)
        @test mean(shared) < mean(independent)
    end
end
