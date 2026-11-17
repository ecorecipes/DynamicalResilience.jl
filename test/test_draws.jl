using SciMLBase: DiscreteProblem
using Statistics: quantile

logistic_ode(u, p, t) = [p.r * u[1] * (1 - u[1] / p.K)]
fold_normal_form(u, p, t) = [p + u[1]^2]      # equilibria ±√(−p) for p < 0, none for p > 0

@testset "metrics over draws" begin
    # logistic growth: equilibrium K, resilience r, reactivity −r
    draws = [(r = r, K = K) for (r, K) in zip(0.5:0.25:2.0, 1.0:0.5:4.0)]
    # start above every K: from below K/2 Newton correctly finds the unstable equilibrium at 0
    build(d) = ODEProblem(logistic_ode, [5.0], (0.0, 1.0), d)
    out = metrics_over(build, draws)
    @test out.n == length(draws)
    @test out.n_converged == out.n_stable == length(draws)
    @test out.n_unconverged == out.n_unstable == 0
    @test [only(u) for u in out.equilibrium] ≈ [d.K for d in draws]
    @test out.metrics.resilience ≈ [d.r for d in draws]
    @test out.metrics.reactivity ≈ [-d.r for d in draws]
    @test out.quantiles.resilience ≈ quantile([d.r for d in draws], [0.025, 0.5, 0.975])
    @test out.probabilities == [0.025, 0.5, 0.975]

    # the same as a hand loop over steady_state and linearise
    hand = map(draws) do d
        prob = build(d)
        ss = steady_state(prob, [5.0])
        resilience(linearise(prob, ss.u))
    end
    @test out.metrics.resilience ≈ hand

    # without warm starts the results are the same here
    cold = metrics_over(build, draws; warm_start = false)
    @test cold.metrics.resilience ≈ out.metrics.resilience

    # unstable and missing equilibria are counted, not dropped
    sn = metrics_over(p -> ODEProblem(fold_normal_form, [1.0], (0.0, 1.0), p), [-1.0, -0.25, 0.5];
                      warm_start = false)
    @test sn.n == 3
    @test sn.converged == [true, true, false]
    @test sn.stable == [false, false, false]       # Newton from u = 1 finds the unstable +√(−p)
    @test sn.n_unconverged == 1
    @test sn.n_unstable == 2
    @test isnan(sn.metrics.resilience[3])
    @test sn.metrics.resilience[1:2] ≈ [-2.0, -1.0]  # −2√(−p)
    @test all(isnan, sn.quantiles.resilience)         # no stable draw

    # discrete-time problems work too: the logistic map's fixed point has multiplier 2 − r
    rs = [2.2, 2.5, 2.8, 3.3]
    dm = metrics_over(r -> DiscreteProblem((u, p, t) -> [p * u[1] * (1 - u[1])], [0.4], (0, 1), r), rs;
                      metrics = (resilience,))
    @test dm.n_converged == 4
    @test dm.stable == [true, true, true, false]
    @test dm.metrics.resilience ≈ [-log(abs(2 - r)) for r in rs]
    @test dm.quantiles.resilience ≈ quantile([-log(abs(2 - r)) for r in rs[1:3]], [0.025, 0.5, 0.975])
end
