# saddle-node normal form dx/dt = μ − x²: the stable branch x = √μ has resilience 2√μ
saddle_node(u, μ, t) = [μ - u[1]^2]

# Hopf normal form in Cartesian coordinates, with frequency ω = 1
hopf(u, μ, t) = [μ * u[1] - u[2] - u[1] * (u[1]^2 + u[2]^2),
                 u[1] + μ * u[2] - u[2] * (u[1]^2 + u[2]^2)]

@testset "resilience_along" begin
    prob = ODEProblem(saddle_node, [1.0], (0.0, 1.0), 1.0)
    μs = collect(1.0:-0.07:-0.2)                    # …, 0.02, −0.05, …: the fold at 0 is not on the grid
    path = resilience_along(prob, (p, v) -> v, μs; u0 = [1.0])
    ok = μs .> 0
    @test all(path.converged[ok])
    @test !any(path.converged[.!ok])
    @test path.metrics.resilience[ok] ≈ 2 .* sqrt.(μs[ok]) rtol = 1e-8
    @test all(path.stable[ok])
    @test path.transition.kind === :fold
    @test -0.05 <= path.transition.estimate <= 0.02
    @test isreal(path.leading[1])

    prob = ODEProblem(hopf, [0.0, 0.0], (0.0, 1.0), -1.0)
    μs = collect(-1.0:0.15:0.5)                     # …, −0.1, 0.05, …
    path = resilience_along(prob, (p, v) -> v, μs; u0 = [0.0, 0.0])
    @test all(path.converged)
    @test path.metrics.resilience ≈ -μs atol = 1e-10
    @test path.stable == (μs .< 0)
    @test path.transition.kind === :hopf
    @test abs(path.transition.estimate) < 1e-8
    @test imag(path.leading[1]) ≈ 1.0
    @test leading_eigenvalue(linearise(prob, [0.0, 0.0]; p = -0.3)) ≈ complex(-0.3, 1.0)

    # no loss of stability along the path
    path = resilience_along(prob, (p, v) -> v, [-1.0, -0.5]; u0 = [0.0, 0.0])
    @test path.transition === nothing
end
