# SIR with births and deaths (R0 = β / (γ + μ)) and SEIR (R0 = β σ / ((σ + μ)(γ + μ)))
sir(u, p, t) = [p.μ - p.β * u[1] * u[2] - p.μ * u[1],
                p.β * u[1] * u[2] - (p.γ + p.μ) * u[2],
                p.γ * u[2] - p.μ * u[3]]
sir_new(u, p, t) = [p.β * u[1] * u[2]]

function seir!(du, u, p, t)
    S, E, I, R = u
    du[1] = p.μ - p.β * S * I - p.μ * S
    du[2] = p.β * S * I - (p.σ + p.μ) * E
    du[3] = p.σ * E - (p.γ + p.μ) * I
    du[4] = p.γ * I - p.μ * R
    return nothing
end
seir_new(u, p, t) = [p.β * u[1] * u[3], 0.0]

@testset "reproduction_number" begin
    for β in (0.5, 2.0)
        p = (β = β, γ = 0.5, μ = 0.02)
        prob = ODEProblem(sir, [1.0, 0.0, 0.0], (0.0, 1.0), p)
        res = reproduction_number(prob, [1.0, 0.0, 0.0]; infected = [2], new_infections = sir_new)
        @test res.R0 ≈ β / (p.γ + p.μ)
        # the disease-free equilibrium is stable exactly when R0 < 1
        @test isstable(linearise(prob, [1.0, 0.0, 0.0])) == (res.R0 < 1)
    end
    for β in (0.3, 1.5)
        p = (β = β, σ = 0.2, γ = 0.25, μ = 0.01)
        prob = ODEProblem(seir!, [1.0, 0.0, 0.0, 0.0], (0.0, 1.0), p)
        res = reproduction_number(prob, [1.0, 0.0, 0.0, 0.0]; infected = [2, 3], new_infections = seir_new)
        @test res.R0 ≈ β * p.σ / ((p.σ + p.μ) * (p.γ + p.μ))
        @test size(res.K) == (2, 2)
        @test isstable(linearise(prob, [1.0, 0.0, 0.0, 0.0])) == (res.R0 < 1)
    end
    p = (β = 2.0, γ = 0.5, μ = 0.02)
    prob = ODEProblem(sir, [1.0, 0.0, 0.0], (0.0, 1.0), p)
    @test_throws ArgumentError reproduction_number(prob, [0.5, 0.0, 0.0]; infected = [2], new_infections = sir_new)
    @test_throws ArgumentError reproduction_number(prob, [1.0, 0.0, 0.0]; infected = [2, 3], new_infections = sir_new)
end
