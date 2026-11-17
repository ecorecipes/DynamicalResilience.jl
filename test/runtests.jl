using DynamicalResilience
using LinearAlgebra
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: ODEProblem
using Test

@testset "DynamicalResilience" begin
    include("test_steady.jl")
    include("test_reproduction.jl")
    include("test_periodic.jl")
    include("test_continuation.jl")
end
