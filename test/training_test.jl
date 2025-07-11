using Test
using NeatTrain
using Types: Genome

@testset "NeatTrain.train tests" begin
    # Test zero generations: should return initial population and empty history
    pop_size = 5
    gens = 0
    population, history = train(pop_size=pop_size, n_generations=gens, ds_name="XOR_DATA")
    @test length(population) == pop_size
    @test isa(population, Vector{Genome})
    @test history == Float64[]

    # Test one generation evolution: history length matches n_generations
    pop_size2 = 4
    gens2 = 1
    population2, history2 = train(pop_size=pop_size2, n_generations=gens2, ds_name="XOR_DATA")
    @test length(population2) == pop_size2
    @test length(history2) == gens2
    @test isa(history2[1], Float64)

    # Test that each genome has an assigned fitness
    @test all(g -> isa(g.fitness, Float64), population2)
end
