using Test
using MyPackage

@testset "evaluate_fitness" begin
    # Manually define a minimal genome for XOR
    nodes = Dict(1 => Node(1, :input), 2 => Node(2, :input), 3 => Node(3, :output))

    connections = Dict(
        (1, 3) => Connection(1, 3, 1.0, true, 1), (2, 3) => Connection(2, 3, 1.0, true, 2)
    )

    genome = Genome(1, nodes, connections)

    fitness = evaluate_fitness(genome)

    # Because weights are both 1.0, XOR error will be large and negative
    # But we just check that it returns a Float64 and is finite
    @test isa(fitness, Float64)
    @test isfinite(fitness)
    @test fitness < 0  # Since the error is squared and negated
end