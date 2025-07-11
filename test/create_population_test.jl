using Test
using Neat

@testset "initialize_population" begin
    num_genomes = 10
    num_inputs = 3
    num_outputs = 2

    population = initialize_population(num_genomes, num_inputs, num_outputs)

    @test length(population) == num_genomes

    for genome in population
        # Test node counts
        input_nodes = [n for n in values(genome.nodes) if n.nodetype == :input]
        output_nodes = [n for n in values(genome.nodes) if n.nodetype == :output]
        
        @test length(input_nodes) == num_inputs
        @test length(output_nodes) == num_outputs

        # Test connections: fully connected from inputs to outputs
        expected_connection_count = num_inputs * num_outputs
        @test length(genome.connections) == expected_connection_count

        for (key, conn) in genome.connections
            @test conn.enabled == true
            @test conn.in_node in [n.id for n in input_nodes]
            @test conn.out_node in [n.id for n in output_nodes]
        end
    end
end
