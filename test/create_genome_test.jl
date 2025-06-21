using Test
using Neat

@testset "create_genome" begin
    genome = create_genome(1, 2, 1)

    # --- Check basic genome structure ---
    @test isa(genome, Genome)
    @test genome.id == 1
    @test length(genome.nodes) == 3  # 2 inputs + 1 output
    @test length(genome.connections) == 2

    # --- Check node types ---
    @test genome.nodes[1].nodetype == :input
    @test genome.nodes[2].nodetype == :input
    @test genome.nodes[3].nodetype == :output

    # --- Check connections ---
    @test haskey(genome.connections, (1, 3))
    @test haskey(genome.connections, (2, 3))

    conn1 = genome.connections[(1, 3)]
    conn2 = genome.connections[(2, 3)]

    @test conn1.enabled == true
    @test conn2.enabled == true

    @test conn1.in_node == 1 && conn1.out_node == 3
    @test conn2.in_node == 2 && conn2.out_node == 3

    @test conn1.innovation_number == 1
    @test conn2.innovation_number == 2
end
