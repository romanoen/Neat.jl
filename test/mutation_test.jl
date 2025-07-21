using Test
using Neat
using Neat.Mutation: causes_cycle, add_connection!, add_node!, mutate_weights!, mutate
using Neat.CreateGenome: create_genome
using Neat.Types: Genome
@testset "Mutation tests" begin
    @testset "mutate_weights!" begin
        genome = create_genome(2, 1)
        old_weights = [c.weight for c in values(genome.connections)]
        mutate_weights!(genome; perturb_chance=1.0, sigma=0.1)
        new_weights = [c.weight for c in values(genome.connections)]
        @test any(abs.(old_weights .- new_weights) .> 0)
    end

    @testset "add_connection! with hidden node" begin
        genome = create_genome(2, 1)
        add_node!(genome)
        num_connections_before = length(genome.connections)
        add_connection!(genome; max_attempts=100)
        num_connections_after = length(genome.connections)
        @test num_connections_after >= num_connections_before
    end
    @testset "add_connection! adds or preserves graph" begin
        genome = create_genome(2, 1)
        add_node!(genome)
        connections_before = length(genome.connections)

        add_connection!(genome; max_attempts=100)

        connections_after = length(genome.connections)
        @test connections_after >= connections_before
    end
    @testset "add_node!" begin
        genome = create_genome(2, 1)
        num_nodes_before = length(genome.nodes)
        num_connections_before = length(genome.connections)
        add_node!(genome)
        num_nodes_after = length(genome.nodes)
        num_connections_after = length(genome.connections)
        @test num_nodes_after == num_nodes_before + 1
        @test num_connections_after == num_connections_before + 2

        num_disabled = count(!c.enabled for c in values(genome.connections))
        @test num_disabled == 1
    end

    @testset "mutate (full)" begin
        genome = create_genome(2, 1)
        mutate(genome; max_attempts=100)
        @test length(genome.nodes) >= 3
        @test all(
            conn.in_node in keys(genome.nodes) && conn.out_node in keys(genome.nodes) for
            conn in values(genome.connections)
        )
    end
end

# @testset "add_connection! should not introduce cycles" begin
#     genome = create_genome(2, 1)

#     function has_cycle(genome::Genome)::Bool
#         visited = Set{Int}()

#         function dfs(node_id::Int, stack::Set{Int})
#             if node_id in stack
#                 return true
#             end
#             if node_id in visited
#                 return false
#             end

#             push!(visited, node_id)
#             push!(stack, node_id)

#             for conn in values(genome.connections)
#                 if conn.enabled && conn.in_node == node_id
#                     if dfs(conn.out_node, stack)
#                         return true
#                     end
#                 end
#             end

#             delete!(stack, node_id)
#             return false
#         end

#         for node_id in keys(genome.nodes)
#             if dfs(node_id, Set{Int}())
#                 return true
#             end
#         end
#         return false
#     end

#     for _ in 1:50
#         add_connection!(genome; max_attempts=100)
#         @test !has_cycle(genome)
#     end
# end

