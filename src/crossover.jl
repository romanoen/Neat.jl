module Crossover

using ..Types
using ..ForwardPass
using Random
using ..CreateGenome: next_genome_id
using ..NeatConfig

export crossover

"""
    crossover(parent1::Genome, parent2::Genome) → Genome

Perform crossover between two parent genomes.  
The fitter parent contributes all disjoint and excess genes.  
Matching genes are randomly inherited from either parent.  
Disabled genes may remain disabled in the child.

# Arguments
- `parent1::Genome`: One parent genome.
- `parent2::Genome`: Another parent genome.

# Returns
- `Genome`: A new child genome composed from both parents' genes.
"""
function crossover(parent1::Genome, parent2::Genome,disable_chance::Float64=get_config()["crossover"]["disable_chance"])
    # Ensure parent1 is the fitter (or equal fitness: keep order)
    if parent2.fitness > parent1.fitness
        parent1, parent2 = parent2, parent1
    end

    # Prepare child structures
    child_nodes = Dict{Int,Node}()
    child_connections = Dict{Tuple{Int,Int},Connection}()

    # Map innovation_number => connection for quick lookup
    p1_innov = Dict(conn.innovation_number => conn for conn in values(parent1.connections))
    p2_innov = Dict(conn.innovation_number => conn for conn in values(parent2.connections))

    # Set of all innovation numbers from both parents
    all_innovs = union(keys(p1_innov), keys(p2_innov))

    for innov in all_innovs
        conn1 = get(p1_innov, innov, nothing)
        conn2 = get(p2_innov, innov, nothing)

        inherited_conn = nothing

        if conn1 !== nothing && conn2 !== nothing
            # Matching gene: randomly choose from either parent
            selected = rand(Bool) ? conn1 : conn2

            # If either is disabled, 75% chance to remain disabled
            enabled = (!conn1.enabled || !conn2.enabled) ? (rand() > disable_chance) : selected.enabled

            inherited_conn = Connection(
                selected.in_node,
                selected.out_node,
                selected.weight,
                enabled,
                selected.innovation_number,
            )

        elseif conn1 !== nothing
            # Disjoint or excess gene from fitter parent
            inherited_conn = Connection(
                conn1.in_node,
                conn1.out_node,
                conn1.weight,
                conn1.enabled,
                conn1.innovation_number,
            )
        end

        if inherited_conn !== nothing
            key = (inherited_conn.in_node, inherited_conn.out_node)
            child_connections[key] = inherited_conn
        end
    end

    # Merge node dictionaries (union of all nodes used in both parents)
    all_nodes = merge(parent1.nodes, parent2.nodes)

    for (nid, node) in all_nodes
        child_nodes[nid] = Node(nid, node.nodetype)
    end

    try
        genome_try = Genome(next_genome_id(), child_nodes, child_connections, 0.0, 0.0)
        _, _ = ForwardPass.topological_sort(genome_try)
        return genome_try
    catch ex
        return parent1
    end
end

end
