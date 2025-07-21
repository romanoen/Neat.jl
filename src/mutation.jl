module Mutation

using ..Types
using ..Innovation
using ..NeatConfig
using Random

export mutate_weights!, mutate, add_connection!, add_node!, causes_cycle

"""
    mutate_weights!(genome::Genome; perturb_chance::Float64, sigma::Float64) -> Nothing

Apply weight mutations to all connections of `genome` by replacing each with a new modified connection.

- With probability `perturb_chance`, add a perturbation drawn from `Normal(0, sigma)`.
- Otherwise, assign a completely new weight sampled from `Normal(0, 1)`.

# Arguments
- `genome` : The `Genome` whose connection weights will be mutated.
- `perturb_chance` : Probability of choosing a small perturbation over full replacement.
- `sigma` : Standard deviation of the Gaussian perturbation.
"""
function mutate_weights!(genome::Genome; perturb_chance::Float64, sigma::Float64) # Im Endeffekt super easy perturb macht ein komplettes Gewicht und sigma ändert das gewicht ein bisschen und zwar für alle Connections innerhalb eines Genoms
    for key in keys(genome.connections)
        conn = genome.connections[key]
        new_weight = rand() < perturb_chance ? conn.weight + randn() * sigma : randn()
        genome.connections[key] = Connection(                                    # Anscheinend effizienter eine neue Connection zuzuweisen
            conn.in_node, conn.out_node,
            new_weight, conn.enabled,
            conn.innovation_number
        )
    end
end

"""
    causes_cycle(genome::Genome, src_id::Int, dst_id::Int) -> Bool

Determine whether adding a connection from node `src_id` to `dst_id` would introduce a cycle.

Performs a depth-first search starting from `dst_id` to see if `src_id` is reachable.

# Arguments
- `genome` : The `Genome` whose topology is checked.
- `src_id` : Identifier of the potential source node.
- `dst_id` : Identifier of the potential destination node.

# Returns
- `true` if a path exists from `dst_id` back to `src_id` (a cycle would form), otherwise `false`.
"""
function causes_cycle(genome::Genome, src_id::Int, dst_id::Int)::Bool # man startet bei dst_id 
    visited = Set{Int}()                                              # der Stack mit den ganzen visited Nodes
    stack = [dst_id]                                                 # enthält alle knoten die wir noch besuchen wollen

    while !isempty(stack)
        current = pop!(stack)
        if current == src_id                                        # wenn der startknoten gefunden ist, dann gibts nen cycle
            return true
        end
        if current in visited                                        # wir müssen verhindern dass knoten doppelt in stack sind
            continue
        end
        push!(visited, current)                                    # wir fügen den current node hinzu zu visited
        for (_, conn) in genome.connections                        
            if conn.enabled && conn.in_node == current            # in dieser schleife fügen wir dann alle nodes hinzu, die von current aus zu erreichen sind (Tiefensuche) 
                push!(stack, conn.out_node)
            end
        end
    end
    return false
end

"""
    add_connection!(genome::Genome; max_attempts::Int=50) -> Nothing

Try up to `max_attempts` times to add a new connection between two previously unconnected nodes without creating a cycle.

Selects random input and output nodes (skipping invalid or existing edges) until a valid pair is found or attempts exhausted.

# Keyword Arguments
- `max_attempts` : Maximum number of trials before giving up (default: 50).
"""
function add_connection!(genome::Genome; max_attempts::Int)
    nodes = collect(values(genome.nodes))                                                            # wir schauen uns alle nodes an
    attempts = 0

    while attempts < max_attempts
        in_node = rand(nodes)                                                                      # wählen zufällig einen input und output node
        out_node = rand(nodes)

        if in_node.id == out_node.id || in_node.nodetype == :output || out_node.nodetype == :input
            attempts += 1; continue
        end

        key = (in_node.id, out_node.id)
        if haskey(genome.connections, key) || causes_cycle(genome, in_node.id, out_node.id) # wenns die connection schon gibt oder wenn sie einen cycle verursachen würde lehnen wir sie ab
            attempts += 1; continue
        end

        innovation_number = get_innovation_number(in_node.id, out_node.id)         # wir holen eine neue innovation number
        genome.connections[key] = Connection(                                     # und generieren die connection
            in_node.id,
            out_node.id,
            randn(),
            true,
            innovation_number,
        )
        return nothing
    end
end

"""
    add_node!(genome::Genome) -> Nothing

Insert a new hidden node by splitting an existing enabled connection.

- Chooses a random enabled connection A → B and disables it.
- Creates a new hidden node C with a unique ID.
- Adds two connections: A → C (weight = 1.0) and C → B (weight equal to the original connection).

This mutation allows the network topology to grow.

# Arguments
- `genome` : The `Genome` to modify.
"""
function add_node!(genome::Genome)
    active_connections = [c for c in values(genome.connections) if c.enabled]
    isempty(active_connections) && return nothing

    old_conn = rand(active_connections)                                        # wir ziehen uns ne random connection
    key = (old_conn.in_node, old_conn.out_node)

    genome.connections[key] = Connection(                                        # deaktivieren diese
        old_conn.in_node,
        old_conn.out_node,
        old_conn.weight,
        false,
        old_conn.innovation_number,
    )

    # Create new node
    new_node_id = maximum(keys(genome.nodes)) + 1                        # erzeugen nen neuen node id

    genome.nodes[new_node_id] = Node(new_node_id, :hidden)                  # hidden node wird erzeugt

    # Add connection A → C
    new_innov1 = get_innovation_number(old_conn.in_node, new_node_id)       # wir bauen die neue connection zu dem neuen Knoten hin 
    genome.connections[(old_conn.in_node, new_node_id)] = Connection(
        old_conn.in_node, new_node_id, 1.0, true, new_innov1
    )

    # Add connection C → B
    new_innov2 = get_innovation_number(new_node_id, old_conn.out_node)     # und vom neuen knoten weg, sodass es sich am ende wieder schließt
    genome.connections[(new_node_id, old_conn.out_node)] = Connection(
        new_node_id, old_conn.out_node, old_conn.weight, true, new_innov2
    )
end

"""
    mutate(genome::Genome; perturb_chance, sigma, add_connection_prob, node_add_prob, max_attempts) -> Nothing

Apply a full suite of mutation operators to `genome` according to configured probabilities.

- Weight mutations via `mutate_weights!`.
- With probability `add_connection_prob`, attempt `add_connection!`.
- With probability `node_add_prob`, attempt `add_node!`.

If any probability or `max_attempts` is not defined, the value is loaded from the `mutation` section of the configuration.

# Keyword Arguments
- `perturb_chance` : Probability for weight perturbation.
- `sigma` : Stddev for weight perturbation.
- `add_connection_prob` : Chance to add a new connection.
- `node_add_prob` : Chance to add a new node.
- `max_attempts` : Max trials for adding a connection.
"""
function mutate(genome::Genome;
    perturb_chance::Float64=get_config()["mutation"]["perturb_chance"],
    sigma::Float64=get_config()["mutation"]["sigma"],
    add_connection_prob::Float64=get_config()["mutation"]["add_connection_prob"],
    node_add_prob::Float64=get_config()["mutation"]["node_add_prob"],
    max_attempts::Int=get_config()["mutation"]["max_attempts"],
)
    mutate_weights!(genome; perturb_chance=perturb_chance, sigma=sigma)

    if rand() < add_connection_prob
        add_connection!(genome; max_attempts=max_attempts)
    end

    if rand() < node_add_prob
        add_node!(genome)
    end
end

end # module Mutation
