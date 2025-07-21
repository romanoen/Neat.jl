module ForwardPass

using ..Types

export forward_pass

"""
    forward_pass(genome::Genome, input::Vector{Float64}) → Dict{Int, Float64}

Performs a forward pass through the network defined by `genome`, computing activation values for all nodes.

# Arguments
- `genome::Genome`: The genome containing nodes and connections.
- `input::Vector{Float64}`: Activation values for input nodes.

# Returns
- `Dict{Int, Float64}`: A dictionary mapping each node ID to its activation value.
"""
function forward_pass(genome::Genome, input::Vector{Float64})::Dict{Int, Float64}
    # Determine evaluation order via topological sorting and get enabled_conns
    sorted_nodes, enabled_conns = topological_sort(genome)                                # wir sortieren topologisch


    # Identify and sort input nodes to align with provided input vector
    input_nodes = sort([n.id for n in values(genome.nodes) if n.nodetype == :input])        # wir identifizieren welche Knoten Input Knoten sind

    # Prepare activation storage
    activations = Dict{Int, Float64}()                                                    # das hier dient als Speicher für die berechneten ergebnisse nach jedem knoten. Id ist die 

    # Assign provided values to input nodes in sorted order
    for (i, nid) in enumerate(input_nodes)                                                # Wir fügen die Input Nodes direkt mitsamt ihrer Werte in das activations dictionary ein. Es muss hier nichts berechnet werden, wir haben ja bereits die Werte der Input Nodes!
        activations[nid] = input[i]
    end


    # Compute activations for non-input nodes
    for node in sorted_nodes                                                               # wir schauen uns jeden Node in unseren sorted_nodes nacheinander an
        # Skip input nodes—they already have values
        if genome.nodes[node].nodetype == :input                                            # die ersten input nodes skippen wir, da haben wir die Ergebnisse ja schon. 
            continue
        end

        # Sum weighted inputs
        sum_input = 0.0                                                                    # jez gehts ab. wir setzen die Summe auf 0. Wir schauen welche Eingehenden Kanten hat unser node. Dann schauen wir uns diese Kante an. Welchen Startknoten hat diese Kante ?!
        for conn in enabled_conns                                                          # Wir berechnen dann den Wert am Eingangspunkt der Kante * das Gewicht der Kante.
            if conn.out_node == node                                                       # So haben wir für jeden Knoten berechnet wie der Wert direkt nach dem Knoten ist
                sum_input += activations[conn.in_node] * conn.weight
            end
        end

        # Apply sigmoid activation function
        activations[node] = 1.0 / (1.0 + exp(-sum_input))                                   # wir speichern jetzt nur noch den neu Berechneten Knoten ab indem wir die Sigmoid benutzen -> wichtig  damit das Netz nichtlinear lernen kann 
    end

    return activations
end

"""
    topological_sort(genome::Genome) → Vector{Int}

Performs a topological sort of all nodes in the `genome`.
The resulting order ensures each node appears only after all its predecessors have been processed.

# Arguments
- `genome::Genome`: The genome containing nodes and connections.

# Returns
- `Vector{Int}`: A list of node IDs in a valid computation order.
- `Vector{Connection}`: A list of all enabled connections in the genome.

# Errors
- Throws an error if the graph contains cycles, making topological sorting impossible.
"""
function topological_sort(genome::Genome)
    # All node IDs
    nodes = collect(keys(genome.nodes))                                                # wir fetzen uns die node id
    # Only consider enabled connections
    enabled_conns = [c for c in values(genome.connections) if c.enabled]                # wir fetzen uns die enabled_connections

    # 1) Count in-degrees for each node
    in_degree = Dict(n => 0 for n in nodes)                                            # wir berechnen wieviele eingehende Kanten ein Knoten hat (man schaut sich die enabled connections an und setzt den counter im dict hoch wenn der ausgehende knoten einmal da ist) 
    for conn in enabled_conns
        in_degree[conn.out_node] += 1
    end

    # 2) Initialize list of nodes with zero in-degree
    no_incoming = [n for n in nodes if in_degree[n] == 0]                                # dict von Nodes die Keine Abhängigkeiten haben weil alles davor schon berechnet wurde. Wir können den Wert dieser Knoten also problemlos berechnen. 
    order = Int[]  # Sorted order container                                              # die sortierreihenfolge 

    # Kahn's algorithm for topological sort
    while !isempty(no_incoming)                                                         # solange es noch nodes gibt die keine eingehenden Kanten haben                                                    
        n = popfirst!(no_incoming)                                                      # wir nehmen den ersten Knoten aus no_incoming
        push!(order, n)                                                                 #packen ihn in die sortierreihenfolge (der Sinn ist, dass wir so ganz vorne in der Sortierreihenfolge die knoten ohne eingehende kanten haben)

        # Remove outgoing edges from n
        for conn in enabled_conns                        
            if conn.in_node == n                                                         # Wir schauen uns alle ausgehenden Verbindungen von n an               
                out = conn.out_node                                                       # und checken welcher Knoten an der anderen Seite der Connection ist
                in_degree[out] -= 1                                                        # Dann verringern wir den counter für die Abhängigkeiten von dem Knoten auf der anderen Seite der Connection
                if in_degree[out] == 0
                    push!(no_incoming, out)                                                # Wenn der Knoten am Ende der Verbindung zwischen n und out garkeine Abhängigkeiten mehr hat, dann packen wir ihn in no_incoming ! Er ist nun bereit berechnet zu werden !!
                end
            end
        end
    end

    # 3) If not all nodes are processed, a cycle exists
    if length(order) != length(nodes)
        error("Graph contains cycles! Topological sort not possible.")                # wenn nicht alle knoten berechnet sind ist iwo ein cycle drin
    end

    return order, enabled_conns
end

end # module ForwardPass
