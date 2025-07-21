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
        parent1, parent2 = parent2, parent1                    # man nimmt den fitteren als parent1 da dieser später disjunkte connections vererbt
    end

    # Prepare child structures
    child_nodes = Dict{Int,Node}()
    child_connections = Dict{Tuple{Int,Int},Connection}()

    # Map innovation_number => connection for quick lookup
    p1_innov = Dict(conn.innovation_number => conn for conn in values(parent1.connections))
    p2_innov = Dict(conn.innovation_number => conn for conn in values(parent2.connections))

    # Set of all innovation numbers from both parents
    all_innovs = union(keys(p1_innov), keys(p2_innov))          # alle innovation nummern von beiden eltern 

    for innov in all_innovs
        conn1 = get(p1_innov, innov, nothing)             # wir holen die connection mit innovation nummer innov , wenns keine gibt dann ist die nothing. Wir iterieren also über die ganzen Inno Nummern und schauen obs welche gibt die sich überschneiden. 
        conn2 = get(p2_innov, innov, nothing)             # selbiges hier

        inherited_conn = nothing

        if conn1 !== nothing && conn2 !== nothing         # wenn beide Eltern die gleiche Connection haben zwischen 2 Nodes dann wählen wir per Zufall die Connection die an das Kind weitergegeben wird !
            
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

        elseif conn1 !== nothing                          # das hier bedeutet ja dass conn2 nothing ist, also dass es für eine Connection die in p1 vorliegt, keine connection in p2 gibt. Da P1 immer fitter ist als P2 nehmen wir dessen Connection. Auch wenn diese in P2 nicht vorhanden ist !
            # Disjoint or excess gene from fitter parent
            inherited_conn = Connection(
                conn1.in_node,
                conn1.out_node,
                conn1.weight,
                conn1.enabled,
                conn1.innovation_number,
            )
        end

        if inherited_conn !== nothing                    # die Connections werden zum dictionary child_connections hinzugefügt
            key = (inherited_conn.in_node, inherited_conn.out_node)
            child_connections[key] = inherited_conn
        end
    end

    # Merge node dictionaries (union of all nodes used in both parents)
    all_nodes = merge(parent1.nodes, parent2.nodes)        # für die Nodes machen wir es so, dass immer alle Knoten vererbt werden. 

    for (nid, node) in all_nodes
        child_nodes[nid] = Node(nid, node.nodetype)
    end

    try
        genome_try = Genome(next_genome_id(), child_nodes, child_connections, 0.0, 0.0) # wir kreiieren ein neues genom mit fitness und adjusted fitness 0 sowie den entsprechenden connections und knoten
        _, _ = ForwardPass.topological_sort(genome_try)                                 # & testen dann auch noch ob die topological sort funktioniert
        return genome_try
    catch ex
        return parent1                                                                   # wenn nicht nehmen wir den besseren Elternteil und vergessen das ganze
    end
end

end
# Fazit dazu: Der Code ist mega ineffizient. Man könnte die  Knoten am Anfang berechnen und im Anschluss durch die Connections iterieren und falls eine Connection einen Cycle erzeugt diese Ablehnen. 
# Aktuell wird das Genom zusammengebaut nur um dann vielleicht abgeleht zu werden und der Parent1 stattdessen genutzt wird. 
# Da beide Netze aber immer azyklisch ist sind cycles recht selten. 

# -> Es kann am Ende Knoten geben die keine Connections haben. Das passiert weil alle Knoten von beiden Parents übernommen werden aber nur von Parent1 immer auch eine Connection (eventuell mit dem Gewicht von Parent2, die Richtung stimmt aber immer auch wenn random gewählt wird)
# Sollte nun eine Verbindung zwischen 2 Knoten innerhalb der vererbten Knoten vorkommen, so kann es sein dass die Connection nicht übernommen wird, da Knoten zwar von beiden Eltern, die Connections aber immer nur von Parent1 übernommen werden. 
