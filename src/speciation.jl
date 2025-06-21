module Speciation

using ..Types

export compatibility_distance
export assign_species!
export adjust_fitness!

"""
    compatibility_distance(g1::Genome, g2::Genome;
                           c1=1.0, c2=1.0, c3=0.4) → Float64

Compute the compatibility distance between two genomes `g1` and `g2`.
Uses NEAT's distance formula:

    δ = (c1 * E / N) + (c2 * D / N) + (c3 * W)

- E = number of excess genes
- D = number of disjoint genes
- W = average weight difference of matching genes
- N = number of genes in the larger genome (set to 1 if small for stability)

# Keyword Arguments
- `c1`, `c2`, `c3`: importance of each factor

# Returns
- A `Float64` distance value (lower means more similar)
"""
function compatibility_distance(g1::Genome, g2::Genome;
                                 c1=1.0, c2=1.0, c3=0.4)

    # Build lookup tables for connections in each genome
    conns1 = Dict(c.innovation_number => c for c in values(g1.connections))
    conns2 = Dict(c.innovation_number => c for c in values(g2.connections))

    # Collect innovation numbers from both genomes
    innovs1 = keys(conns1)
    innovs2 = keys(conns2)
    all_innovs = union(innovs1, innovs2)

    # Determine the highest innovation number in each genome
    max_innov1 = isempty(innovs1) ? 0 : maximum(innovs1)
    max_innov2 = isempty(innovs2) ? 0 : maximum(innovs2)
    max_innov = max(max_innov1, max_innov2)

    # Initialize distance components
    D = 0       # Disjoint gene count
    E = 0       # Excess gene count
    W = 0.0     # Sum of weight differences for matching genes
    M = 0       # Count of matching genes

    # Iterate through all innovation numbers seen in either genome
    for innov in all_innovs
        c1_conn = get(conns1, innov, nothing)
        c2_conn = get(conns2, innov, nothing)

        if c1_conn !== nothing && c2_conn !== nothing
            # Matching gene: both genomes have this innovation number
            W += abs(c1_conn.weight - c2_conn.weight)
            M += 1
        elseif innov <= max_innov1 && innov <= max_innov2
            # Disjoint gene: occurs within range of both genomes
            D += 1
        else
            # Excess gene: occurs beyond the range of one genome
            E += 1
        end
    end

    # Average weight difference for matching genes
    avg_weight_diff = M > 0 ? W / M : 0.0

    # Normalize by the size of the larger genome (or 1 if both are small)
    N = max(length(conns1), length(conns2))
    N = N < 20 ? 1 : N

    # NEAT compatibility distance formula
    return (c1 * E / N) + (c2 * D / N) + (c3 * avg_weight_diff)
end

"""
    assign_species!(population::Vector{Genome}, species_list::Vector{Vector{Genome}};
                    threshold=3.0)

Assign each genome in the population to a species in `species_list` based on
compatibility distance. A genome is added to the first species where distance
to the representative is below the threshold. If no such species exists, a new
species is created with this genome.

# Arguments
- `population`: Vector of genomes to classify
- `species_list`: Vector of species (each a vector of genomes)
- `threshold`: Maximum allowed compatibility distance to join a species

"""
function assign_species!(population::Vector{Genome}, species_list::Vector{Vector{Genome}};
                         threshold::Float64=3.0)

    # Clear existing species
    empty!(species_list)

    for genome in population
        assigned = false

        for species in species_list
            representative = species[1]  # pick first genome as representative
            dist = compatibility_distance(genome, representative)

            if dist < threshold
                push!(species, genome)
                assigned = true
                break
            end
        end

        # If no compatible species found, create a new one
        if !assigned
            push!(species_list, [genome])
        end
    end
end

"""
    adjust_fitness!(species_list::Vector{Vector{Genome}})

Modifies each genome's fitness value in-place by applying NEAT-style fitness sharing:
- Divides each genome's fitness by the number of members in its species.

# Arguments
- `species_list`: A vector of species, where each species is a vector of genomes.

# Side Effect
- Overwrites each genome's `fitness` value with the adjusted fitness.
"""
function adjust_fitness!(species_list::Vector{Vector{Genome}})
    for species in species_list
        s_size = length(species)
        for genome in species
            genome.fitness /= s_size
        end
    end
end

end
