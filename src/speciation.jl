module Speciation

using ..Types
using Random
using ..NeatConfig: get_config

export compatibility_distance
export assign_species!
export adjust_fitness!
export compute_offspring_counts
export select_elites

"""
    compatibility_distance(g1::Genome, g2::Genome; c1::Union{Float64,Missing}=missing, c2::Union{Float64,Missing}=missing, c3::Union{Float64,Missing}=missing) -> Float64

Compute the NEAT compatibility distance between two genomes:

    δ = (c1 * E / N) + (c2 * D / N) + (c3 * W)

where
- `E` is the number of excess genes,
- `D` is the number of disjoint genes,
- `W` is the average weight difference of matching genes,
- `N` is the number of genes in the larger genome (treated as 1 if small for stability).

If any of `c1`, `c2`, or `c3` is not defined, its value is loaded from the speciation section of the configuration.

# Arguments
- `g1`, `g2` : The two `Genome` instances to compare.
- `c1` : Coefficient for excess genes.
- `c2` : Coefficient for disjoint genes.
- `c3` : Coefficient for average weight differences.

# Returns
- `Float64` : The compatibility distance (lower means more similar).
"""
function compatibility_distance(g1::Genome, g2::Genome;                        # wir können immer nur die distanz zwischen 2 genomen berechnen
    c1::Float64                  = get_config()["speciation"]["c1"],
    c2::Float64                  = get_config()["speciation"]["c2"],
    c3::Float64                  = get_config()["speciation"]["c3"],
    )


    # Build lookup tables for connections in each genome
    conns1 = Dict(c.innovation_number => c for c in values(g1.connections))        # wieder mal lookup tables für die ganzen connections
    conns2 = Dict(c.innovation_number => c for c in values(g2.connections))

    # Collect innovation numbers
    innovs1 = keys(conns1)                                                        # alle innovationsnummern
    innovs2 = keys(conns2)
    all_innovs = union(innovs1, innovs2)                                            # die zusammen gepackt

    # Determine the highest innovation number in each genome
    max_innov1 = isempty(innovs1) ? 0 : maximum(innovs1)                        # dann schauen wir nach den jeweils höchsten innovationsnummern
    max_innov2 = isempty(innovs2) ? 0 : maximum(innovs2)

    # Initialize distance components
    D = 0       # Disjoint gene count
    E = 0       # Excess gene count
    W = 0.0     # Sum of weight differences for matching genes
    M = 0       # Count of matching genes

    # Iterate through all innovation numbers seen in either genome
    for innov in all_innovs                                                    # jetzt schleife: wir schauen uns für jede inno nummmer die connection
        c1_conn = get(conns1, innov, nothing)                                    # in beiden genen an
        c2_conn = get(conns2, innov, nothing)

        if c1_conn !== nothing && c2_conn !== nothing                            # wenns bei beiden genomen die connection gibt
            # Matching gene
            W += abs(c1_conn.weight - c2_conn.weight)                            # berechnen wir den gewicht unterschied
            M += 1
        elseif innov <= max_innov1 && innov <= max_innov2                        # wenn nur bei einem und die innov nummer kleiner als die maxima sind ist es disjoint
            # Disjoint gene
            D += 1
        else
            # Excess gene
            E += 1                                                                   # sonst excess
        end
    end

    # Compute average weight difference
    avg_weight_diff = M > 0 ? W / M : 0.0                                       # hier rechnen wir die durchschnittlichen gewichtsunterschiede aus indem 
                                                                                # wir den gesamtgewichtsunterschied durch die anzahl der gleichen Connections teilen

    # Normalize by the size of the larger genome
    N = max(length(conns1), length(conns2))                                       # wir schauen nach der maximalen anzahl der connections aus beiden genomen und nehmen das für die formel
    N = N < 20 ? 1 : N

    return (c1 * E / N) + (c2 * D / N) + (c3 * avg_weight_diff)
end

"""
    assign_species!(population::Vector{Genome}, species_list::Vector{Vector{Genome}}; speciation_threshold::Union{Float64,Missing}=missing, c1::Union{Float64,Missing}=missing, c2::Union{Float64,Missing}=missing, c3::Union{Float64,Missing}=missing)

Divide a population of genomes into species based on compatibility distance.

For each genome (in random order), compute its distance to the representative of each existing species (the first member). If the smallest distance is ≤ `speciation_threshold`, add it to that species; otherwise, start a new species with this genome as its representative.

If `speciation_threshold`, `c1`, `c2`, or `c3` is `missing`, the corresponding value is loaded from the training parameters in the configuration.

# Arguments
- `population` : Vector of genomes to classify.
- `species_list` : Vector of species (each a vector of genomes) to populate.
- `speciation_threshold` : Maximum compatibility distance to join an existing species.
- `c1`, `c2`, `c3` : Coefficients forwarded to `compatibility_distance`.

# Side Effects
- Clears and reassigns `species_list` in-place.
"""
function assign_species!(
    population::Vector{Genome},
    species_list::Vector{Vector{Genome}};
    speciation_threshold::Float64 = get_config()["train_param"]["speciation_threshold"],
    c1::Float64                  = get_config()["speciation"]["c1"],
    c2::Float64                  = get_config()["speciation"]["c2"],
    c3::Float64                  = get_config()["speciation"]["c3"],
)

    empty!(species_list)
    shuffle!(population)                                # wir shuffeln die population einmal durch da es sich hier um einen greedy algorithmus handelt 

    reps = Genome[]                                     # Das hier sind Representatives. Diese Repräsentieren eine Spezies

    for g in population                                 # wenn die Liste der Representatives leer ist packen wir direkt den ersten als Representative rein, ausserdem in die Spezies-List (Ja wir behandeln das unterschiedlich, brauchen am Ende aber nur die Spezies Liste) 
        if isempty(reps)                                # wenn das der fall ist springen wir direkt in die nächste Iteration
            push!(species_list, [g])
            push!(reps, g)
            continue
        end

        # Compute distances to each representative
        dists = [compatibility_distance(g, rep; c1=c1, c2=c2, c3=c3) for rep in reps]      # wir berechnen für nen neuen Genom die Distanzen zu allen repräsentanten
        idx = argmin(dists)                                                                # suchen uns den repräsentant mit geringstem abstand

        if dists[idx] <= speciation_threshold                                              # und schauen ob der im threshold ist
            push!(species_list[idx], g)                                                    # wenn ja wird er der spezies des Repräsentanten zugewiesen
        else
            push!(species_list, [g])                                                        # ansonsten wird er selbst repräsentant
            push!(reps, g)
        end
    end
end

"""
    adjust_fitness!(species_list::Vector{Vector{Genome}})

Applies fitness sharing to scale each genome’s fitness relative to its species size.

Modifies each genome's fitness value in-place by applying NEAT-style fitness sharing:
- Divides each genome's fitness by the number of members in its species.

# Arguments
- `species_list` : A vector of species (each a vector of genomes).

# Side Effects
- Sets each genome’s `adjusted_fitness` to `fitness / species_size`.
"""
function adjust_fitness!(species_list::Vector{Vector{Genome}})                         # wir teilen nur die fitness jedes genoms durch die anzahl der member in der spezies um zu vermeiden dass manche Spezies zu mächtig werden
    for species in species_list
        s_size = length(species)
        for genome in species
            genome.adjusted_fitness = genome.fitness / s_size
        end
    end
end

"""
    compute_offspring_counts(species_list::Vector{Vector{Genome}}, population_size::Int=300) -> Vector{Int}

Allocate the total number of offspring among species proportionally to their total adjusted fitness.

# Arguments
- `species_list` : List of species (each a vector of genomes).
- `population_size` : Total number of offspring to distribute.

# Returns
- A vector of integers indicating the offspring count for each species (in the same order).
"""
function compute_offspring_counts(species_list::Vector{Vector{Genome}}, population_size::Int=300)        # Fixe größe von 300, nicht so gut
    species_fitness_totals = [sum(g.adjusted_fitness for g in s) for s in species_list]
    total_adjusted = sum(species_fitness_totals)

    if total_adjusted == 0
        return fill(div(population_size, length(species_list)), length(species_list))                    # wenn wir keine fitness haben (Was nicht vorkommen sollte)
    end                                                                                                  # dann kriegt jede spezies gleichviel

    counts = [ round(Int, (fit / total_adjusted) * population_size)                                      # sonst je spezies die aggregierte fitness / die gesamte fitness * die population also 300
               for fit in species_fitness_totals ]

    # Correct rounding errors
    diff = population_size - sum(counts)
    for i in 1:abs(diff)
        idx = mod1(i, length(counts))
        counts[idx] += sign(diff)
    end

    return counts
end

"""
    select_elites(species::Vector{T}, elite_frac::Float64) where {T}

Select the top-performing genomes in a species based on adjusted fitness.

# Arguments
- `species` : Vector of genomes (each with an `adjusted_fitness` field).
- `elite_frac` : Fraction of the species to keep as elites (e.g., 0.1 for 10%).

# Returns
- A vector of the top `ceil(elite_frac * length(species))` genomes, sorted by descending `adjusted_fitness`.
"""
function select_elites(species::Vector{T}, elite_frac::Float64) where {T}
    num_elites = max(1, ceil(Int, elite_frac * length(species)))                    # ma schaut wieviele genome dürfen sich fortpflanzen
    sorted = sort(species, by = g -> g.adjusted_fitness, rev = true)                # sortiert dann die genome innerhalb der spezies 
    return sorted[1:num_elites]                                                    # und gibt die n besten zurück
end

export select_elites

end
