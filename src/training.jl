module NeatTrain

using Random
using Plots
using ..NeatConfig: get_config
using ..Types
using ..CreateGenome
using ..ForwardPass: forward_pass
using ..Fitness: evaluate_fitness
using ..Population
using ..Speciation
using ..Mutation: mutate
using ..Crossover: crossover

export train

"""
    train(; pop_size::Union{Int,Missing}=missing,
            n_generations::Union{Int,Missing}=missing,
            input_size::Union{Int,Missing}=missing,
            output_size::Union{Int,Missing}=missing,
            speciation_threshold::Union{Float64,Missing}=missing,
            elite_frac::Union{Float64,Missing}=missing,
            perturb_chance::Union{Float64,Missing}=missing,
            sigma::Union{Float64,Missing}=missing,
            add_connection_prob::Union{Float64,Missing}=missing,
            node_add_prob::Union{Float64,Missing}=missing,
            max_attempts::Union{Int,Missing}=missing,
            c1::Union{Float64,Missing}=missing,
            c2::Union{Float64,Missing}=missing,
            c3::Union{Float64,Missing}=missing,
            ds_name::Union{String,Missing}=missing) -> (Vector{Genome}, Vector{Float64})

Run the full NEAT evolutionary training loop, returning the final population
and the history of best fitness values per generation.

Each keyword argument, if `missing`, is read from the corresponding entry
in the configuration:

- `pop_size`              : Total number of genomes in each generation.
- `n_generations`         : Number of generations to evolve.
- `input_size`            : Number of input neurons in each genome.
- `output_size`           : Number of output neurons.
- `speciation_threshold`  : Compatibility distance threshold for specie assignment.
- `elite_frac`            : Fraction of each species preserved as elites.
- `perturb_chance`        : Probability of small weight perturbation.
- `sigma`                 : Standard deviation of weight perturbation.
- `add_connection_prob`   : Chance to attempt adding a new connection.
- `node_add_prob`         : Chance to attempt adding a new node.
- `max_attempts`          : Max trials when adding a connection.
- `c1`, `c2`, `c3`         : Coefficients for excess, disjoint, and weight‐difference terms in compatibility distance.
- `ds_name`               : Optional name of the training dataset (e.g., `"XOR_DATA"` or `"PARITY3_DATA"`).

# Behavior

1. Initializes a population with `pop_size` genomes.
2. For each generation:
   - Evaluates raw fitness via `evaluate_fitness`.
   - Records the highest fitness in `best_fitness_history`.
   - Performs speciation, fitness sharing, and offspring allocation.
   - Produces a new population by crossover and mutation.
3. After evolving, re-evaluates final fitnesses.
4. Generates and saves a plot of the best‐fitness trajectory to `"fitness.png"`.

# Returns
- `population`           : Vector of `Genome` instances after final generation.
- `best_fitness_history` : Vector of the best fitness value from each generation.
"""
function train(
    ;
    pop_size::Union{Int,Missing}=missing,
    n_generations::Union{Int,Missing}=missing,
    input_size::Union{Int,Missing}=missing,
    output_size::Union{Int,Missing}=missing,
    speciation_threshold::Union{Float64,Missing}=missing,
    elite_frac::Union{Float64,Missing}=missing,
    perturb_chance::Union{Float64,Missing}=missing,
    sigma::Union{Float64,Missing}=missing,
    add_connection_prob::Union{Float64,Missing}=missing,
    node_add_prob::Union{Float64,Missing}=missing,
    max_attempts::Union{Int,Missing}=missing,
    c1::Union{Float64,Missing}=missing,
    c2::Union{Float64,Missing}=missing,
    c3::Union{Float64,Missing}=missing,
    ds_name::Union{String,Missing}=missing
)

    conf = get_config()

    pop_size              = coalesce(pop_size,              conf["train_param"]["pop_size"])
    n_generations         = coalesce(n_generations,         conf["train_param"]["n_generations"])
    input_size            = coalesce(input_size,            conf["train_param"]["input_size"])
    output_size           = coalesce(output_size,           conf["train_param"]["output_size"])
    speciation_threshold  = coalesce(speciation_threshold,  conf["train_param"]["speciation_threshold"])
    elite_frac            = coalesce(elite_frac,            conf["train_param"]["elite_frac"])

    # Initialize the population
    population = initialize_population(pop_size, input_size, output_size)
    best_fitness_history = Float64[]

    for generation in 1:n_generations
        # Evaluate fitness for each genome
        for genome in population
            genome.fitness = evaluate_fitness(genome, ds_name = ds_name)
        end
        # Record the best fitness of this generation


        best_fitness = maximum(g -> g.fitness, population)
        println("Generation $generation: Best fitness = $(best_fitness)")

        push!(best_fitness_history, best_fitness)

        species_list = Vector{Vector{Genome}}()
        assign_species!(population, species_list;
                        speciation_threshold=speciation_threshold,
                        c1=c1, c2=c2, c3=c3)
        adjust_fitness!(species_list)

        # Determine offspring counts for each species
        offspring_counts = compute_offspring_counts(species_list, pop_size)

        # Create new population via crossover and mutation
        new_population = Genome[]
        for (species, count) in zip(species_list, offspring_counts)
            isempty(species) && continue
            # Select elites; if fewer than 2 elites, use entire species
            elites = select_elites(species, elite_frac)
            mating_pool = length(elites) >= 2 ? elites : species

            for _ in 1:count
                parent1, parent2 = rand(mating_pool, 2)
                child = crossover(parent1, parent2)
                mutate(child;
                       perturb_chance=perturb_chance,
                       sigma=sigma,
                       add_connection_prob=add_connection_prob,
                       node_add_prob=node_add_prob,
                       max_attempts=max_attempts)
                push!(new_population, child)
            end
        end
        population = new_population
    end

    # Final fitness evaluation for the evolved population
    for genome in population
        genome.fitness = evaluate_fitness(genome, ds_name = ds_name)
    end

    p = plot(best_fitness_history;
             xlabel="Generation",
             ylabel="Best Fitness",
             title="Evolution of Best Genome Fitness",
             legend=false)
    savefig(p, "fitness.png")

    return population, best_fitness_history
end

end  # module NeatTrain
