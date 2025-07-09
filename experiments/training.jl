using Pkg
Pkg.activate("../")
Pkg.add("Plots")

using Neat
using Random
using Statistics
using Plots

"""
    train(; pop_size=150, n_generations=100, input_size=2, output_size=1, speciation_threshold=0.3)

Run a full NEAT training loop for the specified number of generations.

# Keyword Arguments
- `pop_size`: Number of genomes in the population
- `n_generations`: Number of generations to evolve
- `input_size`: Number of input nodes per genome
- `output_size`: Number of output nodes per genome
- `speciation_threshold`: Compatibility threshold for species assignment

# Returns
- The final evolved population
"""
function train()

    # setup config
    m = CONFIG["train_param"]
    pop_size = m["pop_size"]
    n_generations = m["n_generations"]
    input_size = m["input_size"]
    output_size = m["output_size"]
    speciation_threshold = m["speciation_threshold"]
    elite_frac = m["elite_frac"]



    population = initialize_population(pop_size, input_size, output_size)
    best_fitness_history = Float64[]

    for generation in 1:n_generations
        #println("\n=== Generation $generation ===")

        for genome in population
            genome.fitness = evaluate_fitness(genome)
        end

        push!(best_fitness_history,maximum(g -> g.fitness, population))

        species_list = Vector{Vector{Genome}}()
        assign_species!(population, species_list; threshold=speciation_threshold)

        adjust_fitness!(species_list)

        offspring_counts = compute_offspring_counts(species_list, pop_size)

        new_population = Genome[]
        for (species, count) in zip(species_list, offspring_counts)
            if isempty(species)
                continue
            end

            elites = select_elites(species, elite_frac)
            mating_pool = length(elites) >= 2 ? elites : species

            for _ in 1:count
                parent1, parent2 = rand(mating_pool, 2)
                child = crossover(parent1, parent2)
                mutate(child)
                push!(new_population, child)
            end
        end

        population = new_population
    end # for each generation

    for genome in population
        genome.fitness = evaluate_fitness(genome)
    end

    return population, best_fitness_history
end

final_pop, best_fitness_history = train()

idx = argmax(g -> g.fitness, final_pop)
best = idx
println("Best fitness: ", best.fitness)


p = plot(
    best_fitness_history,
    xlabel="Generation",
    ylabel="Best Fitness",
    title="Evolution of Best Genome Fitness",
    legend=false,
)

savefig(p, "plots/fitness.png")