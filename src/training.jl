module NeatTrain

using Random
using Plots
using ..NeatConfig 
using ..Types
using ..CreateGenome
using ..ForwardPass
using ..Fitness
using ..Population
using ..Innovation
using ..Mutation
using ..Crossover
using ..Speciation
using ..NeatTrain


export train
"""
    train()

Run a full NEAT training loop

# Keyword Arguments
no keywords, parameters are in neat_config.toml that is automatically generated during project intialisation

# Returns
- The final evolved population
"""
function train()

    # setup config
    conf = get_config()
    m = conf["train_param"]
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
    end

    for genome in population
        genome.fitness = evaluate_fitness(genome)
    end

    return population, best_fitness_history
end
end 