"""
Methods:
---------
evaluate_fitness(genome::Genome) -> Float64
    Evaluate how well a genome performs a specific task (e.g., XOR). Can be customized per problem.

select_elites(species::Vector{Vector{Genome}}; num_elites::Int=1) -> Vector{Genome}
    Selects the top N genomes from each species based on fitness to survive unchanged.

select_parents(species::Vector{Vector{Genome}}) -> (Genome, Genome)
    Selects two parents using fitness-proportional selection within species.

run_evolution(; generations::Int=100, pop_size::Int=150, input_size::Int=2, output_size::Int=1) -> Genome
    The main training loop. Runs for the specified number of generations and returns the best genome found.
"""

using Neat  # Assumes Neat.jl is your main module
using Random
using Statistics
using Distributions

# --- User-defined fitness function (example: XOR) ---
function evaluate_fitness(genome)
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    expected = [0.0, 1.0, 1.0, 0.0]
    error = 0.0
    for (x, y) in zip(inputs, expected)
        output = forward_pass(genome, x)[1]
        error += (output - y)^2
    end
    return 4.0 - error  # Higher fitness is better
end

# --- Elitism: Keep the top genomes from each species ---
function select_elites(species::Vector{Vector{Genome}}; num_elites::Int=1)
    elites = Genome[]
    for group in species
        sorted_group = sort(group, by = g -> -g.fitness)  # Sort descending by fitness
        append!(elites, sorted_group[1:min(num_elites, end)])  # Take top n or all if fewer
    end
    return elites
end

# --- Parent selection: Probabilistic based on fitness and species ---
function select_parents(species::Vector{Vector{Genome}})
    # Pick a species weighted by average fitness
    avg_fitnesses = [mean(g.fitness for g in group) for group in species]
    total = sum(avg_fitnesses)
    probs = total == 0 ? fill(1.0 / length(species), length(species)) : avg_fitnesses ./ total
    chosen_species = species[Random.rand(Categorical(probs))]

    # Choose two parents weighted by individual fitness
    fitnesses = [g.fitness for g in chosen_species]
    total_fit = sum(fitnesses)
    parent_probs = total_fit == 0 ? fill(1.0 / length(chosen_species), length(chosen_species)) : fitnesses ./ total_fit

    parent1 = chosen_species[Random.rand(Categorical(parent_probs))]
    parent2 = chosen_species[Random.rand(Categorical(parent_probs))]

    return parent1, parent2
end

# --- Training loop ---
function run_evolution(; generations=100, pop_size=150, input_size=2, output_size=1)
    println("Initializing population...")
    population = create_population(pop_size, input_size, output_size)

    for gen in 1:generations
        println("Generation $gen")

        # Evaluate fitness
        for genome in population
            genome.fitness = evaluate_fitness(genome)
        end

        # Speciation
        species = assign_species(population)

        # Elitism - keep top genome(s)
        elites = select_elites(species, num_elites=1)

        # Reproduce
        offspring = Genome[]
        while length(offspring) < pop_size - length(elites)
            parent1, parent2 = select_parents(species)
            child = crossover(parent1, parent2)
            mutate!(child)
            push!(offspring, child)
        end

        # Replace population
        population = vcat(elites, offspring)
    end

    # Return the best genome found
    best_idx = findmax([g.fitness for g in population])[2]
    return population[best_idx]
end

# --- Run the evolution ---
best_genome = run_evolution(generations=100)
println("Best genome fitness: ", best_genome.fitness)
