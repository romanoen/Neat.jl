using Neat
using Test

@testset "Neat.jl" begin
        include("forward_pass_test.jl")
        include("evaluate_fitness_test.jl")
        include("create_genome_test.jl")
        include("types_test.jl")
        include("crossover_test.jl")
        include("speciation_test.jl")
        include("create_population_test.jl")
        include("mutation_test.jl")
        include("visualize_test.jl")
        # include("training_test.jl")
end

