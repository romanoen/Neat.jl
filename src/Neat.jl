# src/Neat.jl
module Neat

include("genome.jl")
include("create_genome.jl")
include("forward_pass.jl")
include("fitness.jl")
include("create_population.jl")
include("mutation.jl")

using .Types
using .CreateGenome
using .ForwardPass
using .Fitness
using .Population
using .Mutation

export Genome,
    Node,
    Connection,
    create_genome,
    forward_pass,
    evaluate_fitness,
    initialize_population,
    mutate

end # module