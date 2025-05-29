module MyPackage

include("Types.jl")
include("CreateGenome.jl")
include("ForwardPass.jl")
include("Fitness.jl")

using .Types
using .CreateGenome
using .ForwardPass
using .Fitness

export Genome, Node, Connection, forward_pass, evaluate_fitness, create_genome


# feed network with xor_data -> get random output
R = create_genome(1, 2, 1)
fitness = evaluate_fitness(R)
println("Fitness of R: $fitness")

println("\nDetails:")
for ((in, out), conn) in R.connections
    println("Connection $in → $out, weight=$(round(conn.weight, digits=3))")
end
end