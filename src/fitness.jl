module Fitness

using ..Types
using ..ForwardPass
using ..NeatConfig

export evaluate_fitness

# ----- trainingdata, can be outsourced in future -------

const XOR_DATA = [
    ([0.0, 0.0], 0.0), 
    ([0.0, 1.0], 1.0), 
    ([1.0, 0.0], 1.0), 
    ([1.0, 1.0], 0.0)]

const PARITY3_DATA = [
    ([0.0, 0.0, 0.0], 0.0),
    ([0.0, 0.0, 1.0], 1.0),
    ([0.0, 1.0, 0.0], 1.0),
    ([0.0, 1.0, 1.0], 0.0),
    ([1.0, 0.0, 0.0], 1.0),
    ([1.0, 0.0, 1.0], 0.0),
    ([1.0, 1.0, 0.0], 0.0),
    ([1.0, 1.0, 1.0], 1.0),
]

"""
    evaluate_fitness(genome::Genome) → Float64

Evaluate a `Genome`’s performance on the XOR task by summing squared errors 
for all four input–output pairs and returning the negative total error.

# Arguments
- `genome::Genome`: The genome whose neural network weights are tested on XOR.

# Returns
- `Float64`: The negative sum of squared errors over the four XOR cases.
  (Higher value ⇒ lower error ⇒ better fitness.)
"""
function evaluate_fitness(genome::Genome)::Float64
    conf = get_config()
    total_error = 0.0
    m = conf["data"]
    ds_name = m["training_data"]

    ds = getfield(Fitness, Symbol(ds_name))

    for (x, target) in ds
        acts = forward_pass(genome, x)

        output_nodes = [n.id for n in values(genome.nodes) if n.nodetype == :output]

        output_value = acts[output_nodes[1]] 
        total_error += (output_value - target)^2  
    end

    return -total_error 
end

end
