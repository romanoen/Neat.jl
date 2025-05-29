module Fitness

using ..Types
using ..ForwardPass

export evaluate_fitness

# Fitness Function that returns an error for XOR

function evaluate_fitness(genome::Genome)::Float64
    xor_data = [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)]

    total_error = 0.0

    for (x, target) in xor_data
        output = forward_pass(genome, x) # compute forward_pass
        total_error += (output - target)^2  # compute error
    end

    return -total_error # the lower the error, the higher the fitness
end

end
