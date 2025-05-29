module ForwardPass

using ..Types
using ..CreateGenome

export forward_pass

function forward_pass(genome::Genome, input::Vector{Float64})::Float64

    # this function calculates the output of an associated phenotype from genotype genom

    output_sum = 0.0
    for i in 1:length(input)
        conn_key = (i, 3) # assuming output node is always 3 (see first cell; hardcoded)
        if haskey(genome.connections, conn_key) # if there is edge from i to 3 (op) 
            conn = genome.connections[conn_key]
            if conn.enabled # and if connection is enabled
                output_sum += input[i] * conn.weight # add input*weight to output
            end
        end
    end
    return 1.0 / (1.0 + exp(-output_sum)) # apply sigmoid (because why not)
end

end