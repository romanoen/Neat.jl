module CreateGenome

using ..Types

export create_genome

# function to instantiate a network
# number of inputs & outputs is choosable, connections are hardcoded though

function create_genome(id::Int, num_inputs::Int, num_outputs::Int)
    nodes = Dict{Int,Node}()
    connections = Dict{Tuple{Int,Int},Connection}()

    # create input nodes
    for i in 1:num_inputs
        nodes[i] = Node(i, :input)
    end

    # create output nodes
    for i in 1:num_outputs
        nid = num_inputs + i
        nodes[nid] = Node(nid, :output)
    end

    # Connection from input 1 to output 1 and input 2 to output 1
    connections[(1, num_inputs + 1)] = Connection(1, num_inputs + 1, randn(), true, 1)
    connections[(2, num_inputs + 1)] = Connection(2, num_inputs + 1, randn(), true, 2)

    return Genome(id, nodes, connections)
end

end