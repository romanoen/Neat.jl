module Fitness

using ..Types
using ..ForwardPass
using ..NeatConfig

export evaluate_fitness

# ----- training data; can be externalized in future -------

const XOR_DATA = [
    ([0.0, 0.0], 0.0), 
    ([0.0, 1.0], 1.0), 
    ([1.0, 0.0], 1.0), 
    ([1.0, 1.0], 0.0)
]

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
    evaluate_fitness(genome::Genome; ds_name::Union{String,Missing}=missing) -> Float64

Compute the fitness of a `genome` on a specified dataset by running its neural network
and summing the squared errors between outputs and targets, then returning the negative total error.

A more positive return value indicates better performance (lower overall error).

# Keyword Arguments
- `genome` : The `Genome` whose network will be evaluated.
- `ds_name` : Optional name of the dataset to use (e.g., `"XOR_DATA"` or `"PARITY3_DATA"`).
              If not defined, the default `training_data` key from the configuration is used.

# Returns
- `Float64` : Negative sum of squared errors over all input–target pairs in the dataset.
"""
function evaluate_fitness(genome::Genome; ds_name::String=get_config()["data"]["training_data"])
    total_error = 0.0

    ds = getfield(Fitness, Symbol(ds_name))

    for (x, target) in ds
        acts = forward_pass(genome, x)                                                        # wir berechnen die berechneten ergebnisse nach den knoten mit dem forward pass, rückgabe ist ein dict mit den node IDs sowie deren ergebnis
        output_nodes = [n.id for n in values(genome.nodes) if n.nodetype == :output]            # wir holen uns die ids der output knoten
        output_value = acts[output_nodes[1]]                                                    # schauen uns an wie hoch der output wert ist des output knotens (ACHTUNG: Mit dieser Implementierung können wir nur einen einzigen Output Knoten nutzen) 
        total_error += (output_value - target)^2                                                # error berechnet sich durch predicted und target
    end

    return -total_error
end

end
