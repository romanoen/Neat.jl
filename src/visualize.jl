module Visualize

ENV["GKSwstype"] = "1" 
using ..Types: Genome                    
using Plots                      
using Graphs                     
using GraphPlot: gplot, spring_layout    
using Compose                            

import Cairo                      
import Fontconfig            
import Compose: draw, PNG      

export plot_fitness_history, visualize_genome

"""
    plot_fitness_history(best_fitness_history; filename="fitness_history.png")

Create and save a line plot of the best fitness values over generations.

# Arguments
- `best_fitness_history::Vector{Float64}`: History of best fitness per generation.
- `filename::String`: Output file path (defaults to "fitness_history.png").
"""
function plot_fitness_history(best_fitness_history::Vector{Float64}; filename::String = "fitness_history.png")
    # Build a line plot 
    p = plot(
        best_fitness_history,
        xlabel = "Generation",
        ylabel = "Best Fitness",
        title  = "Evolution of Best Genome Fitness",
        legend = false,
        seriestype = :line,
        show = false
    )

    # Save the figure 
    savefig(p, filename)

    return p
end

"""
    visualize_genome(
        genome::Genome;
        filename::String = "genome.png",
        width::Int = 800,
        height::Int = 800
    )

Draws the topology of the given genome and saves it as a PNG file.

# Keyword Arguments
- `filename::String`: Output file path (default: "genome.png").
- `width::Int`, `height::Int`: Dimensions of the output image in pixels.
"""
function visualize_genome(
    genome::Genome;
    filename::String = "best_genome.png",
    width::Int = 800,
    height::Int = 800
)
    node_ids = collect(keys(genome.nodes))
    id2idx = Dict(id => i for (i, id) in enumerate(node_ids))

    g = DiGraph(length(node_ids))

    for ((src, dst), conn) in genome.connections
        if conn.enabled
            add_edge!(g, id2idx[src], id2idx[dst])
        end
    end

    labels = string.(node_ids)
    colors = [
        genome.nodes[id].nodetype == :input  ? "blue"  :   
        genome.nodes[id].nodetype == :output ? "green" :   
                                               "gray"  
        for id in node_ids
    ]


    ctx = gplot(
        g;
        layout         = spring_layout,
        nodelabel      = labels,          
        nodefillc      = colors,          
        nodelabelsize  = 250,             
        nodestrokelw   = 0.5,             
        nodesize       = 0.2              
    )


    draw(PNG(filename, width, height), ctx);
end

end 
