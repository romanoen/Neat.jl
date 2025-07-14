using Test
using Neat.Visualize: plot_fitness_history, visualize_genome
using Neat.Types: Genome, Node, Connection

@testset "Visualize.plot_fitness_history tests" begin
    # Test with empty history: should produce a plot and save an empty-line graph
    empty_hist = Float64[]
    empty_file = "test_empty_fitness.png"
    p_empty = plot_fitness_history(empty_hist; filename=empty_file)
    @test isfile(empty_file)
    rm(empty_file; force=true)

    # Test with a sample history
    sample_hist = [0.2, 0.4, 0.6, 0.8, 1.0]
    sample_file = "test_sample_fitness.png"
    p_sample = plot_fitness_history(sample_hist; filename=sample_file)
    @test isfile(sample_file)
    # Verify the plotted data matches the input history
    series = p_sample.series_list[1]
    @test length(series[:x]) == length(sample_hist)
    @test length(series[:y]) == length(sample_hist)
    @test series[:y] == sample_hist
    rm(sample_file; force=true)
end

@testset "Visualize.visualize_genome tests" begin
    # Setup a simple genome with one enabled and one disabled connection
    nodes = Dict(
        1 => Node(1, :input),
        2 => Node(2, :output)
    )
    connections = Dict(
        (1, 2) => Connection(1, 2, 0.5, true),   # enabled connection
        (2, 1) => Connection(2, 1, -0.5, false)  # disabled connection
    )
    genome = Genome(nodes, connections)
    gen_file = "test_genome.png"
    # Generate visualization
    visualize_genome(genome; filename=gen_file, width=100, height=100)
    @test isfile(gen_file)
    rm(gen_file; force=true)
end
