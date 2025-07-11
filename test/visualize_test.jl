using Test
using Neat.Visualize: plot_fitness_history

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
