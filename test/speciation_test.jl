using Test
using Neat
using Neat: Genome, select_elites, select_parents

struct MockGenome
    adjusted_fitness::Float64
end

@testset "Speciation" begin

    @testset "compatibility_distance" begin
        g1 = create_genome(1, 2, 1)
        g2 = create_genome(2, 2, 1)

        d = compatibility_distance(g1, g2)
        @test isa(d, Float64)
        @test d ≥ 0
    end

    @testset "assign_species!" begin
        pop = [create_genome(i, 2, 1) for i in 1:4]
        for g in pop
            g.fitness = evaluate_fitness(g)
        end

        species_list = Vector{Vector{Genome}}()
        assign_species!(pop, species_list; threshold=3.0)

        @test sum(length.(species_list)) == length(pop)
        @test all(length(s) > 0 for s in species_list)
    end

    @testset "adjust_fitness!" begin
        pop = [create_genome(i, 2, 1) for i in 1:3]
        for g in pop
            g.fitness = -1.0
        end

        species_list = [[pop[1], pop[2]], [pop[3]]]
        adjust_fitness!(species_list)

        @test pop[1].adjusted_fitness ≈ -0.5
        @test pop[2].adjusted_fitness ≈ -0.5
        @test pop[3].adjusted_fitness ≈ -1.0

    end

    @testset "compute_offspring_counts" begin
        g1 = create_genome(1, 2, 1)
        g2 = create_genome(2, 2, 1)
        g3 = create_genome(3, 2, 1)
        g1.fitness = 2.0
        g2.fitness = 2.0
        g3.fitness = 6.0

        species_list = [[g1, g2], [g3]]
        adjust_fitness!(species_list)

        counts = compute_offspring_counts(species_list, 10)
        @test length(counts) == 2
        @test sum(counts) == 10
        @test counts[2] > counts[1]  # more fit species gets more offspring
    end

    @testset "Selection Tests" begin
    # Create mock genomes with varying adjusted fitness
    genomes = [MockGenome(i) for i in 1:10]


    @testset "select_elites tests" begin
    # Test with unique fitness values
    species = [MockGenome(1.0), MockGenome(3.0), MockGenome(2.0), MockGenome(4.0)]
    elites = select_elites(species, 0.5)
    @test length(elites) == max(1, ceil(Int, 0.5 * length(species))) == 2
    @test [g.adjusted_fitness for g in elites] == [4.0, 3.0]

    # Test that at least one elite is returned when fraction is very small
    small_species = [MockGenome(10.0), MockGenome(20.0)]
    elites_small = select_elites(small_species, 0.1)
    @test length(elites_small) == 1
    @test elites_small[1].adjusted_fitness == 20.0

    # Test full selection (elite_frac = 1.0)
    full_species = [MockGenome(5.0), MockGenome(15.0), MockGenome(10.0)]
    elites_full = select_elites(full_species, 1.0)
    @test length(elites_full) == length(full_species) == 3
    @test [g.adjusted_fitness for g in elites_full] == [15.0, 10.0, 5.0]

    # Test non-integer result rounding up
    varied = [MockGenome(i) for i in 1.0:5.0]
    # 40% of 5 = 2, so elite count should be 2
    elites_varied = select_elites(varied, 0.4)
    @test length(elites_varied) == ceil(Int, 0.4 * length(varied)) == 2

    # Test that input order doesn't affect output beyond fitness
    shuffled = shuffle(varied)
    elites_shuffled = select_elites(shuffled, 0.4)
    @test [g.adjusted_fitness for g in elites_shuffled] == [5.0, 4.0]
end
end
end