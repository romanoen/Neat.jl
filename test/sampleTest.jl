using MyPackage
using Test
using Flux

@testset "timestwo" begin
    @test timestwo(4.0) == 8.0
end