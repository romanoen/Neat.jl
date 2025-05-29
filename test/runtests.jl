using MyPackage
using Test

@testset "MyPackage.jl" begin
    include("sampleTest.jl") # <--- include tests in runtests.jl
end