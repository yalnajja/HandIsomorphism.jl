using HandIsomorphism
using Test

using StaticArrays
include("hand-isomorphismWrapper.jl")


@testset "HandIsomorphism.jl" begin
    include("test_sameIndex.jl")
    include("test_unindex_equivalence.jl")
end
