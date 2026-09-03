using HandIsomorphism
using Test

using StaticArrays
include("hand-isomorphismWrapper.jl")


@testset "HandIsomorphism.jl" begin
    include("test_sameIndex.jl")
    include("test_unindex_equivalence.jl")
    include("test_hand_indexer_weight.jl")
end
