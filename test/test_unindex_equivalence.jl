# using Test
# using StaticArrays
# include("../src/HandIsomorphism.jl") 
# using .HandIsomorphism

# include("hand-isomorphismWrapper.jl")

# -------------------------------------------------------------------
# Direct Hand Isomorphism Check (No Re-Indexing)
# -------------------------------------------------------------------
"""
    canonical_suit_profiles(cards, rounds, cards_per_round)

Computes the 4-element canonical (sorted) suit profile of a hand.
Each suit is represented as a 64-bit integer packing its rank bitmask
for each round (13 bits per round).
"""
function canonical_suit_profiles(cards::AbstractVector{UInt8}, rounds::Integer, cards_per_round::Vector{UInt8})
    # profiles for suits 0, 1, 2, 3
    profiles = @MVector zeros(UInt64, 4)
    
    card_idx = 1
    for r in 1:rounds
        cnt = cards_per_round[r]
        shift = (r - 1) * 13 # 13 ranks per round
        
        for _ in 1:cnt
            c = cards[card_idx]  # 0-based: 0..51
            suit = (c & 3) + 1             # 1..4
            rank = c >> 2                  # 0..12
            
            profiles[suit] |= (UInt64(1) << rank) << shift
            card_idx += 1
        end
    end
    
    # Sort profiles so suit ordering doesn't matter (suit invariance)
    return sort(profiles)
end

"""
    are_hands_isomorphic(cards1, cards2, rounds, cards_per_round)

Directly checks if cards1 and cards2 represent the exact same hand equivalence class.
"""
function are_hands_isomorphic(cards1::AbstractVector{UInt8}, cards2::AbstractVector{UInt8}, 
                              rounds::Integer, cards_per_round::Vector{UInt8})
    p1 = canonical_suit_profiles(cards1, rounds, cards_per_round)
    p2 = canonical_suit_profiles(cards2, rounds, cards_per_round)
    return p1 == p2
end


# -------------------------------------------------------------------
# Test Suite
# -------------------------------------------------------------------
@testset "Direct Hand Isomorphism Equivalence Tests" begin

    test_configs = [
        # (1, UInt8[2],          "Preflop"),
        (2, UInt8[2, 3],       "Flop"), 
        # (2, UInt8[2, 4],    "Turn 2 round"),
        # (2, UInt8[2, 5], "River 2 round"),
        # (4, UInt8[2, 3,1,1], "River")

    ]

    for (rounds, cards_per_round, name) in test_configs
        @testset "$name Testing" begin
            native_indexer = HandIndexer()
            
            @test hand_indexer_init!(rounds, cards_per_round, native_indexer) == true

            c_indexer = Ref{hand_indexer_s}(hand_indexer_s())
            @test c_hand_indexer_init(rounds, cards_per_round, c_indexer) == true

            total_cards = sum(cards_per_round)
            c_cards_buf = zeros(UInt8, total_cards)
            jl_cards_buf = zeros(UInt8, total_cards)
            unindex_scratch = HandIsomorphism.HandUnindexState()

            final_round = rounds - 1
            max_size = hand_indexer_size(native_indexer, final_round)

            # Sample test indices across the round
            step_size = max_size > 50_000 ? max_size ÷ 200 : 1
            step_size = 1

            test_indices = unique(vcat(
                UInt64[1, 2, 3, max_size ÷ 2, max_size - 1, max_size],
                collect(1:step_size:max_size)
            ))

            for idx in test_indices
                fill!(c_cards_buf, 0)
                fill!(jl_cards_buf, 0)

                # Unindex using both libraries
                @test c_hand_unindex(c_indexer, final_round, idx, c_cards_buf) == true
                @test hand_unindex!(native_indexer, final_round, idx, jl_cards_buf, unindex_scratch) == true

                # Direct Isomorphism Test:
                # Verifies that both card sets have the identical suit-rank profile
                @test are_hands_isomorphic(jl_cards_buf, c_cards_buf, rounds, cards_per_round)
            end

            c_hand_indexer_free(c_indexer)
        end
    end
end