# using Test
# include("../src/HandIsomorphism.jl") 
# using .HandIsomorphism

# include("hand-isomorphismWrapper.jl")


# -------------------------------------------------------------------
# Hand Deal Enumerator (Zero Allocations during iteration)
# -------------------------------------------------------------------
"""
    for_each_deal(f, cards_per_round::Vector{UInt8})

Enumerates all valid combinations of cards across rounds without replacement.
Cards within each round are strictly sorted (`c_1 < c_2 < ... < c_k`).
"""

function for_each_deal(f, cards_per_round::Vector{UInt8})
    total_cards = sum(cards_per_round)
    hand = zeros(UInt8, total_cards)
    used = falses(52)

    function recurse(round_idx, card_in_round, start_card, global_idx)
        if round_idx > length(cards_per_round)
            f(hand)
            return
        end

        k = cards_per_round[round_idx]
        if card_in_round > k
            # Advance to next round: reset search start to card 0
            recurse(round_idx + 1, 1, 0, global_idx)
            return
        end

        # Pick strictly increasing cards (0 to 51) within the current round
        for c in start_card:51
            if !used[c + 1]  # Julia arrays/bitvectors are 1-indexed
                used[c + 1] = true
                hand[global_idx] = UInt8(c)
                recurse(round_idx, card_in_round + 1, c + 1, global_idx + 1)
                used[c + 1] = false
            end
        end
    end

    recurse(1, 1, 0, 1)
end
# Fallback alias for non-do-block calls
for_each_deal(cards_per_round::Vector{UInt8}, f) = for_each_deal(f, cards_per_round)

# -------------------------------------------------------------------
# Exhaustive Enumeration Tests
# -------------------------------------------------------------------
@testset "Exhaustive Deal Indexing Equivalence" begin

    test_configs = [
        # (rounds, cards_per_round, name, total_combinations)
        (1, UInt8[2], "Preflop", 1_326),                     
        (2, UInt8[2, 3], "Flop", 25_989_600),             
        # (2, UInt8[2, 5], "River 2 round", 133_784_560),
    ]

    for (rounds, cards_per_round, name, expected_deals) in test_configs
        @testset "$name ($expected_deals deals)" begin
            # 1. Initialize Native Julia Indexer
            native_indexer = HandIndexer(cards_per_round)

            # 2. Initialize C Indexer
            c_indexer = Ref{hand_indexer_s}(hand_indexer_s())
            @test c_hand_indexer_init(rounds, cards_per_round, c_indexer) == true

            state = HandIsomorphism.HandIndexerState()
            
            # Reusable buffer to prevent C wrapper mutating the hand
            c_test_buf = zeros(UInt8, sum(cards_per_round))

            mismatches = 0
            count = 0
            first_mismatch_info = ""

            println("Enumerating all $expected_deals deals for $name...")
            @time for_each_deal(cards_per_round) do hand
                count += 1
                
                # 1. Compute Native Julia Index
                jl_idx = hand_index_last!(native_indexer, hand, state)

                # 2. Compute C Index
                copyto!(c_test_buf, hand)
                c_idx = c_hand_index_last(c_indexer, c_test_buf)

                # 3. Check for mismatch
                if UInt64(jl_idx) != UInt64(c_idx)
                    mismatches += 1
                    if mismatches == 1
                        first_mismatch_info = "First mismatch at deal #$count, Hand: $hand -> Julia: $jl_idx vs C: $c_idx"
                    end
                end
            end

            if mismatches > 0
                println(first_mismatch_info)
            end

            # @test count == expected_deals
            @test mismatches == 0
        end
    end
end