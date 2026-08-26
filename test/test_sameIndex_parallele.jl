using Test
using Base.Threads

include("../src/HandIsomorphism.jl") 
using .HandIsomorphism

include("hand-isomorphismWrapper.jl")



# -------------------------------------------------------------------
# Helper: Generate combinations for Round 1 to split across threads
# -------------------------------------------------------------------
function get_first_round_combos(k::Integer)
    combos = Vector{Vector{UInt8}}()
    buf = zeros(UInt8, k)
    function gen(card_in_round, start_card)
        if card_in_round > k
            push!(combos, copy(buf))
            return
        end
        for c in start_card:51
            buf[card_in_round] = UInt8(c)
            gen(card_in_round + 1, c + 1)
        end
    end
    gen(1, 0)
    return combos
end

# -------------------------------------------------------------------
# Parallel Hand Deal Enumerator & Validator
# -------------------------------------------------------------------
function run_parallel_deal_test(native_indexer, c_indexer, rounds::Integer, cards_per_round::Vector{UInt8})
    total_cards = sum(cards_per_round)
    k1 = cards_per_round[1]
    
    # 1. Get all root choices (e.g., 1,326 pairs for hold'em preflop)
    r1_combos = get_first_round_combos(k1)
    
    # Partition root combos across threads
    n_chunks = Threads.nthreads()
    chunks = [r1_combos[i:n_chunks:end] for i in 1:n_chunks]

    # Task results tracking
    results = Vector{NamedTuple{(:count, :mismatches, :first_err), Tuple{Int64, Int64, String}}}(undef, n_chunks)

    # 2. Parallel recursion over subtrees
    @sync for (chunk_id, chunk) in enumerate(chunks)
        Threads.@spawn begin
            # Task-local buffers & states (allocated once per task)
            hand = zeros(UInt8, total_cards)
            c_test_buf = zeros(UInt8, total_cards)
            state = HandIsomorphism.HandIndexerState()
            
            local_count = 0
            local_mismatches = 0
            first_err = ""

            # Recursive DFS for rounds 2..N using a 64-bit bitmask for zero allocations
            function recurse(round_idx, card_in_round, start_card, global_idx, used_mask::UInt64)
                if round_idx > length(cards_per_round)
                    local_count += 1

                    # 1. Julia Index
                    jl_idx = hand_index_last!(native_indexer, hand, state)

                    # 2. C Index
                    copyto!(c_test_buf, hand)
                    c_idx = c_hand_index_last(c_indexer, c_test_buf)

                    # 3. Compare
                    if UInt64(jl_idx) != UInt64(c_idx)
                        local_mismatches += 1
                        if local_mismatches == 1
                            first_err = "Mismatch at Hand: $hand -> Julia: $jl_idx vs C: $c_idx"
                        end
                    end
                    return
                end

                k = cards_per_round[round_idx]
                if card_in_round > k
                    # Move to next round: reset card search to 0
                    recurse(round_idx + 1, 1, 0, global_idx, used_mask)
                    return
                end

                # Deal strictly increasing cards within the current round
                for c in start_card:51
                    bit = UInt64(1) << c
                    if (used_mask & bit) == 0
                        hand[global_idx] = UInt8(c)
                        recurse(round_idx, card_in_round + 1, c + 1, global_idx + 1, used_mask | bit)
                    end
                end
            end

            # Run assigned subtrees
            for combo in chunk
                used_mask = UInt64(0)
                for i in 1:k1
                    c = combo[i]
                    hand[i] = c
                    used_mask |= (UInt64(1) << c)
                end

                if length(cards_per_round) == 1
                    # Only 1 round (Preflop only)
                    local_count += 1
                    jl_idx = hand_index_last!(native_indexer, hand, state)
                    copyto!(c_test_buf, hand)
                    c_idx = c_hand_index_last(c_indexer, c_test_buf)
                    if UInt64(jl_idx) != UInt64(c_idx)
                        local_mismatches += 1
                        if local_mismatches == 1
                            first_err = "Mismatch at Hand: $hand -> Julia: $jl_idx vs C: $c_idx"
                        end
                    end
                else
                    # Recurse from Round 2
                    recurse(2, 1, 0, k1 + 1, used_mask)
                end
            end

            results[chunk_id] = (count=local_count, mismatches=local_mismatches, first_err=first_err)
        end
    end

    # 3. Aggregate results
    total_deals = sum(r.count for r in results)
    total_mismatches = sum(r.mismatches for r in results)
    
    for r in results
        if !isempty(r.first_err)
            println(r.first_err)
            break
        end
    end

    return total_deals, total_mismatches
end

# -------------------------------------------------------------------
# Test Suite
# -------------------------------------------------------------------
@testset "Exhaustive Deal Indexing Equivalence" begin

    test_configs = [
        # (rounds, cards_per_round, name, total_combinations)
        (1, UInt8[2], "Preflop", 1_326),                     
        (2, UInt8[2, 3], "Flop", 25_989_600),             
        (2, UInt8[2, 5], "River 2 round", binomial(52,2) * binomial(50,5)),  # 2809475760
        (2, UInt8[2, 3 ,1 ,1], "River 2 round", binomial(52,2) * binomial(50,3) * 46 * 45),
    ]

    for (rounds, cards_per_round, name, expected_deals) in test_configs
        @testset "$name ($expected_deals deals)" begin
            # Initialize indexers
            native_indexer = HandIndexer()
            @test hand_indexer_init!(rounds, cards_per_round, native_indexer) == true

            c_indexer = Ref{hand_indexer_s}(hand_indexer_s())
            @test c_hand_indexer_init(rounds, cards_per_round, c_indexer) == true

            println("Enumerating all $expected_deals deals for $name in parallel...")
            @time deals_tested, mismatches = run_parallel_deal_test(
                native_indexer, c_indexer, rounds, cards_per_round
            )

            @test deals_tested == expected_deals
            @test mismatches == 0

            c_hand_indexer_free(c_indexer)
        end
    end
end