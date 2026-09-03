@testset "test hand_indexer_weight " begin
    cards = Vector{UInt8}(undef, 5)
    scratch = HandUnindexState()
    # 1. Preflop (169 canonical hands) -> must sum to 1,326 = C(52, 2)
    indexer_preflop = HandIndexer([2])
    total_preflop_combos = sum(hand_unindex!(indexer_preflop, 0, idx, cards, scratch) for idx in 1:169)
    println(total_preflop_combos) # Output: 1326
    @test total_preflop_combos == binomial(52, 2)

    # 2. Flop Boards (1,755 canonical flops) -> must sum to 22,100 = C(52, 3)
    indexer_flop = HandIndexer([3])
    total_flop_combos = sum(hand_unindex!(indexer_flop, 0, idx, cards, scratch) for idx in 1:1755)

    @test total_flop_combos == binomial(52, 3)

    # 3. River Boards (1,217,304 canonical rivers) -> must sum to 2,598,960 = C(52, 5)
    indexer_river = HandIndexer([5])


    # total_river_combos = sum(hand_indexer_weight(indexer_river, 0, idx, scratch) for idx in 1:1217304)
    total_river_combos = sum(hand_unindex!(indexer_river, 0, idx, cards, scratch) for idx in 1:1217304)
    println(total_river_combos)   # Output: 2598960

    @test total_river_combos == binomial(52, 5)
end

@testset "Isomorphism Weights Verification" begin

    # ====================================================================
    # TEST 1: Preflop Hands (2 cards -> 169 canonical hands, 1,326 total)
    # ====================================================================
    @testset "Preflop [2] (169 canonical hands)" begin
        indexer = HandIndexer([2])
        num_canonical = hand_indexer_size(indexer, 0)
        @test num_canonical == 169

        # 1. Brute-force counting across all C(52, 2) = 1,326 combinations
        brute_force_weights = zeros(UInt32, num_canonical)
        indexer_state = HandIndexerState()
        cards = zeros(UInt8, 2)

        for c1 in 0:51
            for c2 in (c1+1):51
                cards[1] = UInt8(c1)
                cards[2] = UInt8(c2)
                idx = hand_index_last!(indexer, cards, indexer_state)
                brute_force_weights[idx] += 1
            end
        end

        # 2. Analytic weights from hand_unindex!
        analytic_weights = zeros(UInt32, num_canonical)
        unindex_scratch = HandUnindexState()
        reconstructed = zeros(UInt8, 2)

        for idx in 1:num_canonical
            analytic_weights[idx] = hand_unindex!(indexer, 0, idx, reconstructed, unindex_scratch)
        end

        # 3. Assert exact equality
        @test brute_force_weights == analytic_weights
        println("✓ Preflop [2] passed: All 169 canonical weights match brute-force.")
    end

    # ====================================================================
    # TEST 2: Flop Boards (3 cards -> 1,755 canonical flops, 22,100 total)
    # ====================================================================
    @testset "Flop Boards [3] (1,755 canonical boards)" begin
        indexer = HandIndexer([3])
        num_canonical = hand_indexer_size(indexer, 0)
        @test num_canonical == 1755

        # 1. Brute-force counting across all C(52, 3) = 22,100 combinations
        brute_force_weights = zeros(UInt32, num_canonical)
        indexer_state = HandIndexerState()
        cards = zeros(UInt8, 3)

        for c1 in 0:51
            for c2 in (c1+1):51
                for c3 in (c2+1):51
                    cards[1] = UInt8(c1)
                    cards[2] = UInt8(c2)
                    cards[3] = UInt8(c3)
                    idx = hand_index_last!(indexer, cards, indexer_state)
                    brute_force_weights[idx] += 1
                end
            end
        end

        # 2. Analytic weights from hand_unindex!
        analytic_weights = zeros(UInt32, num_canonical)
        unindex_scratch = HandUnindexState()
        reconstructed = zeros(UInt8, 3)

        for idx in 1:num_canonical
            analytic_weights[idx] = hand_unindex!(indexer, 0, idx, reconstructed, unindex_scratch)
        end

        # 3. Assert exact equality

        @test brute_force_weights == analytic_weights
        println("✓ Flop Boards [3] passed: All 1,755 canonical weights match brute-force.")
    end

    # ====================================================================
    # TEST 3: River Boards (5 cards -> 134,459 canonical boards, 2,598,960 total)
    # ====================================================================
    @testset "River Boards [5] (134,459 canonical boards)" begin
        indexer = HandIndexer([5])
        num_canonical = hand_indexer_size(indexer, 0)
        @test num_canonical == 134459

        # 1. Brute-force counting across all C(52, 5) = 2,598,960 combinations
        println("Running brute-force on 2.6M river board combinations (takes ~0.5s)...")
        brute_force_weights = zeros(UInt32, num_canonical)
        indexer_state = HandIndexerState()
        cards = zeros(UInt8, 5)

        for c1 in 0:51
            for c2 in (c1+1):51
                for c3 in (c2+1):51
                    for c4 in (c3+1):51
                        for c5 in (c4+1):51
                            cards[1] = UInt8(c1)
                            cards[2] = UInt8(c2)
                            cards[3] = UInt8(c3)
                            cards[4] = UInt8(c4)
                            cards[5] = UInt8(c5)
                            idx = hand_index_last!(indexer, cards, indexer_state)
                            brute_force_weights[idx] += 1
                        end
                    end
                end
            end
        end

        # 2. Analytic weights from hand_unindex!
        analytic_weights = zeros(UInt32, num_canonical)
        unindex_scratch = HandUnindexState()
        reconstructed = zeros(UInt8, 5)

        for idx in 1:num_canonical
            analytic_weights[idx] = hand_unindex!(indexer, 0, idx, reconstructed, unindex_scratch)
        end

        # 3. Assert exact equality
        @test brute_force_weights == analytic_weights
        println("✓ River Boards [5] passed: All 1,217,304 canonical weights match brute-force.")
    end

end