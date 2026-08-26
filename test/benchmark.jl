include("../src/HandIsomorphism.jl")

using .HandIsomorphism
include("./hand-isomorphismWrapper.jl")

using BenchmarkTools
using Random

# Uncomment these if you need to include the files in the same run
# include("HandIsomorphism.jl")
# include("hand-isomorphismWrapper.jl")

function run_benchmarks()
    # Define the configurations requested
    configs = [
        UInt8[2, 3],          # Flop
        UInt8[2, 4],          # Turn (abbreviated)
        UInt8[2, 5],          # River (abbreviated)
        UInt8[2, 3, 1, 1]     # Full Hold'em River (Preflop, Flop, Turn, River)
    ]
    
    N_SAMPLES = 10_000 # Number of random hands to evaluate per configuration
    
    for cards_per_round in configs
        println("==========================================================")
        println("Configuration: ", cards_per_round)
        println("==========================================================")
        
        rounds = length(cards_per_round)
        total_cards = sum(cards_per_round)
        final_round_idx = rounds - 1 # 0-based index for the final round
        
        # ---------------------------------------------------------
        # 1. Initialize Julia Indexer
        # ---------------------------------------------------------
        jl_indexer = HandIsomorphism.HandIndexer(cards_per_round)
        
        # Preallocate zero-allocation states
        jl_state = HandIsomorphism.HandIndexerState()
        jl_scratch = HandIsomorphism.HandUnindexState()
        
        # ---------------------------------------------------------
        # 2. Initialize C Indexer
        # ---------------------------------------------------------
        c_indexer = Ref{hand_indexer_s}()
        c_hand_indexer_init(rounds, cards_per_round, c_indexer)
        
        # ---------------------------------------------------------
        # 3. Pre-sample Data
        # ---------------------------------------------------------
        # Find maximum isomorphic hand index for this game
        max_idx = c_hand_indexer_size(c_indexer, final_round_idx)
        
        Random.seed!(42)
        # Sample N random isomorphic valid indices
        sampled_indices = rand(1:max_idx, N_SAMPLES)
        
        # Produce valid canonical hands by unindexing the random indices
        sampled_hands = Vector{Vector{UInt8}}(undef, N_SAMPLES)
        for i in 1:N_SAMPLES
            hand = zeros(UInt8, total_cards)
            c_hand_unindex(c_indexer, final_round_idx, sampled_indices[i], hand)
            sampled_hands[i] = hand
        end
        
        # Output buffers for unindexing benchmarks
        out_cards_jl = zeros(UInt8, total_cards)
        out_cards_c = zeros(UInt8, total_cards)
        
        # ---------------------------------------------------------
        # 4. Benchmark Indexing (hand_index_last! vs c_hand_index_last)
        # ---------------------------------------------------------
        println("--> Benchmarking Indexing (Julia hand_index_last! vs C c_hand_index_last)")
        
        print("Julia: ")
        @btime begin
            for hand in $sampled_hands
                HandIsomorphism.hand_index_last!($jl_indexer, hand, $jl_state)
            end
        end
        
        print("C:     ")
        @btime begin
            for hand in $sampled_hands
                c_hand_index_last($c_indexer, hand)
            end
        end
        
        # ---------------------------------------------------------
        # 5. Benchmark Unindexing (hand_unindex! vs c_hand_unindex)
        # ---------------------------------------------------------
        println("\n--> Benchmarking Unindexing (Julia hand_unindex! vs C c_hand_unindex)")
        
        print("Julia: ")
        @btime begin
            for idx in $sampled_indices
                HandIsomorphism.hand_unindex!($jl_indexer, $final_round_idx, idx, $out_cards_jl, $jl_scratch)
            end
        end
        
        print("C:     ")
        @btime begin
            for idx in $sampled_indices
                c_hand_unindex($c_indexer, $final_round_idx, idx, $out_cards_c)
            end
        end
        
        # Free C Memory allocated by init
        c_hand_indexer_free(c_indexer)
        println()
    end
end

# Run the benchmark sequence
run_benchmarks()