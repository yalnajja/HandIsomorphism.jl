using HandIsomorphism_jll

const C_MAX_ROUNDS = 8 
const C_SUITS = 4

struct hand_indexer_s
    cards_per_round::NTuple{C_MAX_ROUNDS, UInt8}
    round_start::NTuple{C_MAX_ROUNDS, UInt8}

    rounds::UInt64
    configurations::NTuple{C_MAX_ROUNDS, UInt64}
    permutations::NTuple{C_MAX_ROUNDS, UInt64}
    round_size::NTuple{C_MAX_ROUNDS, UInt64}

    permutation_to_configuration::NTuple{C_MAX_ROUNDS, Ptr{Cvoid}}
    permutation_to_pi::NTuple{C_MAX_ROUNDS, Ptr{Cvoid}}
    configuration_to_equal::NTuple{C_MAX_ROUNDS, Ptr{Cvoid}}

    configuration::NTuple{C_MAX_ROUNDS, Ptr{Cvoid}}
    configuration_to_suit_size::NTuple{C_MAX_ROUNDS, Ptr{Cvoid}}
    configuration_to_offset::NTuple{C_MAX_ROUNDS, Ptr{Cvoid}}

    hand_indexer_s() = new()
end

function c_hand_indexer_init(rounds::Integer, cards_per_round::Vector{UInt8}, indexer::Ref{hand_indexer_s})
    ret = ccall((:hand_indexer_init, libhandisomorphism),
        Bool, (UInt64, Ptr{UInt8}, Ptr{hand_indexer_s}),
        UInt64(rounds), cards_per_round, indexer)
    return ret
end

function c_hand_indexer_size(indexer::Ref{hand_indexer_s}, round::Integer)
    ret = ccall((:hand_indexer_size, libhandisomorphism),
        UInt64, (Ptr{hand_indexer_s}, UInt64),
        indexer, UInt64(round))
    return ret
end

function c_hand_index_last(indexer::Ref{hand_indexer_s}, cards::Vector{UInt8})
    ret = ccall((:hand_index_last, libhandisomorphism),
        UInt64, (Ptr{hand_indexer_s}, Ptr{UInt8}),
        indexer, cards)
    return ret + 1 # Returns 1-based index
end

# 1-based indexes
function c_hand_unindex(indexer::Ref{hand_indexer_s}, round::Integer, index::Integer, cards::Vector{UInt8})
    success = ccall((:hand_unindex, libhandisomorphism),
        Bool, (Ptr{hand_indexer_s}, UInt64, UInt64, Ptr{UInt8}),
        indexer, UInt64(round), UInt64(index - 1), cards)
    return success 
end 

function c_hand_indexer_free(indexer::Ref{hand_indexer_s})
    ccall((:hand_indexer_free, libhandisomorphism), Cvoid, (Ptr{hand_indexer_s},), indexer)
end