module HandIsomorphism

export HandIndexer, HandIndexerState, HandUnindexState, hand_indexer_init, hand_indexer_size,
    hand_indexer_state_init!, hand_index_all!, hand_index_last!,
    hand_index_next_round!, hand_unindex!, deck_get_rank, deck_get_suit, deck_make_card


using StaticArrays

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
const SUITS = 4
const RANKS = 13
const CARDS = 52
const MAX_ROUNDS = 8
const MAX_GROUP_INDEX = 0x100000
const ROUND_SHIFT = 4
const ROUND_MASK = 0x0f

# -------------------------------------------------------------------
# Offset/equal packing
#
# configuration_to_offset_equal packs, per configuration:
#   bits [63:3] -> cumulative offset into the round's index space
#   bits [2:0]  -> equal_index (always 0-7, since it's `equal >> 1`
#                  over a (SUITS-1)-bit mask, i.e. at most 3 bits)
#
# This replaces two separate arrays (configuration_to_offset::UInt64,
# configuration_to_equal::UInt32) with one UInt64 array: one fewer
# cache line touched per binary-search step, and ~1/3 less memory for
# the largest (Turn/River) tables. Offset headroom: 61 bits comfortably
# covers any realistic round_size (River is ~2.4e9 for NLHE; even large
# custom games are nowhere close to 2^61).
# -------------------------------------------------------------------
@inline pack_offset_equal(offset::UInt64, equal::UInt32) = (offset << 3) | UInt64(equal)
@inline unpack_offset(v::UInt64) = v >> 3
@inline unpack_equal(v::UInt64) = UInt32(v & 0x7)

# -------------------------------------------------------------------
# equal_index bit test
#
# Previously a (1<<(SUITS-1)) x SUITS Bool matrix (`equal_mask`)
# populated at __init__ time and read via `equal_mask[equal_index+1, i+1]`
# in the hottest loops of hand_index_next_round! and hand_unindex!.
# The mask was populated by exactly `(i & (1 << (j - 1))) != 0`, i.e.
# equal_index's own bits -- there was never any information in the
# table that wasn't already in equal_index. Testing the bit directly
# removes a 2D array load (and its bounds check) from every suit-group
# check, and lets the const table + its __init__ population loop be
# deleted entirely.
#
# Valid for i in 1:(SUITS-1), matching how it was ever indexed before
# (the guard `i+1 <= SUITS` etc. at each call site already keeps i in
# that range).
# -------------------------------------------------------------------
@inline is_equal(equal_index::UInt32, i::Integer) = ((equal_index >> (i - 1)) & 0x1) != 0

# -------------------------------------------------------------------
# Deck Utilities
# -------------------------------------------------------------------
@inline deck_get_suit(card::Integer) = card & 3
@inline deck_get_rank(card::Integer) = card >> 2
@inline deck_make_card(suit::Integer, rank::Integer) = (rank << 2) | suit

# -------------------------------------------------------------------
# Data Structures
# -------------------------------------------------------------------
mutable struct HandIndexer
    rounds::UInt32
    cards_per_round::Vector{UInt8}
    round_start::Vector{UInt8}
    configurations::Vector{UInt32}
    permutations::Vector{UInt32}
    round_size::Vector{UInt64}

    permutation_to_configuration::Vector{Vector{UInt32}}
    permutation_to_pi::Vector{Vector{UInt32}}
    configuration::Vector{Matrix{UInt32}}
    configuration_to_suit_size::Vector{Matrix{UInt32}}
    configuration_to_offset_equal::Vector{Vector{UInt64}}   # packed: offset<<3 | equal

    function HandIndexer()
        new(0, UInt8[], UInt8[], zeros(UInt32, MAX_ROUNDS), zeros(UInt32, MAX_ROUNDS),
            zeros(UInt64, MAX_ROUNDS), Vector{Vector{UInt32}}(), Vector{Vector{UInt32}}(),
            Vector{Matrix{UInt32}}(), Vector{Matrix{UInt32}}(), Vector{Vector{UInt64}}())
    end
end

# -------------------------------------------------------------------
# HandIndexerState
#
# The small per-suit/per-round buffers below (suit_index, ranks,
# suit_idx, indices, ...) are all fixed at exactly SUITS=4 or
# MAX_ROUNDS=4 elements for the lifetime of the struct -- they never
# resize. Storing them as plain `Vector{T}` means every access carries
# a runtime-length field the compiler can't reason about, which blocks
# loop unrolling/SIMD over `for i in 1:SUITS`. Using `MVector{4,T}`
# instead gives the compiler a compile-time-constant size to unroll
# and bounds-check against.
#
# Note this is *not* a heap-allocation elimination: HandIndexerState
# is a `mutable struct`, and MVector is itself a mutable, non-isbits
# type, so each of these fields is still a separate heap object
# reached through a pointer -- the same as before. What changes is the
# object's internal layout (its data lives inline in the MVector's own
# allocation rather than in an out-of-line Array buffer, so it's one
# allocation instead of two) and, more importantly, that the compiler
# now knows the length statically. If true inline/stack storage is
# wanted, that requires immutable `SVector`/`NTuple` fields with
# functional updates (`state.ranks = setindex(state.ranks, v, i)`),
# which is a larger structural change than this drop-in swap.
# -------------------------------------------------------------------
mutable struct HandIndexerState
    suit_index::MVector{SUITS,UInt64}
    suit_multiplier::MVector{SUITS,UInt64}
    round::UInt32
    permutation_index::UInt32
    permutation_multiplier::UInt32
    used_ranks::MVector{SUITS,UInt32}

    # Preallocated buffers for zero-allocation hand_index_next_round!
    ranks::MVector{SUITS,UInt32}
    shifted_ranks::MVector{SUITS,UInt32}
    suit_idx::MVector{SUITS,UInt64}
    suit_mult::MVector{SUITS,UInt64}

    # Preallocated buffer for zero-allocation hand_index_all! / hand_index_last!
    indices::MVector{MAX_ROUNDS,UInt64}

    function HandIndexerState()
        new(
            @MVector(zeros(UInt64, SUITS)),
            @MVector(ones(UInt64, SUITS)),
            0, 0, 1,
            @MVector(zeros(UInt32, SUITS)),
            @MVector(zeros(UInt32, SUITS)),
            @MVector(zeros(UInt32, SUITS)),
            @MVector(zeros(UInt64, SUITS)),
            @MVector(zeros(UInt64, SUITS)),
            @MVector(zeros(UInt64, MAX_ROUNDS)),
        )
    end
end

# Preallocated scratch buffers for zero-allocation hand_unindex!
# Same fixed-size rationale as HandIndexerState above.
mutable struct HandUnindexState
    suit_index::MVector{SUITS,UInt64}
    used::MVector{SUITS,UInt32}
    m_arr::MVector{SUITS,Int}
    location::MVector{MAX_ROUNDS,Int}

    function HandUnindexState()
        new(
            @MVector(zeros(UInt64, SUITS)),
            @MVector(zeros(UInt32, SUITS)),
            @MVector(zeros(Int, SUITS)),
            @MVector(zeros(Int, MAX_ROUNDS)),
        )
    end
end

# Temporary context used only during hand_indexer_init. Holds the
# indexer being built plus per-round scratch arrays for raw (unpacked)
# offsets/equal values, since those aren't final (cumulative) until
# after the full enumeration pass completes.
struct ConfigTabulateCtx
    indexer::HandIndexer
    temp_offset::Vector{Vector{UInt64}}
    temp_equal::Vector{Vector{UInt32}}
end

# -------------------------------------------------------------------
# Static Tables
# -------------------------------------------------------------------
# const nth_unset = fill(0xff, 1 << RANKS, 33)
const nth_unset = fill(0xff, 1 << RANKS, RANKS)
const nCr_ranks = zeros(UInt32, RANKS + 1, RANKS + 1)
const rank_set_to_index = zeros(UInt32, 1 << RANKS)
const index_to_rank_set = zeros(UInt32, RANKS + 1, 1 << RANKS)
const suit_permutations = Matrix{UInt32}(undef, 24, SUITS)
const nCr_groups = zeros(UInt64, MAX_GROUP_INDEX, SUITS + 1)

function __init__()
    for i in 0:((1 << RANKS) - 1)
        set = (~i) & ((1 << RANKS) - 1)
        j = 0
        while set != 0 && j < RANKS
            nth_unset[i + 1, j + 1] = UInt8(trailing_zeros(set))
            set &= set - 1
            j += 1
        end
    end
    # for i in 0:((1<<RANKS)-1)
    #     set = (~i) & ((1 << RANKS) - 1)
    #     for j in 0:31
    #         if set != 0
    #             nth_unset[i+1, j+1] = UInt8(trailing_zeros(set))
    #             set &= set - 1
    #         end
    #     end
    # end

    nCr_ranks[1, 1] = 1
    for i in 1:RANKS
        nCr_ranks[i+1, 1] = 1
        nCr_ranks[i+1, i+1] = 1
        for j in 1:(i-1)
            nCr_ranks[i+1, j+1] = nCr_ranks[i, j] + nCr_ranks[i, j+1]
        end
    end

    nCr_groups[1, 1] = 1
    for i in 1:(MAX_GROUP_INDEX-1)
        nCr_groups[i+1, 1] = 1
        limit = min(i, SUITS)
        if i < SUITS + 1
            nCr_groups[i+1, i+1] = 1
        end
        for j in 1:(limit-1)
            nCr_groups[i+1, j+1] = nCr_groups[i, j] + nCr_groups[i, j+1]
        end
    end

    for i in 0:((1<<RANKS)-1)
        set = i
        j = 1
        idx = 0
        while set != 0
            cz = trailing_zeros(set)
            idx += nCr_ranks[cz+1, j+1]
            j += 1
            set &= set - 1
        end
        rank_set_to_index[i+1] = idx
        popcnt = count_ones(i)
        index_to_rank_set[popcnt+1, idx+1] = i
    end

    num_permutations = 24
    for i in 0:(num_permutations-1)
        idx = i
        used = 0
        for j in 0:(SUITS-1)
            suit = idx % (SUITS - j)
            idx ÷= (SUITS - j)
            shifted_suit = nth_unset[used+1, suit+1]
            suit_permutations[i+1, j+1] = shifted_suit
            used |= (1 << shifted_suit)
        end
    end
    # __init_nth_unset__()
end

function enumerate_configurations_r!(rounds, cards_per_round, round, remaining, suit, equal, used, configuration, observe_fn, data)
    if suit == SUITS
        observe_fn(round, configuration, data)
        if round + 1 < rounds
            enumerate_configurations_r!(rounds, cards_per_round, round + 1, cards_per_round[round+2], 0, equal, used, configuration, observe_fn, data)
        end
    else
        min_cards = (suit == SUITS - 1) ? remaining : 0
        max_cards = min(remaining, RANKS - used[suit+1])
        previous = RANKS + 1
        was_equal = (equal & (1 << suit)) != 0
        if was_equal
            previous = (configuration[suit] >> (ROUND_SHIFT * (rounds - round - 1))) & ROUND_MASK
            max_cards = min(max_cards, previous)
        end
        old_cfg = configuration[suit+1]
        old_used = used[suit+1]
        for i in min_cards:max_cards
            new_cfg = old_cfg | (UInt32(i) << (ROUND_SHIFT * (rounds - round - 1)))
            new_equal = (equal & ~(1 << suit)) | (UInt32(was_equal && (i == previous)) << suit)
            used[suit+1] = old_used + i
            configuration[suit+1] = new_cfg
            enumerate_configurations_r!(rounds, cards_per_round, round, remaining - i, suit + 1, new_equal, used, configuration, observe_fn, data)
            configuration[suit+1] = old_cfg
            used[suit+1] = old_used
        end
    end
end

function enumerate_configurations(rounds, cards_per_round, observe_fn, data)
    used = @MVector zeros(UInt32, SUITS)
    configuration = @MVector zeros(UInt32, SUITS)
    enumerate_configurations_r!(rounds, cards_per_round, 0, cards_per_round[1], 0, (1 << SUITS) - 2, used, configuration, observe_fn, data)
end

function count_configurations(round, configuration, data)
    data[round+1] += 1
end

# Writes into ctx.temp_offset/ctx.temp_equal (raw, not-yet-cumulative
# values) rather than directly into packed indexer storage, since the
# offset here is still a per-configuration *size* at this point -- it
# only becomes a true offset after the prefix-sum pass in
# hand_indexer_init.
function tabulate_configurations(round, configuration, ctx::ConfigTabulateCtx)
    indexer = ctx.indexer
    r = round + 1
    offset_r = ctx.temp_offset[r]
    equal_r = ctx.temp_equal[r]

    id = indexer.configurations[r]
    indexer.configurations[r] += 1
    while id > 0
        smaller = false
        for i in 1:SUITS
            if configuration[i] < indexer.configuration[r][i, id]
                smaller = true
                break
            elseif configuration[i] > indexer.configuration[r][i, id]
                break
            end
        end
        if smaller
            for i in 1:SUITS
                indexer.configuration[r][i, id+1] = indexer.configuration[r][i, id]
                indexer.configuration_to_suit_size[r][i, id+1] = indexer.configuration_to_suit_size[r][i, id]
            end
            offset_r[id+1] = offset_r[id]
            equal_r[id+1] = equal_r[id]
            id -= 1
        else
            break
        end
    end
    target_id = id + 1
    offset_r[target_id] = 1
    for i in 1:SUITS
        indexer.configuration[r][i, target_id] = configuration[i]
    end
    equal = 0
    i = 1
    while i <= SUITS
        size = UInt64(1)
        for j in 0:round
            ranks = (configuration[i] >> (ROUND_SHIFT * (indexer.rounds - j - 1))) & ROUND_MASK
            remaining = RANKS
            for k in 0:(j-1)
                remaining -= (configuration[i] >> (ROUND_SHIFT * (indexer.rounds - k - 1))) & ROUND_MASK
            end
            size *= nCr_ranks[remaining+1, ranks+1]
        end
        j = i + 1
        while j <= SUITS && configuration[j] == configuration[i]
            j += 1
        end
        for k in i:(j-1)
            indexer.configuration_to_suit_size[r][k, target_id] = UInt32(size)
        end
        offset_r[target_id] *= nCr_groups[size+j-i, j-i+1]
        for k in (i+1):(j-1)
            equal |= (1 << (k - 1))
        end
        i = j
    end
    equal_r[target_id] = equal >> 1
end

function enumerate_permutations_r!(rounds, cards_per_round, round, remaining, suit, used, count, observe_fn, data)
    if suit == SUITS
        observe_fn(round, count, data)
        if round + 1 < rounds
            enumerate_permutations_r!(rounds, cards_per_round, round + 1, cards_per_round[round+2], 0, used, count, observe_fn, data)
        end
    else
        min_cards = (suit == SUITS - 1) ? remaining : 0
        max_cards = min(remaining, RANKS - used[suit+1])
        old_cnt = count[suit+1]
        old_used = used[suit+1]
        for i in min_cards:max_cards
            new_cnt = old_cnt | (UInt32(i) << (ROUND_SHIFT * (rounds - round - 1)))
            used[suit+1] = old_used + i
            count[suit+1] = new_cnt
            enumerate_permutations_r!(rounds, cards_per_round, round, remaining - i, suit + 1, used, count, observe_fn, data)
            count[suit+1] = old_cnt
            used[suit+1] = old_used
        end
    end
end

function enumerate_permutations(rounds, cards_per_round, observe_fn, data)
    used = @MVector zeros(UInt32, SUITS)
    count = @MVector zeros(UInt32, SUITS)
    enumerate_permutations_r!(rounds, cards_per_round, 0, cards_per_round[1], 0, used, count, observe_fn, data)
end

function count_permutations(round, count, indexer)
    r = round + 1
    idx = 0
    mult = 1
    for i in 0:round
        remaining = indexer.cards_per_round[i+1]
        for j in 0:(SUITS-2)
            sz = (count[j+1] >> ((indexer.rounds - i - 1) * ROUND_SHIFT)) & ROUND_MASK
            idx += mult * sz
            mult *= (remaining + 1)
            remaining -= sz
        end
    end
    if indexer.permutations[r] < idx + 1
        indexer.permutations[r] = idx + 1
    end
end

function tabulate_permutations(round, count, indexer)
    r = round + 1
    idx = 0
    mult = 1
    for i in 0:round
        remaining = indexer.cards_per_round[i+1]
        for j in 0:(SUITS-2)
            sz = (count[j+1] >> ((indexer.rounds - i - 1) * ROUND_SHIFT)) & ROUND_MASK
            idx += mult * sz
            mult *= (remaining + 1)
            remaining -= sz
        end
    end
    pi_arr = UInt32[0, 1, 2, 3]
    for i in 2:SUITS
        j = i
        pi_i = pi_arr[i]
        while j > 1
            if count[pi_i+1] > count[pi_arr[j-1]+1]
                pi_arr[j] = pi_arr[j-1]
                j -= 1
            else
                break
            end
        end
        pi_arr[j] = pi_i
    end
    pi_idx = 0
    pi_mult = 1
    pi_used = 0
    for i in 1:SUITS
        this_bit = 1 << pi_arr[i]
        smaller = count_ones((this_bit - 1) & pi_used)
        pi_idx += (pi_arr[i] - smaller) * pi_mult
        pi_mult *= (SUITS - i + 1)
        pi_used |= this_bit
    end
    indexer.permutation_to_pi[r][idx+1] = pi_idx
    low = 1
    high = indexer.configurations[r]
    found_idx = 0
    while low <= high
        mid = (low + high) ÷ 2
        cmp = 0
        for i in 1:SUITS
            this_val = count[pi_arr[i]+1]
            other_val = indexer.configuration[r][i, mid]
            if other_val > this_val
                cmp = -1
                break
            elseif other_val < this_val
                cmp = 1
                break
            end
        end
        if cmp == -1
            high = mid - 1
        elseif cmp == 0
            found_idx = mid - 1
            break
        else
            low = mid + 1
        end
    end
    indexer.permutation_to_configuration[r][idx+1] = found_idx
end

function hand_indexer_init(rounds::Integer, cards_per_round::Vector{UInt8}, indexer::HandIndexer)
    if rounds == 0 || rounds > MAX_ROUNDS || sum(cards_per_round[1:rounds]) > CARDS
        return false
    end

    indexer.rounds = rounds
    indexer.cards_per_round = copy(cards_per_round)
    indexer.round_start = zeros(UInt8, rounds)

    j = 0
    for i in 1:rounds
        indexer.round_start[i] = j
        j += cards_per_round[i]
    end

    fill!(indexer.configurations, 0)
    enumerate_configurations(rounds, cards_per_round, count_configurations, indexer.configurations)

    resize!(indexer.configuration, rounds)
    resize!(indexer.configuration_to_suit_size, rounds)
    resize!(indexer.configuration_to_offset_equal, rounds)

    # Scratch arrays: raw (unpacked, not-yet-cumulative) offset sizes
    # and equal indices, live only for the duration of this function.
    temp_offset = Vector{Vector{UInt64}}(undef, rounds)
    temp_equal = Vector{Vector{UInt32}}(undef, rounds)

    for i in 1:rounds
        num_cfgs = indexer.configurations[i]
        indexer.configuration[i] = zeros(UInt32, SUITS, num_cfgs)
        indexer.configuration_to_suit_size[i] = zeros(UInt32, SUITS, num_cfgs)
        temp_offset[i] = zeros(UInt64, num_cfgs)
        temp_equal[i] = zeros(UInt32, num_cfgs)
    end

    fill!(indexer.configurations, 0)
    ctx = ConfigTabulateCtx(indexer, temp_offset, temp_equal)
    enumerate_configurations(rounds, cards_per_round, tabulate_configurations, ctx)

    for i in 1:rounds
        accum = UInt64(0)
        offs = temp_offset[i]
        eqs = temp_equal[i]
        num_cfgs = indexer.configurations[i]
        packed = Vector{UInt64}(undef, num_cfgs)
        for k in 1:num_cfgs
            nxt = accum + offs[k]
            packed[k] = pack_offset_equal(accum, eqs[k])
            accum = nxt
        end
        indexer.configuration_to_offset_equal[i] = packed
        indexer.round_size[i] = accum
    end

    fill!(indexer.permutations, 0)
    enumerate_permutations(rounds, cards_per_round, count_permutations, indexer)

    resize!(indexer.permutation_to_configuration, rounds)
    resize!(indexer.permutation_to_pi, rounds)

    for i in 1:rounds
        num_perms = indexer.permutations[i]
        indexer.permutation_to_configuration[i] = zeros(UInt32, num_perms)
        indexer.permutation_to_pi[i] = zeros(UInt32, num_perms)
    end

    enumerate_permutations(rounds, cards_per_round, tabulate_permutations, indexer)
    return true
end

function hand_indexer_size(indexer::HandIndexer, round::Integer)
    @assert round < indexer.rounds "Round exceeds initialized indexer bounds"
    return indexer.round_size[round+1]
end

function hand_indexer_state_init!(state::HandIndexerState)
    fill!(state.suit_index, 0)
    fill!(state.suit_multiplier, 1)
    state.round = 0
    state.permutation_index = 0
    state.permutation_multiplier = 1
    fill!(state.used_ranks, 0)
end

@inline Base.@propagate_inbounds function swap!(v::AbstractVector{UInt64}, u::Int, w::Int)
    if v[u] > v[w]
        v[u], v[w] = v[w], v[u]
    end
end

function hand_index_next_round!(indexer::HandIndexer, cards::AbstractVector{<:Integer}, state::HandIndexerState)
    round = state.round
    state.round += 1
    r = round + 1

    fill!(state.ranks, 0)
    fill!(state.shifted_ranks, 0)

    num_cards = indexer.cards_per_round[r]

    @inbounds begin
    for i in 1:num_cards
        card = cards[i]
        rank = deck_get_rank(card)
        suit = deck_get_suit(card)
        rank_bit = UInt32(1) << rank

        state.ranks[suit+1] |= rank_bit
        state.shifted_ranks[suit+1] |= rank_bit >> count_ones((rank_bit - 1) & state.used_ranks[suit+1])
    end

    for i in 1:SUITS
        used_size = count_ones(state.used_ranks[i])
        this_size = count_ones(state.ranks[i])
        state.suit_index[i] += state.suit_multiplier[i] * rank_set_to_index[state.shifted_ranks[i]+1]
        state.suit_multiplier[i] *= nCr_ranks[RANKS-used_size+1, this_size+1]
        state.used_ranks[i] |= state.ranks[i]
    end

    remaining = num_cards
    for i in 1:(SUITS-1)
        this_size = count_ones(state.ranks[i])
        state.permutation_index += state.permutation_multiplier * this_size
        state.permutation_multiplier *= (remaining + 1)
        remaining -= this_size
    end

    perm_idx = state.permutation_index + 1
    configuration = indexer.permutation_to_configuration[r][perm_idx]
    pi_index = indexer.permutation_to_pi[r][perm_idx]

    pv = indexer.configuration_to_offset_equal[r][configuration+1]
    equal_index = unpack_equal(pv)
    offset = unpack_offset(pv)

    for i in 1:SUITS
        pi_suit = suit_permutations[pi_index+1, i] + 1
        state.suit_idx[i] = state.suit_index[pi_suit]
        state.suit_mult[i] = state.suit_multiplier[pi_suit]
    end

    index = offset
    multiplier = UInt64(1)
    i = 1

    while i <= SUITS
        part = UInt64(0)
        size = UInt64(0)
        if i + 1 <= SUITS && is_equal(equal_index, i)
            if i + 2 <= SUITS && is_equal(equal_index, i + 1)
                if i + 3 <= SUITS && is_equal(equal_index, i + 2)
                    swap!(state.suit_idx, i, i + 1)
                    swap!(state.suit_idx, i + 2, i + 3)
                    swap!(state.suit_idx, i, i + 2)
                    swap!(state.suit_idx, i + 1, i + 3)
                    swap!(state.suit_idx, i + 1, i + 2)

                    part = state.suit_idx[i] +
                           nCr_groups[Int(state.suit_idx[i+1])+2, 3] +
                           nCr_groups[Int(state.suit_idx[i+2])+3, 4] +
                           nCr_groups[Int(state.suit_idx[i+3])+4, 5]
                    size = nCr_groups[Int(state.suit_mult[i])+4, 5]
                    i += 4
                else
                    swap!(state.suit_idx, i, i + 1)
                    swap!(state.suit_idx, i, i + 2)
                    swap!(state.suit_idx, i + 1, i + 2)

                    part = state.suit_idx[i] +
                           nCr_groups[Int(state.suit_idx[i+1])+2, 3] +
                           nCr_groups[Int(state.suit_idx[i+2])+3, 4]
                    size = nCr_groups[Int(state.suit_mult[i])+3, 4]
                    i += 3
                end
            else
                swap!(state.suit_idx, i, i + 1)

                part = state.suit_idx[i] + nCr_groups[Int(state.suit_idx[i+1])+2, 3]
                size = nCr_groups[Int(state.suit_mult[i])+2, 3]
                i += 2
            end
        else
            part = state.suit_idx[i]
            size = state.suit_mult[i]
            i += 1
        end
        index += multiplier * part
        multiplier *= size
    end
    end # @inbounds

    return index + 1
end

function hand_index_all!(indexer::HandIndexer, cards::AbstractVector{<:Integer}, indices::AbstractVector{<:Integer},state::HandIndexerState)
    if indexer.rounds > 0
        hand_indexer_state_init!(state)
        card_offset = 1
        for i in 1:indexer.rounds
            cards_cnt = indexer.cards_per_round[i]
            # Zero-allocation slice pointer using @views
            round_cards = @views cards[card_offset:(card_offset+cards_cnt-1)]
            indices[i] = hand_index_next_round!(indexer, round_cards, state)
            card_offset += cards_cnt
        end
        return indices[indexer.rounds]
    end
    return UInt64(0)
end

"""
    hand_index_last!(indexer::HandIndexer, cards::AbstractVector{<:Integer}, state::HandIndexerState) -> UInt64

Compute and return the canonical isomorphic hand index for the **final round** of a 
multi-round poker hand.

### Arguments:
- `indexer`: The initialized `HandIndexer` describing the game structure (rounds, cards per round).
- `cards`: An ordered array of 0-based card IDs (`0:51`) representing all cards dealt up to the final round.
- `state`: Preallocated `HandIndexerState` holding scratch buffers and state trackers to ensure zero heap allocations.

### Returns:
- `UInt64`: The 1-based isomorphic equivalence index for the final round.
"""

function hand_index_last!(indexer::HandIndexer, cards::AbstractVector{<:Integer}, state::HandIndexerState)
    # Reuse the preallocated buffer on `state` instead of allocating a fresh
    # MVector on every call — the MVector was escaping to the heap because
    # it crossed a call boundary typed as AbstractVector{<:Integer}, and
    # escape analysis couldn't prove it didn't need to.
    return hand_index_all!(indexer, cards, state.indices, state)
end

# Convenience overload: allocates a scratch buffer for you. Prefer the
# 5-argument method below (with an explicit HandUnindexState) in hot loops
# to avoid paying this allocation on every call.
function hand_unindex!(indexer::HandIndexer, round::Integer, index::Integer, cards::AbstractVector{<:Integer})
    return hand_unindex!(indexer, round, index, cards, HandUnindexState())
end


"""
    hand_unindex!(indexer::HandIndexer, round::Integer, index::Integer, cards::AbstractVector{<:Integer}, scratch::HandUnindexState) -> Bool

Reconstruct a canonical representative set of cards for a given canonical hand `index` and `round`. This is the inverse of `hand_index_next_round!` / `hand_index_last!`.

### Arguments:
- `indexer`: The initialized `HandIndexer`.
- `round`: The 0-based round index to unindex (e.g., `0` for Preflop, `3` for River in Texas Hold'em).
- `index`: The 1-based canonical index to reconstruct.
- `cards`: Output buffer (destination vector) where reconstructed 0-based card IDs (`0:51`) will be written.
- `scratch`: Preallocated `HandUnindexState` holding reusable vectors for zero heap allocations.

### Returns:
- `Bool`: `true` if unindexing succeeded; `false` if `round` or `index` is out of bounds.
"""

function hand_unindex!(indexer::HandIndexer, round::Integer, index::Integer, cards::AbstractVector{<:Integer}, scratch::HandUnindexState)
    index -= 1
    r = round + 1
    @inbounds if round >= indexer.rounds || index >= indexer.round_size[r]
        return false
    end

    @inbounds packed_r = indexer.configuration_to_offset_equal[r]
    @inbounds suitsize_r = indexer.configuration_to_suit_size[r]
    @inbounds config_r = indexer.configuration[r]

    # Binary search over packed (offset<<3 | equal) entries. Only the
    # offset half is compared; the equal half rides along for free once
    # we land on cfg_idx, saving a separate array touch afterward.
    low = 1
    high = indexer.configurations[r]
    cfg_idx = 1
    @inbounds while low <= high
        mid = (low + high) ÷ 2
        if unpack_offset(packed_r[mid]) <= index
            cfg_idx = mid
            low = mid + 1
        else
            high = mid - 1
        end
    end

    @inbounds pv = packed_r[cfg_idx]
    idx_rem = index - unpack_offset(pv)
    equal_index = unpack_equal(pv)

    suit_index = scratch.suit_index
    fill!(suit_index, 0)

    i = 1
    @inbounds while i <= SUITS
        group_len = 1
        if i + 1 <= SUITS && is_equal(equal_index, i)
            group_len = 2
            if i + 2 <= SUITS && is_equal(equal_index, i + 1)
                group_len = 3
                if i + 3 <= SUITS && is_equal(equal_index, i + 2)
                    group_len = 4
                end
            end
        end

        j = i + group_len
        suit_size = UInt64(suitsize_r[i, cfg_idx])
        group_size = nCr_groups[suit_size+group_len, group_len+1]
        q, rem = divrem(idx_rem, group_size)
        idx_rem = q
        group_index = rem

        if group_len == 1
            suit_index[i] = group_index

        elseif group_len == 2
            base = UInt64(2); col = 3
            low_b = UInt64(0); high_b = suit_size
            low_val = nCr_groups[base, col]
            while low_b < high_b
                mid_b = (low_b + high_b + 1) >>> 1
                v = nCr_groups[mid_b+base, col]
                if v <= group_index
                    low_b = mid_b; low_val = v
                else
                    high_b = mid_b - 1
                end
            end
            suit_index[i] = low_b
            suit_index[i+1] = group_index - low_val

        elseif group_len == 3
            base = UInt64(3); col = 4
            low_b = UInt64(0); high_b = suit_size
            low_val = nCr_groups[base, col]
            while low_b < high_b
                mid_b = (low_b + high_b + 1) >>> 1
                v = nCr_groups[mid_b+base, col]
                if v <= group_index
                    low_b = mid_b; low_val = v
                else
                    high_b = mid_b - 1
                end
            end
            suit_index[i] = low_b
            group_index -= low_val

            base2 = UInt64(2); col2 = 3
            low_b2 = UInt64(0); high_b2 = suit_size
            low_val2 = nCr_groups[base2, col2]
            while low_b2 < high_b2
                mid_b2 = (low_b2 + high_b2 + 1) >>> 1
                v2 = nCr_groups[mid_b2+base2, col2]
                if v2 <= group_index
                    low_b2 = mid_b2; low_val2 = v2
                else
                    high_b2 = mid_b2 - 1
                end
            end
            suit_index[i+1] = low_b2
            suit_index[i+2] = group_index - low_val2

        else # group_len == 4
            base = UInt64(4); col = 5
            low_b = UInt64(0); high_b = suit_size
            low_val = nCr_groups[base, col]
            while low_b < high_b
                mid_b = (low_b + high_b + 1) >>> 1
                v = nCr_groups[mid_b+base, col]
                if v <= group_index
                    low_b = mid_b; low_val = v
                else
                    high_b = mid_b - 1
                end
            end
            suit_index[i] = low_b
            group_index -= low_val

            base2 = UInt64(3); col2 = 4
            low_b2 = UInt64(0); high_b2 = suit_size
            low_val2 = nCr_groups[base2, col2]
            while low_b2 < high_b2
                mid_b2 = (low_b2 + high_b2 + 1) >>> 1
                v2 = nCr_groups[mid_b2+base2, col2]
                if v2 <= group_index
                    low_b2 = mid_b2; low_val2 = v2
                else
                    high_b2 = mid_b2 - 1
                end
            end
            suit_index[i+1] = low_b2
            group_index -= low_val2

            base3 = UInt64(2); col3 = 3
            low_b3 = UInt64(0); high_b3 = suit_size
            low_val3 = nCr_groups[base3, col3]
            while low_b3 < high_b3
                mid_b3 = (low_b3 + high_b3 + 1) >>> 1
                v3 = nCr_groups[mid_b3+base3, col3]
                if v3 <= group_index
                    low_b3 = mid_b3; low_val3 = v3
                else
                    high_b3 = mid_b3 - 1
                end
            end
            suit_index[i+2] = low_b3
            suit_index[i+3] = group_index - low_val3
        end

        i = j
    end

    fill!(cards, 0)

    used = scratch.used
    m_arr = scratch.m_arr
    location = scratch.location
    fill!(used, 0)
    fill!(m_arr, 0)
    @inbounds for i in 1:indexer.rounds
        location[i] = Int(indexer.round_start[i]) + 1
    end

    @inbounds for rnd in 0:round
        rnd_idx = rnd + 1
        loc = location[rnd_idx]
        for s in 1:SUITS
            n = (config_r[s, cfg_idx] >> (ROUND_SHIFT * (indexer.rounds - rnd - 1))) & ROUND_MASK
            if n == 0
                continue
            end

            round_sz = nCr_ranks[RANKS-m_arr[s]+1, n+1]
            si = suit_index[s]
            sub_idx = si % round_sz
            suit_index[s] = si ÷ round_sz
            m_arr[s] += n

            shifted_cards = index_to_rank_set[n+1, sub_idx+1]
            used_before = used[s]
            rank_acc = UInt32(0)
            for k in 1:n
                shifted_card = shifted_cards & -shifted_cards
                shifted_cards ⊻= shifted_card
                tz = trailing_zeros(shifted_card)
                card_rank = nth_unset[used_before+1, tz+1]
                rank_acc |= (1 << card_rank)

                cards[loc] = UInt8(deck_make_card(s - 1, card_rank))
                loc += 1
            end
            used[s] = used_before | rank_acc
        end
        location[rnd_idx] = loc
    end
    return true
end

# function __init_nth_unset__()
#     for i in 0:((1<<RANKS)-1)
#         set = (~i) & ((1 << RANKS) - 1)
#         for j in 0:(RANKS-1)
#             if set != 0
#                 nth_unset[j+1, i+1] = UInt8(trailing_zeros(set))
#                 set &= set - 1
#             end
#         end
#     end
# end


end