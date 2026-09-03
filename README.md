# HandIsomorphism.jl

[![CI](https://github.com/yalnajja/HandIsomorphism.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/yalnajja/HandIsomorphism.jl/actions)
[![Coverage](https://codecov.io/gh/yalnajja/HandIsomorphism.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/yalnajja/HandIsomorphism.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A high-performance, pure-Julia implementation of Kevin Waugh’s **suit-isomorphic hand indexing algorithm**. 

`HandIsomorphism.jl` computes minimal canonical equivalence classes for poker hands in multi-round extensive-form games such as Texas Hold'em, Omaha, mapping each hand to a contiguous integer index. These equivalence classes can be used to reduce game-state representations and strategy spaces in applications such as Counterfactual Regret Minimization (CFR).

---

## Key Features

- **Fast**: Usualy faster than Waugh's reference C implementation (`-O3 -DNDEBUG`) across most indexing and unindexing benchmarks when called with ccall.
- **Zero Allocations**: Preallocated state structs (`HandIndexerState`, `HandUnindexState`) ensure $0\text{ bytes}$ allocated on hot paths.
- **Arbitrary Game Structures**: Supports any multi-round card structure (e.g., 2-round Flop `[2, 3]`, 4-round Texas Hold'em `[2, 3, 1, 1]`, Omaha `[4, 3, 1, 1]`).
- **Bidirectional**: Supports fast indexing (`cards -> index`) and unindexing/reconstruction (`index -> canonical cards`).
- **Memory Optimized**: Employs packed 64-bit cumulative offsets and equal-suit bitmasks to minimize cache footprint.

---

## Card Representation

Cards are encoded as zero-based 8-bit unsigned integers (`0:51`):

$$\text{card} = (\text{rank} \ll 2) \mid \text{suit}$$

- **Ranks**: `0` = 2, `1` = 3, ..., `8` = 10, `9` = Jack, `10` = Queen, `11` = King, `12` = Ace
- **Suits**: `0`, `1`, `2`, `3`

Helper utility functions:
```julia
deck_make_card(suit, rank)  # (rank << 2) | suit
deck_get_rank(card)         # card >> 2
deck_get_suit(card)         # card & 3
```

---

## Quick Start

### 1. Initialization

Define the structure of your game by passing the number of cards dealt in each round:

```julia
using HandIsomorphism

# Texas Hold'em: Preflop (2 cards), Flop (3 cards), Turn (1 card), River (1 card)
cards_per_round = UInt8[2, 3, 1, 1]
rounds = length(cards_per_round)

indexer = HandIndexer(cards_per_round)

# Query the size of the isomorphic state space at each round
for r in 0:(rounds - 1)
    println("Round $(r) canonical hand count: ", hand_indexer_size(indexer, r))
end
# Round 0 (Preflop): 169
# Round 1 (Flop):    1,286,792
# Round 2 (Turn):    55,190,538
# Round 3 (River):   2,428,287,420
```

---

### 2. Indexing Hands (Cards $\to$ Index)

Indices are **1-based** canonical values.
> **Index Equivalence Note:**  
> Julia uses 1-based indexing conventions. `HandIsomorphism.jl` produces canonical indices that are strictly **$I_{\text{Julia}} = I_{\text{C}} + 1$**, matching Kevin Waugh's reference C implementation identically. 
```julia
# Preallocate the reusable state tracker for 0-allocation calls
state = HandIndexerState()

# Construct a 7-card hand: [As, Ks] + [Ah, Kh, 2c] + [3d] + [4s]
# As = (12 << 2) | 0 = 48, Ks = (11 << 2) | 0 = 44, etc.
hand = UInt8[
    deck_make_card(0, 12), deck_make_card(0, 11), # Hole cards: As, Ks
    deck_make_card(1, 12), deck_make_card(1, 11), deck_make_card(2, 0), # Flop: Ah, Kh, 2c
    deck_make_card(3, 1),                          # Turn: 3d
    deck_make_card(0, 2)                           # River: 4s
]

# Option A: Get the final round index directly
river_idx = hand_index_last!(indexer, hand, state)
println("River Canonical Index: ", river_idx)

# Option B: Get indices for all intermediate rounds simultaneously
indices = zeros(UInt64, rounds)
hand_index_all!(indexer, hand, indices, state)
println("Preflop Index: ", indices[1])
println("Flop Index:    ", indices[2])
println("Turn Index:    ", indices[3])
println("River Index:   ", indices[4])
```

---

### 3. Step-by-Step Incremental Indexing

For game trees where cards are revealed sequentially round-by-round:

```julia
hand_indexer_state_init!(state)

# Preflop (2 cards)
hole_cards = UInt8[deck_make_card(0, 12), deck_make_card(1, 12)] # Pair of Aces: As, Ah
preflop_idx = hand_index_next_round!(indexer, hole_cards, state)

# Flop (3 cards)
flop_cards = UInt8[deck_make_card(0, 11), deck_make_card(1, 11), deck_make_card(2, 0)] # Ks, Kh, 2c
flop_idx = hand_index_next_round!(indexer, flop_cards, state)
```

---

### 4. Unindexing Hands (Index $\to$ Canonical Cards)

When unindexing, `HandIsomorphism.jl` expects a 1-based index `[1, hand_indexer_size]`. 


Reconstruct a canonical hand from an index:

```julia
unindexState = HandUnindexState()
out_cards = zeros(UInt8, sum(cards_per_round))

target_round = 3 # 0-based (3 = River)
canonical_index = river_idx

success = hand_unindex!(indexer, target_round, canonical_index, out_cards, unindexState)

if success
    println("Reconstructed cards: ", out_cards)
end
```

### Performance Benchmarks (Per Hand)



Benchmarked against Kevin Waugh's reference C implementation compiled with `-std=c99 -Wall -O3 -DNDEBUG -fPIC` called from Julia using ccall across 10,000 random hands per configuration:

| Game / Configuration | Operation | Julia (`HandIsomorphism.jl`) | C Reference | Speedup | Allocations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Flop** `[2, 3]` | Indexing | **67.36 ns / hand** | 86.50 ns / hand | **+28.4% faster** | 0 bytes |
| | Unindexing | **68.30 ns / hand** | 89.94 ns / hand | **+31.7% faster** | 0 bytes |
| **Turn (Abbrev.)** `[2, 4]` | Indexing | **69.77 ns / hand** | 88.73 ns / hand | **+27.2% faster** | 0 bytes |
| | Unindexing | **82.82 ns / hand** | 100.50 ns / hand | **+21.3% faster** | 0 bytes |
| **River (Abbrev.)** `[2, 5]` | Indexing | **70.27 ns / hand** | 92.42 ns / hand | **+31.5% faster** | 0 bytes |
| | Unindexing | **83.95 ns / hand** | 94.16 ns / hand | **+12.2% faster** | 0 bytes |
| **Full Hold'em River** `[2, 3, 1, 1]` | Indexing | **141.00 ns / hand** | 159.60 ns / hand | **+13.2% faster** | 0 bytes |
| | Unindexing | **114.80 ns / hand** | 106.20 ns / hand | *~7.5% slower* | 0 bytes |

---

## Architecture & Optimizations

1. **Offset-Equal Bit Packing**: Stores cumulative configuration offsets and equal-suit permutation flags in a single packed `UInt64` (`offset << 3 | equal`), saving cache space and cutting memory lookups during binary search.
2. **Branchless Bitmask Checks**: Direct bit tests eliminate matrix lookups in hot suit-grouping routines.

---

## API Reference

### Structs
- `HandIndexer(card_per_round)`: Top-level structure holding configuration tables and combinatorial mappings for a game.
- `HandIndexerState()`: Mutable state container for indexing operations.
- `HandUnindexState()`: Mutable state container for unindexing operations.

### Indexing Methods
- `hand_indexer_size(indexer, round) -> UInt64`
- `hand_indexer_state_init!(indexerState)`
- `hand_index_next_round!(indexer, cards, indexerState) -> UInt64`
- `hand_index_all!(indexer, cards, indices, indexerState) -> UInt64`
- `hand_index_last!(indexer, cards, indexerState) -> UInt64`

### Unindexing Methods
- `hand_unindex!(indexer, round, index, cards, unindexState) -> Bool`

---

## References

- **Kevin Waugh** (2013): *A Fast and Optimal Hand Isomorphism Algorithm*. Computer Poker Research Group, University of Alberta.
- [Reference C implementation by Kevin Waugh](https://github.com/kdub0/hand-isomorphism)

---

## License

This project is licensed under the [MIT License](LICENSE).

The C implementation vendored in `test/hand-isomorphism/` is third-party code by Kevin Waugh and is distributed under its own BSD-style license. This code is subject to the terms of that separate license.
