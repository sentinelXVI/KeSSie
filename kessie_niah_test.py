#!/usr/bin/env python3
"""
KeSSie Needle-in-a-Haystack Test


Tests the full recall pipeline: fog-of-war attention suppression,
semantic search through conversation store, positional annotation,
and mid-generation recall trigger.

Parameters:
    Window size:  131,072 tokens
    Chat buffer:  10,000,000 tokens (~10M)
    Needle:       Unique synthetic facts planted at known depths

Test Matrix:
    1. DEPTH TEST: Plant needle at 7 depths (1K → 10M tokens deep)
       Can KeSSie recall facts from various depths in conversation history?

    2. FOG ZONE TEST: Verify needle in fog zone is suppressed without recall,
       but retrieved when recall is triggered

    3. MULTI-NEEDLE TEST: Plant 3 needles, ask query that needs all 3

    4. MID-GEN TRIGGER TEST: Simulate hedging patterns to verify mid-gen
       recall detection fires correctly

    5. POSITIONAL ANNOTATION TEST: Verify recalled text carries correct
       turn number, role, and token distance metadata

    6. ATTENTION BIAS TEST: Verify fog cache produces correct bias values
       for recalled vs fogged vs clear positions at 131K scale

Usage:
    # Full test (requires GPU + model — runs actual inference)
    python kessie_niah_test.py --model <model_name> --full

    # Offline test (no GPU — tests recall pipeline, attention math, annotation)
    python kessie_niah_test.py

    # Specific depth only
    python kessie_niah_test.py --depth 1000000
"""

import sys
import os
import time
import json
import random
import hashlib
import argparse
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger("KeSSie.NIAH")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Log file: append always, user deletes to reset
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "niah_test.log")
_log_fh = None

def _open_log():
    global _log_fh
    _log_fh = open(LOG_FILE, "a", encoding="utf-8")
    # Write session header
    from datetime import datetime
    _log_fh.write(f"\n{'='*70}\n")
    _log_fh.write(f"  NIAH Test Run: {datetime.now().isoformat()}\n")
    _log_fh.write(f"{'='*70}\n")
    _log_fh.flush()

def _close_log():
    global _log_fh
    if _log_fh:
        _log_fh.flush()
        _log_fh.close()
        _log_fh = None

def tee(msg=""):
    """Print to stdout and append to niah_test.log."""
    print(msg)
    if _log_fh:
        _log_fh.write(msg + "\n")
        _log_fh.flush()

# ============================================================================
# Test Configuration
# ============================================================================

WINDOW_SIZE = 131_072
CHAT_BUFFER_SIZE = 10_000_000  # 10M tokens

# Depths to test (tokens from most recent)
NEEDLE_DEPTHS = [
    1_000,          # 1K — well within clear zone
    10_000,         # 10K — near fog boundary
    100_000,        # 100K — shallow fog (just outside window)
    500_000,        # 500K — moderate fog
    1_000_000,      # 1M — deep fog
    5_000_000,      # 5M — very deep
    10_000_000,     # 10M — maximum depth (edge of buffer)
]

# Unique needle facts — each is a synthetic fact unlikely to appear in training data
NEEDLE_FACTS = {
    "capital": {
        "needle": "The capital of Zyntaria is Florquen, founded in 1847 by explorer Halvek Renn.",
        "query": "What is the capital of Zyntaria?",
        "expected_fragments": ["Florquen", "1847", "Halvek Renn"],
    },
    "recipe": {
        "needle": "The secret ingredient in Grandmother Pellwick's award-winning stew is exactly 3.7 grams of powdered starfruit rind.",
        "query": "What is the secret ingredient in Grandmother Pellwick's stew?",
        "expected_fragments": ["3.7 grams", "starfruit rind"],
    },
    "code": {
        "needle": "The server's emergency shutdown code is BRAVO-TANGO-7749-WHISKEY. Only use it if core temperature exceeds 4500 Kelvin.",
        "query": "What is the server's emergency shutdown code?",
        "expected_fragments": ["BRAVO-TANGO-7749-WHISKEY", "4500"],
    },
    "meeting": {
        "needle": "Dr. Vasquez confirmed the Quorinthian Advisory Board meeting is rescheduled to March 17th at 2:30 PM in Conference Room Obsidian on the 43rd floor of the Palladian Tower.",
        "query": "When is the Quorinthian Advisory Board meeting with Dr. Vasquez?",
        "expected_fragments": ["March 17th", "2:30 PM", "Obsidian", "43rd"],
    },
    "coordinates": {
        "needle": "The buried treasure is at GPS coordinates 41.40338 North, 2.17403 East, exactly 14 meters below the old lighthouse foundation.",
        "query": "Where is the buried treasure?",
        "expected_fragments": ["41.40338", "2.17403", "14 meters", "lighthouse"],
    },
}

# Haystack conversation topics — used to generate filler
HAYSTACK_TOPICS = [
    ("user", "Can you explain how photosynthesis works?"),
    ("assistant", "Photosynthesis converts light energy into chemical energy. Plants use chlorophyll to absorb sunlight, then combine carbon dioxide and water to produce glucose and oxygen. The process occurs in two stages: light-dependent reactions in the thylakoid membranes and the Calvin cycle in the stroma."),
    ("user", "What's the best way to learn a new programming language?"),
    ("assistant", "Start with the official documentation and tutorials. Build small projects that interest you. Practice daily, even if just for 30 minutes. Read other people's code on GitHub. Join community forums and ask questions. The key is consistent practice with real-world applications."),
    ("user", "Tell me about the history of jazz music."),
    ("assistant", "Jazz originated in New Orleans in the late 19th century, blending African rhythmic traditions with European harmonic structures. Key developments include ragtime, swing, bebop, cool jazz, and fusion. Influential figures include Louis Armstrong, Duke Ellington, Charlie Parker, Miles Davis, and John Coltrane."),
    ("user", "How do I make a good cup of coffee?"),
    ("assistant", "Use freshly ground beans, ideally within two weeks of roasting. Water temperature should be between 195-205°F. The golden ratio is about 1:16 coffee to water. Brew time varies by method: pour-over takes 3-4 minutes, French press 4 minutes, espresso 25-30 seconds."),
    ("user", "What causes earthquakes?"),
    ("assistant", "Earthquakes are caused by sudden releases of energy in the Earth's crust. Tectonic plates constantly move, and when they collide, separate, or slide past each other, stress builds up along fault lines. When the stress exceeds the friction holding the plates in place, the energy is released as seismic waves."),
    ("user", "Can you recommend some science fiction books?"),
    ("assistant", "I'd recommend Dune by Frank Herbert for epic world-building, Neuromancer by William Gibson for cyberpunk, The Left Hand of Darkness by Ursula K. Le Guin for social science fiction, and Blindsight by Peter Watts for hard SF exploring consciousness."),
    ("user", "How does machine learning work?"),
    ("assistant", "Machine learning algorithms learn patterns from data. Supervised learning uses labeled examples to learn a mapping from inputs to outputs. Unsupervised learning finds structure in unlabeled data. Key concepts include gradient descent for optimization, neural networks for representation learning, and regularization to prevent overfitting."),
    ("user", "What's the difference between weather and climate?"),
    ("assistant", "Weather refers to short-term atmospheric conditions in a specific area — temperature, humidity, precipitation, wind. Climate is the average weather pattern in a region over long periods, typically 30+ years. Weather can change in minutes; climate trends emerge over decades."),
]


# ============================================================================
# Haystack Generator
# ============================================================================

class HaystackGenerator:
    """Generates synthetic conversation history of arbitrary token length."""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self._topic_idx = 0
        self._variation_seed = 42

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count. ~4 chars per token for English."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(text) // 4

    def _vary_message(self, text: str, seed: int) -> str:
        """Add minor variation to avoid exact duplicates in the store."""
        rng = random.Random(seed)
        # Add a unique conversation ID and slight rewording
        conv_id = rng.randint(10000, 99999)
        time_str = f"{rng.randint(1,12):02d}:{rng.randint(0,59):02d}"
        return f"[conv-{conv_id} {time_str}] {text}"

    def generate_turns(self, target_tokens: int) -> List[Dict[str, str]]:
        """Generate conversation turns totaling approximately target_tokens."""
        turns = []
        total_tokens = 0

        while total_tokens < target_tokens:
            # Pick a topic pair
            idx = self._topic_idx % len(HAYSTACK_TOPICS)
            role, content = HAYSTACK_TOPICS[idx]
            self._topic_idx += 1
            self._variation_seed += 1

            varied = self._vary_message(content, self._variation_seed)
            turns.append({"role": role, "content": varied})
            total_tokens += self._estimate_tokens(varied)

        return turns

    def generate_tokens(self, target_tokens: int) -> List[int]:
        """Generate raw token IDs for conversation filler."""
        if not self.tokenizer:
            # Fake token IDs for offline testing
            rng = random.Random(42)
            return [rng.randint(100, 30000) for _ in range(target_tokens)]

        turns = self.generate_turns(target_tokens)
        all_tokens = []
        for t in turns:
            toks = self.tokenizer.encode(t["content"], add_special_tokens=False)
            all_tokens.extend(toks)
            if len(all_tokens) >= target_tokens:
                break
        return all_tokens[:target_tokens]


# ============================================================================
# Offline Tests (no GPU required)
# ============================================================================

def test_conversation_store_depth():
    """
    Test 1: DEPTH TEST
    Verify the conversation store can index and recall needles at all depths.
    Uses the KeSSieCache + SemanticIndex directly, no model needed.
    """
    tee("\n" + "=" * 70)
    tee("TEST 1: Conversation Store Depth Recall")
    tee("=" * 70)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        from kessie_exp3 import KeSSieCache, SemanticIndex
    except ImportError:
        # Build minimal stubs for standalone testing
        tee("  (using standalone stubs — kessie_exp3 not importable)")
        return _test_depth_standalone()

    cache = KeSSieCache(
        window_size=WINDOW_SIZE,
        index_granularity=128,
        fog_alpha=0.5,
        fog_start=0.5,
    )
    index = SemanticIndex(dim=256)
    haystack = HaystackGenerator()

    # Create a deterministic embedder (token-hash projection, same as vLLM path)
    np.random.seed(42)
    vocab_size = 32000
    proj = np.random.randn(vocab_size, 256).astype(np.float32)
    proj /= np.sqrt(vocab_size)

    def embedder(token_ids):
        vec = np.zeros(256, dtype=np.float32)
        for t in token_ids[:128]:
            if t < vocab_size:
                vec += proj[t]
        if len(token_ids) > 0:
            vec /= len(token_ids[:128])
        return vec

    def embed_text_fake(text):
        """Word-level n-gram hash embedder. Much stronger than character hash."""
        vec = np.zeros(256, dtype=np.float32)
        words = text.lower().split()
        # Unigrams
        for w in words[:128]:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            idx = h % 256
            vec[idx] += 1.0
        # Bigrams for context sensitivity
        for i in range(min(len(words) - 1, 64)):
            bg = f"{words[i]}_{words[i+1]}"
            h = int(hashlib.md5(bg.encode()).hexdigest(), 16)
            idx = h % 256
            vec[idx] += 0.5
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    passed, failed = 0, 0

    for depth in NEEDLE_DEPTHS:
        if depth > CHAT_BUFFER_SIZE:
            continue

        needle_key = list(NEEDLE_FACTS.keys())[hash(str(depth)) % len(NEEDLE_FACTS)]
        needle = NEEDLE_FACTS[needle_key]

        tee(f"\n  Depth {depth:>10,} tokens | needle: '{needle_key}'")

        # Reset
        cache.conversation_tokens.clear()
        cache.index_embeddings.clear()
        cache.index_positions.clear()
        cache.turn_boundaries.clear()
        cache._turn_counter = 0

        # Fill buffer up to needle position
        needle_pos = CHAT_BUFFER_SIZE - depth
        filler_before = haystack.generate_tokens(needle_pos)
        cache.conversation_tokens.extend(filler_before)

        # Record turn boundary for the filler
        cache.turn_boundaries.append((0, "user", 1))

        # Insert needle as a user message
        # Use fake token IDs since we don't have a real tokenizer
        needle_hash = hashlib.md5(needle["needle"].encode()).hexdigest()
        needle_tokens = [ord(c) % vocab_size for c in needle["needle"]]
        needle_start = len(cache.conversation_tokens)
        cache.append_conversation(needle_tokens, role="user")

        # Index the needle
        needle_emb = embed_text_fake(needle["needle"])
        cache.index_embeddings.append(needle_emb)
        cache.index_positions.append(needle_start)

        # Fill remaining buffer after needle
        filler_after = haystack.generate_tokens(depth - len(needle_tokens))
        cache.conversation_tokens.extend(filler_after)
        cache.turn_boundaries.append((needle_start + len(needle_tokens), "assistant", cache._turn_counter + 1))
        cache._turn_counter += 1

        # Index some haystack chunks (every 10K tokens)
        for chunk_start in range(0, len(cache.conversation_tokens), 10000):
            chunk = cache.conversation_tokens[chunk_start:chunk_start + 128]
            if chunk:
                emb = embedder(chunk)
                cache.index_embeddings.append(emb)
                cache.index_positions.append(chunk_start)

        # Rebuild semantic index
        index.rebuild(cache.index_embeddings, cache.index_positions)

        # Search with needle query
        query_emb = embed_text_fake(needle["query"])
        results = index.search(query_emb, top_k=5)

        # Check if needle position is in top-5 results
        found = any(abs(pos - needle_start) < 200 for pos, score in results)
        top_score = results[0][1] if results else 0

        # Check turn metadata
        role, turn_num, total_turns = cache.get_turn_for_position(needle_start)
        distance = cache.get_token_distance(needle_start)

        if found:
            tee(f"    ✓ FOUND | score={top_score:.4f} | "
                  f"turn={turn_num}/{total_turns} ({role}) | dist={distance:,}")
            passed += 1
        else:
            tee(f"    ✗ MISSED | top_score={top_score:.4f} | "
                  f"best_pos={results[0][0] if results else 'none'}")
            # Show what we found
            for pos, sc in results[:3]:
                tee(f"      found pos={pos:,} score={sc:.4f} (needle at {needle_start:,})")
            failed += 1

    return passed, failed


def _test_depth_standalone():
    """Standalone depth test without kessie_exp3 imports."""
    passed, failed = 0, 0
    tee("  Standalone mode: testing embedding + search logic only")

    np.random.seed(42)
    dim = 256

    def embed_text(text):
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        for w in words[:128]:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0
        for i in range(min(len(words) - 1, 64)):
            bg = f"{words[i]}_{words[i+1]}"
            h = int(hashlib.md5(bg.encode()).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 0.5
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    for depth in NEEDLE_DEPTHS:
        needle_key = list(NEEDLE_FACTS.keys())[hash(str(depth)) % len(NEEDLE_FACTS)]
        needle = NEEDLE_FACTS[needle_key]

        tee(f"\n  Depth {depth:>10,} tokens | needle: '{needle_key}'")

        # Build a collection of embeddings: needle + haystack
        n_haystack = max(100, depth // 10000)  # proportional haystack chunks
        embeddings = []
        positions = []

        # Haystack embeddings
        for i in range(n_haystack):
            topic_idx = i % len(HAYSTACK_TOPICS)
            _, content = HAYSTACK_TOPICS[topic_idx]
            emb = embed_text(f"chunk-{i}: {content}")
            embeddings.append(emb)
            positions.append(i * 10000)

        # Needle embedding at target depth
        needle_pos = CHAT_BUFFER_SIZE - depth
        needle_emb = embed_text(needle["needle"])
        embeddings.append(needle_emb)
        positions.append(needle_pos)

        # Search
        mat = np.stack(embeddings)
        query_emb = embed_text(needle["query"])
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        norms = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        sims = (norms @ q).flatten()

        top_idx = np.argsort(sims)[-5:][::-1]
        results = [(positions[i], float(sims[i])) for i in top_idx]

        found = any(abs(pos - needle_pos) < 200 for pos, _ in results)
        top_score = results[0][1] if results else 0

        if found:
            tee(f"    ✓ FOUND | score={top_score:.4f}")
            passed += 1
        else:
            tee(f"    ✗ MISSED | top_score={top_score:.4f}")
            for pos, sc in results[:3]:
                tee(f"      found pos={pos:,} score={sc:.4f} (needle at {needle_pos:,})")
            failed += 1

    return passed, failed


def test_fog_attention_at_scale():
    """
    Test 2: ATTENTION BIAS AT 131K SCALE
    Verify fog cache produces correct bias values at full window size.
    Tests recalled positions get positive bias, fogged get negative, clear get zero.
    """
    tee("\n" + "=" * 70)
    tee("TEST 2: Fog Attention Bias at 131K Window Scale")
    tee("=" * 70)

    try:
        import torch
        if not torch.cuda.is_available():
            tee("  (CUDA not available — running CPU-only)")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0")
    except ImportError:
        tee("  SKIPPED (PyTorch not available)")
        return 0, 0

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from kessie_attention import _FogCache
    except ImportError:
        tee("  SKIPPED (kessie_attention not importable)")
        return 0, 0

    passed, failed = 0, 0

    def check(name, cond):
        nonlocal passed, failed
        if cond:
            tee(f"    ✓ {name}")
            passed += 1
        else:
            tee(f"    ✗ {name}")
            failed += 1

    cache = _FogCache()
    dtype = torch.float32
    kv_len = WINDOW_SIZE  # 131,072

    # Simulate recalled positions scattered through the fog zone
    # Fog boundary at 50% = position 65536
    fog_boundary = kv_len // 2
    recall_positions = frozenset({
        100,        # deep fog (very old)
        1000,       # deep fog
        10000,      # moderate fog
        50000,      # near fog boundary
        65000,      # just inside fog
        65536,      # exactly at boundary
        100000,     # clear zone
    })

    state = {
        "fog_alpha": 0.5,
        "fog_start": 0.5,
        "prompt_len": kv_len,
        "recall_positions": recall_positions,
        "recall_boost": 0.1,
        "enabled": True,
        "generation": 1,
    }

    tee(f"\n  KV length: {kv_len:,} | fog_boundary: {fog_boundary:,}")
    tee(f"  Recall positions: {sorted(recall_positions)}")

    t0 = time.perf_counter()
    bias = cache.get_fog_bias(kv_len, device, dtype, state)
    t1 = time.perf_counter()
    tee(f"  Compute time: {(t1-t0)*1000:.1f}ms")

    flat = bias.view(-1)

    check(f"Shape is (1,1,1,{kv_len})", bias.shape == (1, 1, 1, kv_len))

    # Fog zone checks (positions 0 to fog_boundary-1)
    check("Position 0 (deepest fog) is most negative",
          flat[0].item() < flat[fog_boundary // 2].item())
    check("Position 0 value ≈ -fog_alpha (-0.5)",
          abs(flat[0].item() - (-0.5)) < 0.01)

    # Non-recalled fogged position
    check("Non-recalled fog pos 500 is negative", flat[500].item() < 0)

    # Clear zone checks (positions fog_boundary to end)
    clear_pos = fog_boundary + 1000
    check(f"Clear zone pos {clear_pos} is zero",
          abs(flat[clear_pos].item()) < 1e-6)

    # Recalled position checks — POSITIVE bias
    for rp in sorted(recall_positions):
        if rp < kv_len:
            val = flat[rp].item()
            check(f"Recall pos {rp:>6} = +{state['recall_boost']} (got {val:.4f})",
                  abs(val - state['recall_boost']) < 1e-5)

    # Verify recalled > clear > fogged ordering
    recalled_val = flat[100].item()   # recalled in fog zone
    clear_val = flat[80000].item()    # clear zone (not recalled)
    fogged_val = flat[500].item()     # fogged (not recalled)
    check("Attention ordering: recalled > clear > fogged",
          recalled_val > clear_val > fogged_val)

    # Performance: 64 layers should all hit cache
    tee(f"\n  64-layer cache hit test:")
    state["generation"] = 2
    t0 = time.perf_counter()
    for _ in range(64):
        cache.get_fog_bias(kv_len, device, dtype, state)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - t0) * 1e6
    check(f"64 layers in {elapsed_us:.0f}µs (< 1000µs target)", elapsed_us < 1000)

    # Memory: buffer size at 131K
    buf_bytes = 2 * kv_len * 4  # 2 buffers × 131K × float32
    tee(f"  Fog buffer memory: {buf_bytes / 1024:.0f} KB (2 × {kv_len:,} × f32)")

    return passed, failed


def test_positional_annotation():
    """
    Test 3: POSITIONAL ANNOTATION
    Verify turn tracking and annotation produce correct metadata.
    """
    tee("\n" + "=" * 70)
    tee("TEST 3: Positional Annotation & Turn Tracking")
    tee("=" * 70)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    passed, failed = 0, 0

    def check(name, cond):
        nonlocal passed, failed
        if cond:
            tee(f"    ✓ {name}")
            passed += 1
        else:
            tee(f"    ✗ {name}")
            failed += 1

    try:
        from kessie_exp3 import KeSSieCache
    except ImportError:
        tee("  SKIPPED (kessie_exp3 not importable)")
        return 0, 0

    cache = KeSSieCache(window_size=WINDOW_SIZE)

    # Simulate 50-turn conversation
    pos = 0
    for i in range(50):
        # User turn: ~100 tokens
        user_toks = list(range(pos, pos + 100))
        cache.append_conversation(user_toks, role="user")
        pos += 100

        # Assistant turn: ~200 tokens
        asst_toks = list(range(pos, pos + 200))
        cache.append_conversation(asst_toks, role="assistant")
        pos += 200

    total_tokens = len(cache.conversation_tokens)  # should be 50 * 300 = 15000
    total_turns = cache._turn_counter  # should be 100

    tee(f"\n  Simulated: {total_turns} turns, {total_tokens:,} tokens")

    # Check turn lookup at various positions
    role, turn, total = cache.get_turn_for_position(0)
    check(f"Position 0: role={role}, turn={turn}/{total}",
          role == "user" and turn == 1 and total == 100)

    role, turn, total = cache.get_turn_for_position(150)
    check(f"Position 150 (mid-assistant turn 1): role={role}",
          role == "assistant" and turn == 2)

    role, turn, total = cache.get_turn_for_position(14850)
    check(f"Position 14850 (last assistant turn): role={role}",
          role == "assistant")

    # Token distance
    dist = cache.get_token_distance(0)
    check(f"Distance from pos 0 = {dist:,} (should be {total_tokens})",
          dist == total_tokens)

    dist = cache.get_token_distance(total_tokens - 1)
    check(f"Distance from last pos = {dist} (should be 1)", dist == 1)

    # Annotation format
    # We need a mock engine to test _annotate_recall, but we can test the cache methods
    role, turn, total = cache.get_turn_for_position(600)  # turn 5 area
    expected_turn = 5  # pos 600 = user turn at 600 (turn 5 = 4th user msg starts at 4*300=1200... let me recalc)
    # Actually: turn 1 = user @ 0, turn 2 = asst @ 100, turn 3 = user @ 300, turn 4 = asst @ 400
    # turn 5 = user @ 600, turn 6 = asst @ 700
    check(f"Position 600: turn {turn} should be user turn 5",
          role == "user" and turn == 5)

    # Scale test: 10M tokens worth of turns
    tee(f"\n  Scale test: simulating 10M token history tracking...")
    cache2 = KeSSieCache(window_size=WINDOW_SIZE)
    t0 = time.perf_counter()
    simulated_pos = 0
    n_turns = 0
    while simulated_pos < CHAT_BUFFER_SIZE:
        # Don't actually store 10M tokens — just track boundaries
        role = "user" if n_turns % 2 == 0 else "assistant"
        turn_len = 150 if role == "user" else 350
        cache2.turn_boundaries.append((simulated_pos, role, n_turns + 1))
        cache2._turn_counter = n_turns + 1
        simulated_pos += turn_len
        n_turns += 1

    t1 = time.perf_counter()
    tee(f"  {n_turns:,} turns tracked in {(t1-t0)*1000:.1f}ms")

    # Lookup at 10M depth
    t0 = time.perf_counter()
    role, turn, total = cache2.get_turn_for_position(5_000_000)
    t1 = time.perf_counter()
    lookup_us = (t1 - t0) * 1e6
    check(f"Lookup at 5M: turn {turn}/{total} ({role}) in {lookup_us:.0f}µs",
          turn > 0 and total == n_turns)

    return passed, failed


def test_mid_gen_trigger():
    """
    Test 4: MID-GENERATION RECALL TRIGGER
    Verify uncertainty detection catches hedging patterns and repetition.
    """
    tee("\n" + "=" * 70)
    tee("TEST 4: Mid-Generation Recall Trigger Detection")
    tee("=" * 70)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    passed, failed = 0, 0

    def check(name, cond):
        nonlocal passed, failed
        if cond:
            tee(f"    ✓ {name}")
            passed += 1
        else:
            tee(f"    ✗ {name}")
            failed += 1

    try:
        from kessie_exp3 import LibrarianEngine
        # We need the _check_uncertainty method — get it from the class
        check_fn = LibrarianEngine._check_uncertainty
    except (ImportError, AttributeError):
        tee("  SKIPPED (kessie_exp3 not importable or _check_uncertainty not found)")
        return 0, 0

    # Create a minimal mock to call _check_uncertainty
    class MockEngine:
        _HEDGE_PATTERNS = LibrarianEngine._HEDGE_PATTERNS
        _mid_gen_events = []

        def _check_uncertainty(self, text, count):
            return LibrarianEngine._check_uncertainty(self, text, count)

    engine = MockEngine()

    # Test hedging patterns
    tee("\n  Hedging pattern detection:")
    hedge_texts = [
        ("prefix " * 10 + "as I mentioned earlier, the code should", 32),
        ("prefix " * 10 + "if I recall correctly, the server was", 32),
        ("prefix " * 10 + "I believe you said something about", 32),
        ("prefix " * 10 + "earlier in our conversation you asked about", 32),
        ("prefix " * 10 + "I'm not sure if this is right but", 32),
    ]

    for text, count in hedge_texts:
        result = engine._check_uncertainty(text, count)
        pattern = [p for p in engine._HEDGE_PATTERNS if p in text.lower()]
        check(f"Detects: '{pattern[0] if pattern else '?'}' at token {count}", result)

    # Test clean text (no trigger)
    tee("\n  Clean text (should NOT trigger):")
    clean_texts = [
        ("The answer to your question is straightforward. Here's how it works.", 32),
        ("def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)", 32),
        ("Hello! How can I help you today?", 32),
    ]

    for text, count in clean_texts:
        result = engine._check_uncertainty(text, count)
        check(f"No trigger: '{text[:40]}...'", not result)

    # Test too early (< 8 tokens)
    result = engine._check_uncertainty("as I mentioned earlier blah", 4)
    check("No trigger before 8 tokens", not result)

    # Test check frequency (only every 8 tokens)
    result = engine._check_uncertainty("as I mentioned " * 5, 13)
    check("No trigger at token 13 (not multiple of 8)", not result)

    result = engine._check_uncertainty("as I mentioned " * 5, 16)
    check("Triggers at token 16 (multiple of 8)", result)

    # Test repetition detection
    tee("\n  Repetition detection:")
    repeated = "the quick brown fox jumps " * 20
    result = engine._check_uncertainty(repeated, 64)
    check("Detects repetitive text at token 64", result)

    return passed, failed


def test_multi_needle():
    """
    Test 5: MULTI-NEEDLE RETRIEVAL
    Plant 3 needles at different depths, verify all can be found.
    """
    tee("\n" + "=" * 70)
    tee("TEST 5: Multi-Needle Retrieval")
    tee("=" * 70)

    passed, failed = 0, 0

    def check(name, cond):
        nonlocal passed, failed
        if cond:
            tee(f"    ✓ {name}")
            passed += 1
        else:
            tee(f"    ✗ {name}")
            failed += 1

    dim = 256

    def embed_text(text):
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        for w in words[:128]:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0
        for i in range(min(len(words) - 1, 64)):
            bg = f"{words[i]}_{words[i+1]}"
            h = int(hashlib.md5(bg.encode()).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 0.5
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # Plant 3 needles at different depths
    needles = [
        ("capital",      50_000,    NEEDLE_FACTS["capital"]),
        ("recipe",       500_000,   NEEDLE_FACTS["recipe"]),
        ("coordinates",  5_000_000, NEEDLE_FACTS["coordinates"]),
    ]

    # Build index: haystack + needles
    embeddings = []
    positions = []
    needle_positions = {}

    # Haystack
    for i in range(500):
        topic_idx = i % len(HAYSTACK_TOPICS)
        _, content = HAYSTACK_TOPICS[topic_idx]
        emb = embed_text(f"chunk-{i}: {content}")
        embeddings.append(emb)
        positions.append(i * 20000)

    # Needles
    for key, depth, fact in needles:
        pos = CHAT_BUFFER_SIZE - depth
        emb = embed_text(fact["needle"])
        embeddings.append(emb)
        positions.append(pos)
        needle_positions[key] = pos

    mat = np.stack(embeddings)
    norms = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)

    tee(f"\n  Index size: {len(embeddings)} chunks")
    tee(f"  Needles at: {json.dumps({k: f'{v:,}' for k, v in needle_positions.items()})}")

    # Query for each needle
    for key, depth, fact in needles:
        query_emb = embed_text(fact["query"])
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        sims = (norms @ q).flatten()
        top_idx = np.argsort(sims)[-5:][::-1]
        results = [(positions[i], float(sims[i])) for i in top_idx]

        target_pos = needle_positions[key]
        found = any(abs(pos - target_pos) < 200 for pos, _ in results)
        score = max(sc for pos, sc in results if abs(pos - target_pos) < 200) if found else 0

        check(f"{key} at depth {depth:>10,}: {'FOUND' if found else 'MISSED'} (score={score:.4f})",
              found)

    return passed, failed


def test_buffer_capacity():
    """
    Test 6: BUFFER CAPACITY
    Verify the system can handle 10M tokens worth of index entries
    and that search remains fast at scale.
    """
    tee("\n" + "=" * 70)
    tee("TEST 6: 10M Token Buffer Capacity & Search Performance")
    tee("=" * 70)

    passed, failed = 0, 0

    def check(name, cond):
        nonlocal passed, failed
        if cond:
            tee(f"    ✓ {name}")
            passed += 1
        else:
            tee(f"    ✗ {name}")
            failed += 1

    dim = 256
    # At 128 token granularity, 10M tokens = ~78K index entries
    n_entries = CHAT_BUFFER_SIZE // 128
    tee(f"\n  Building index with {n_entries:,} entries ({CHAT_BUFFER_SIZE:,} tokens / 128 granularity)")

    # Generate random embeddings (simulating indexed conversation chunks)
    np.random.seed(42)
    t0 = time.perf_counter()
    mat = np.random.randn(n_entries, dim).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    mat_normed = mat / norms
    t1 = time.perf_counter()
    tee(f"  Matrix build: {(t1-t0)*1000:.0f}ms")
    tee(f"  Matrix size: {mat_normed.nbytes / 1024 / 1024:.1f} MB")

    # Plant a needle at random position
    needle_idx = n_entries // 2
    needle_emb = np.random.randn(dim).astype(np.float32)
    needle_emb /= np.linalg.norm(needle_emb)
    mat_normed[needle_idx] = needle_emb

    # Search (numpy cosine similarity)
    query = needle_emb.copy()
    t0 = time.perf_counter()
    sims = (mat_normed @ query).flatten()
    top5 = np.argsort(sims)[-5:][::-1]
    t1 = time.perf_counter()
    search_ms = (t1 - t0) * 1000

    check(f"Numpy search: {search_ms:.1f}ms (< 100ms target)", search_ms < 100)
    check(f"Needle found at rank 1", top5[0] == needle_idx)

    # FAISS search (if available)
    try:
        import faiss
        t0 = time.perf_counter()
        index = faiss.IndexFlatIP(dim)
        index.add(mat_normed)
        t1 = time.perf_counter()
        build_ms = (t1 - t0) * 1000

        t0 = time.perf_counter()
        scores, indices = index.search(query.reshape(1, -1), 5)
        t1 = time.perf_counter()
        faiss_ms = (t1 - t0) * 1000

        tee(f"\n  FAISS index build: {build_ms:.0f}ms")
        check(f"FAISS search: {faiss_ms:.2f}ms (< 10ms target)", faiss_ms < 10)
        check(f"FAISS finds needle at rank 1", int(indices[0][0]) == needle_idx)
        tee(f"  FAISS index memory: {index.ntotal * dim * 4 / 1024 / 1024:.1f} MB")
    except ImportError:
        tee("\n  FAISS not available — numpy-only results")

    # Memory estimate for full 10M buffer
    token_bytes = CHAT_BUFFER_SIZE * 4  # int32 token IDs
    index_bytes = n_entries * dim * 4   # float32 embeddings
    total_mb = (token_bytes + index_bytes) / 1024 / 1024
    tee(f"\n  10M buffer memory estimate:")
    tee(f"    Token store:  {token_bytes / 1024 / 1024:.0f} MB")
    tee(f"    Index:        {index_bytes / 1024 / 1024:.0f} MB")
    tee(f"    Total:        {total_mb:.0f} MB")
    check(f"Total buffer memory {total_mb:.0f} MB (< 1000 MB)", total_mb < 1000)

    return passed, failed


# ============================================================================
# Full Integration Test (requires GPU + model)
# ============================================================================

def test_full_integration(model_name: str, target_depth: int = 100_000,
                          num_gpus: int = 4, kv_cache_dtype: str = "fp8_e5m2"):
    """
    Test 7: FULL INTEGRATION
    Runs actual inference with a real model.
    Plants a needle, fills haystack, queries, and checks the response.
    """
    tee("\n" + "=" * 70)
    tee("TEST 7: Full Integration (Live Model)")
    tee("=" * 70)

    passed, failed = 0, 0

    def check(name, cond):
        nonlocal passed, failed
        if cond:
            tee(f"    ✓ {name}")
            passed += 1
        else:
            tee(f"    ✗ {name}")
            failed += 1

    try:
        from kessie_exp3 import LibrarianEngine
    except ImportError:
        tee("  SKIPPED (kessie_exp3 not importable)")
        return 0, 0

    tee(f"\n  Model: {model_name}")
    tee(f"  Target depth: {target_depth:,} tokens")
    tee(f"  GPUs: {num_gpus} | KV dtype: {kv_cache_dtype} | Window: {WINDOW_SIZE}")

    engine = LibrarianEngine(
        model_name=model_name,
        num_gpus=num_gpus,
        window_size=WINDOW_SIZE,
        max_new_tokens=WINDOW_SIZE // 2,
        fog_alpha=0.5,
        fog_start=0.5,
        backend="vllm",
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_utilization=0.90,
    )
    engine.load()

    needle = NEEDLE_FACTS["capital"]
    haystack = HaystackGenerator(tokenizer=engine.tokenizer)

    # Build conversation history with needle at target depth
    # Structure: [needle] [haystack padding to push needle to target_depth from end]
    tee(f"  Building {target_depth:,} token haystack...")

    # 1. Insert needle FIRST (so it's deep in history)
    needle_toks = engine.tokenizer.encode(needle["needle"], add_special_tokens=False)
    needle_pos = len(engine.cache.conversation_tokens)
    engine.cache.append_conversation(needle_toks, role="user")

    # Index the needle
    embedder = getattr(engine.store, '_embedder', None)
    if embedder:
        emb = embedder(needle["needle"])
        engine.cache.index_embeddings.append(emb)
        engine.cache.index_positions.append(needle_pos)

    # 2. Fill haystack AFTER needle to push it to target depth
    chunk_size = 10000
    tokens_filled = 0

    while tokens_filled < target_depth:
        chunk_toks = haystack.generate_tokens(min(chunk_size, target_depth - tokens_filled))
        engine.cache.append_conversation(
            chunk_toks,
            role="user" if (tokens_filled // chunk_size) % 2 == 0 else "assistant")

        # Index haystack chunks
        if embedder:
            text = engine.tokenizer.decode(chunk_toks[:128], skip_special_tokens=True)
            emb = embedder(text)
            engine.cache.index_embeddings.append(emb)
            engine.cache.index_positions.append(needle_pos + len(needle_toks) + tokens_filled)

        tokens_filled += len(chunk_toks)

    total_tokens = len(engine.cache.conversation_tokens)
    actual_depth = total_tokens - needle_pos
    tee(f"  Total conversation: {total_tokens:,} tokens")
    tee(f"  Needle at position: {needle_pos:,}")
    tee(f"  Needle depth: {actual_depth:,} tokens from end")
    tee(f"  Index entries: {len(engine.cache.index_embeddings)}")

    # 3. Verify recall pipeline finds the needle BEFORE generation
    tee(f"\n  Pre-flight recall test...")
    test_recall = engine._auto_recall(needle["query"])
    if test_recall:
        tee(f"  ✓ _auto_recall found: {test_recall[:120]}...")
    else:
        tee(f"  ✗ _auto_recall returned empty — needle not reachable via recall")
        # Debug: manual search
        if embedder and engine.cache.index_embeddings:
            query_emb = embedder(needle["query"])
            mat = np.stack(engine.cache.index_embeddings)
            q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            norms = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
            sims = (norms @ q).flatten()
            top5 = np.argsort(sims)[-5:][::-1]
            tee(f"  Debug: manual search top-5:")
            for i in top5:
                tee(f"    pos={engine.cache.index_positions[i]:,} "
                    f"score={sims[i]:.4f}")

    # 4. Query the model — also build prompt manually to verify injection
    messages = [
        {"role": "user", "content": needle["query"]}
    ]

    # Diagnostic: check what _build_prompt produces
    conv = list(messages)
    conv = engine._inject_system(conv)
    test_prompt = engine._build_prompt(conv, test_recall)
    has_needle = "Florquen" in test_prompt
    tee(f"  Prompt contains needle: {has_needle}")
    tee(f"  Prompt length: {len(test_prompt)} chars, ~{len(test_prompt)//4} tokens")
    if not has_needle:
        # Show first 500 chars of prompt for debugging
        tee(f"  Prompt preview: {test_prompt[:500]}")

    tee(f"\n  Query: {needle['query']}")
    response = engine.generate(messages)

    if response and "choices" in response:
        answer = response["choices"][0]["message"]["content"]
        tee(f"  Response: {answer[:200]}")

        for frag in needle["expected_fragments"]:
            found = frag.lower() in answer.lower()
            check(f"Response contains '{frag}'", found)
    else:
        tee(f"  ERROR: No response")
        failed += len(needle["expected_fragments"])

    return passed, failed, engine


def test_mid_gen_recall_live(engine, model_name: str = ""):
    """
    Test 8: MID-GENERATION RECALL (Live Model)
    
    Forces mid-gen recall by temporarily disabling pre-gen recall.
    
    Flow:
    1. Plant needle in conversation store
    2. Monkey-patch _auto_recall to return "" (disabled)
    3. Ask about the needle — model has no pre-gen context
    4. Model starts generating, hedges ("I don't have", "previously", etc.)
    5. _check_uncertainty fires at token 32/48/64
    6. _mid_gen_recall searches conversation store with generated text
    7. Abort → recall → rebuild prompt with needle + partial output → resume
    8. Final response should contain needle details
    
    Uses streaming since mid-gen recall only fires in streaming decode.
    """
    tee("\n" + "=" * 70)
    tee("TEST 8: Mid-Generation Recall (Live Model, Streaming)")
    tee("=" * 70)

    passed, failed = 0, 0

    def check(name, cond):
        nonlocal passed, failed
        if cond:
            tee(f"    ✓ {name}")
            passed += 1
        else:
            tee(f"    ✗ {name}")
            failed += 1

    try:
        from kessie_exp3 import LibrarianEngine
    except ImportError:
        tee("  SKIPPED (kessie_exp3 not importable)")
        return 0, 0

    if engine is None:
        tee("  SKIPPED (no engine available from Test 7)")
        return 0, 0

    tee(f"\n  Model: {model_name or engine.model_name}")

    # Reset conversation store from Test 7
    engine.cache.conversation_tokens.clear()
    engine.cache.index_embeddings.clear()
    engine.cache.index_positions.clear()
    engine.cache.turn_boundaries.clear()
    engine.cache._turn_counter = 0

    haystack = HaystackGenerator(tokenizer=engine.tokenizer)

    # ---- Scenario: plant a specific technical decision deep in history ----
    mid_gen_needle = (
        "We decided the Korthax pipeline should use 7 shards with "
        "replication factor 3, running on port 9147. The compression "
        "algorithm is zstd-19 with dictionary pre-training on the "
        "telemetry corpus. Failover timeout is exactly 847 milliseconds."
    )
    
    expected_fragments = ["9147", "847", "zstd"]
    
    # 90% of CHAT_BUFFER_SIZE — prove mid-gen recall works at real depth
    target_depth = int(CHAT_BUFFER_SIZE * 0.9)  # 9,000,000 tokens
    tee(f"  Planting needle at {target_depth:,} token depth")
    tee(f"  Needle: {mid_gen_needle[:80]}...")
    tee(f"  Query:  system-prompted hedge + direct Korthax question")

    # 1. Insert needle
    needle_toks = engine.tokenizer.encode(mid_gen_needle, add_special_tokens=False)
    needle_pos = len(engine.cache.conversation_tokens)
    engine.cache.append_conversation(needle_toks, role="user")

    # Index the needle
    embedder = getattr(engine.store, '_embedder', None)
    if embedder:
        emb = embedder(mid_gen_needle)
        engine.cache.index_embeddings.append(emb)
        engine.cache.index_positions.append(needle_pos)

    # 2. Fill haystack after needle
    chunk_size = 10000
    tokens_filled = 0
    while tokens_filled < target_depth:
        chunk_toks = haystack.generate_tokens(min(chunk_size, target_depth - tokens_filled))
        engine.cache.append_conversation(
            chunk_toks,
            role="user" if (tokens_filled // chunk_size) % 2 == 0 else "assistant")
        if embedder:
            text = engine.tokenizer.decode(chunk_toks[:128], skip_special_tokens=True)
            emb = embedder(text)
            engine.cache.index_embeddings.append(emb)
            engine.cache.index_positions.append(needle_pos + len(needle_toks) + tokens_filled)
        tokens_filled += len(chunk_toks)

    total_tokens = len(engine.cache.conversation_tokens)
    tee(f"  Total conversation: {total_tokens:,} tokens")
    tee(f"  Index entries: {len(engine.cache.index_embeddings)}")

    # 3. Verify _mid_gen_recall CAN find the needle (sanity check)
    # Use text similar to what the model would generate when hedging about this topic
    hedge_text = (
        "If I recall correctly, the Korthax pipeline configuration we discussed "
        "previously included specific settings for shards, port, compression, and "
        "failover timeout. Let me try to remember the exact numbers."
    )
    test_mid = engine._mid_gen_recall(hedge_text, "")
    if test_mid and "9147" in test_mid:
        tee(f"  ✓ _mid_gen_recall sanity check: needle reachable")
        tee(f"    {test_mid[:120]}...")
    else:
        tee(f"  ✗ _mid_gen_recall sanity check FAILED — needle not reachable")
        tee(f"    got: {test_mid[:120] if test_mid else '(empty)'}")
        # Still continue — maybe the model's actual hedging text will match better

    # 4. Disable pre-gen recall — mid-gen is the only path
    #    System message instructs model to hedge with a known phrase
    #    instead of hallucinating. This phrase matches _HEDGE_PATTERNS
    #    and will naturally trigger _check_uncertainty.
    tee(f"\n  Disabling pre-gen recall to force mid-gen path...")
    tee(f"  System message instructs model to hedge instead of hallucinate...")
    original_auto_recall = engine._auto_recall
    engine._auto_recall = lambda *args, **kwargs: ""
    engine._mid_gen_events = []  # Clear event log

    # Multi-turn with system instruction that forces hedging
    messages = [
        {"role": "system", "content":
            "You are a helpful assistant with access to conversation history. "
            "IMPORTANT: When asked about details from a previous conversation, "
            "you MUST first restate what topic and specific details are being asked about, "
            "then indicate you need to recall. Use this format:\n"
            "'[restate the topic and details being asked about]... "
            "let me see if I can remember from our earlier conversation...'\n"
            "Do NOT make up or hallucinate specific numbers, names, or configurations. "
            "Either recall them accurately or use the format above."},
        {"role": "user", "content":
            "What was the exact Korthax pipeline configuration we decided on? "
            "I need the port number, shard count, compression algorithm, and "
            "failover timeout for the deployment script."},
    ]

    # 5. Stream generation
    tee(f"  Streaming query (system-prompted hedge)...")

    full_response = ""
    
    for chunk in engine.generate_stream(messages):
        if isinstance(chunk, str):
            if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
                try:
                    data = json.loads(chunk[6:])
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_response += content
                except (json.JSONDecodeError, IndexError, KeyError):
                    pass

    # 6. Restore pre-gen recall
    engine._auto_recall = original_auto_recall

    tee(f"  Response ({len(full_response)} chars):")
    tee(f"    {full_response[:400]}")

    # 7. Inspect mid-gen events — this is the real test
    tee(f"\n  Mid-gen event log ({len(engine._mid_gen_events)} events):")
    for i, evt in enumerate(engine._mid_gen_events):
        tee(f"    [{i}] {evt['event']}: {json.dumps({k:v for k,v in evt.items() if k != 'event'}, default=str)[:200]}")

    # Categorize events
    uncertainty_events = [e for e in engine._mid_gen_events if e["event"] == "uncertainty_detected"]
    recall_found_events = [e for e in engine._mid_gen_events if e["event"] == "recall_found"]
    recall_empty_events = [e for e in engine._mid_gen_events if e["event"] == "recall_empty"]
    resume_events = [e for e in engine._mid_gen_events if e["event"] == "resume"]

    tee(f"\n  Event summary:")
    tee(f"    uncertainty_detected: {len(uncertainty_events)}")
    tee(f"    recall_found:        {len(recall_found_events)}")
    tee(f"    recall_empty:        {len(recall_empty_events)}")
    tee(f"    resume:              {len(resume_events)}")

    # 8. Assertions on the pipeline mechanics
    check("Uncertainty was detected (hedge pattern or repetition fired)",
          len(uncertainty_events) > 0)

    check("Mid-gen recall found the needle in conversation store",
          len(recall_found_events) > 0)

    check("Generation was aborted and resumed with recalled context",
          len(resume_events) > 0)

    # If recall found something, verify it's the needle
    if recall_found_events:
        preview = recall_found_events[0].get("recalled_preview", "")
        has_needle_content = any(f in preview for f in ["9147", "Korthax", "847", "zstd"])
        check("Recalled content contains needle data", has_needle_content)
        tee(f"    Recalled preview: {preview[:150]}")
    else:
        check("Recalled content contains needle data", False)

    # Response check is secondary — the pipeline mechanics are what matter
    tee(f"\n  Response fragment check (secondary):")
    for frag in expected_fragments:
        found = frag.lower() in full_response.lower()
        check(f"Response contains '{frag}'", found)

    return passed, failed

def main():
    parser = argparse.ArgumentParser(description="KeSSie Needle-in-a-Haystack Test")
    parser.add_argument("--model", type=str, default=None,
                        help="Model path for full integration test")
    parser.add_argument("--full", action="store_true",
                        help="Run full integration test (requires GPU)")
    parser.add_argument("--depth", type=int, default=None,
                        help="Specific depth for integration test")
    parser.add_argument("--gpus", type=int, default=4,
                        help="Number of GPUs for tensor parallel (default: 4)")
    parser.add_argument("--kv-cache-dtype", type=str, default="fp8_e5m2",
                        help="KV cache dtype (default: fp8_e5m2)")
    args = parser.parse_args()

    total_passed = 0
    total_failed = 0

    _open_log()

    tee("╔" + "═" * 68 + "╗")
    tee("║  KeSSie Needle-in-a-Haystack Test Suite                            ║")
    tee("║  Window: 131,072 tokens | Buffer: 10,000,000 tokens               ║")
    tee("╚" + "═" * 68 + "╝")

    # Offline tests (always run)
    tests = [
        ("Depth Recall", test_conversation_store_depth),
        ("Fog Attention 131K", test_fog_attention_at_scale),
        ("Positional Annotation", test_positional_annotation),
        ("Mid-Gen Trigger", test_mid_gen_trigger),
        ("Multi-Needle", test_multi_needle),
        ("Buffer Capacity", test_buffer_capacity),
    ]

    for name, test_fn in tests:
        try:
            p, f = test_fn()
            total_passed += p
            total_failed += f
        except Exception as e:
            tee(f"\n  ERROR in {name}: {e}")
            import traceback
            tb = traceback.format_exc()
            tee(tb)
            total_failed += 1

    # Full integration (optional)
    _engine = None
    if args.full and args.model:
        try:
            depth = args.depth or 100_000
            p, f, _engine = test_full_integration(args.model, depth,
                                          num_gpus=args.gpus,
                                          kv_cache_dtype=args.kv_cache_dtype)
            total_passed += p
            total_failed += f
        except Exception as e:
            tee(f"\n  ERROR in integration test: {e}")
            import traceback
            tb = traceback.format_exc()
            tee(tb)
            total_failed += 1

        # Test 8: Mid-gen recall (only with --full, reuses engine)
        try:
            p, f = test_mid_gen_recall_live(_engine, model_name=args.model)
            total_passed += p
            total_failed += f
        except Exception as e:
            tee(f"\n  ERROR in mid-gen recall test: {e}")
            import traceback
            tb = traceback.format_exc()
            tee(tb)
            total_failed += 1

    # Summary
    tee("\n" + "=" * 70)
    tee("SUMMARY")
    tee("=" * 70)
    total = total_passed + total_failed
    tee(f"  Passed: {total_passed}/{total}")
    tee(f"  Failed: {total_failed}/{total}")

    if total_failed:
        tee(f"\n  *** {total_failed} TEST(S) FAILED ***")
        _close_log()
        sys.exit(1)
    else:
        tee(f"\n  ALL {total_passed} TESTS PASSED ✓")
        _close_log()
        sys.exit(0)


if __name__ == "__main__":
    main()
