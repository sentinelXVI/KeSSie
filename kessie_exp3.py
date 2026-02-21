#!/usr/bin/env python3
"""
KeSSie Experiment 3  -  KeSSie Engine
=====================================
PROPRIETARY & CONFIDENTIAL

Architecture:
  Dual backend: HuggingFace transformers or vLLM
  - HF:   Direct model access, KeSSie-managed KV (GPU<->CPU tiered), fog-of-war attention
  - vLLM: PagedAttention KV (vLLM managed), prefix caching, fp8/MoE/VL native support
  Common: KeSSie semantic memory, auto-recall, knowledge store, tool system

  KeSSie Memory Layer (both backends):
  - Conversation token tracking with semantic indexing
  - Auto-recall: embed query -> search conversation index -> inject relevant context
  - Knowledge store: SQLite + FAISS vector search, store_learned/retrieve_learned tools
  - Fog-of-war: HF=KV mask, vLLM=prompt-level context management

  vLLM Features:
  - AsyncLLMEngine for true token-by-token streaming
  - Automatic tensor parallel, fp8 quantized models, MoE, VL models
  - enable_prefix_caching for KV reuse across requests
  - SamplingParams: temperature, top_p, top_k, presence_penalty, repetition_penalty

Usage:
  # vLLM backend (recommended for large/quantized models):
  python kessie_exp3.py serve --backend vllm --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 --gpus 4 --port 5999

  # HF backend (full KeSSie KV management):
  python kessie_exp3.py serve --backend hf --model Qwen/Qwen3-0.6B --gpus 4 --seed --port 8200

  python kessie_exp3.py chat --backend vllm --model Qwen/Qwen3-0.6B --gpus 1
  python kessie_exp3.py store seed
"""

import os, sys, json, time, uuid, re, sqlite3, math
import logging, argparse, threading, traceback, queue
import numpy as np
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple, Generator
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

logger = logging.getLogger("KeSSie.Exp3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.info("FAISS not found  -  using numpy cosine fallback")


# =============================================================================
# Vector Index  -  FAISS CPU or numpy fallback
# =============================================================================

class VectorIndex:
    """CPU-bound cosine similarity. FAISS IndexFlatIP if available."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self._lock = threading.RLock()
        self._vectors: List[np.ndarray] = []
        self._ids: List[int] = []
        self._faiss_index = None
        self._np_mat = None

    def rebuild(self, vectors: List[np.ndarray], ids: List[int]):
        if not vectors:
            with self._lock:
                self._vectors, self._ids = [], []
                self._faiss_index = self._np_mat = None
            return
        mat = np.stack([v.astype(np.float32) for v in vectors])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        mat_normed = mat / norms
        fi = None
        if HAS_FAISS:
            fi = faiss.IndexFlatIP(self.dim)
            fi.add(mat_normed)
        with self._lock:
            self._vectors = [v.astype(np.float32) for v in vectors]
            self._ids = list(ids)
            self._faiss_index = fi
            self._np_mat = mat_normed

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        q = query.astype(np.float32).reshape(1, -1)
        qn = np.linalg.norm(q)
        if qn < 1e-8: return []
        q = q / qn
        with self._lock:
            if not self._ids: return []
            k = min(top_k, len(self._ids))
            if self._faiss_index is not None:
                scores, indices = self._faiss_index.search(q, k)
                return [(self._ids[int(indices[0][i])], float(scores[0][i]))
                        for i in range(k) if 0 <= int(indices[0][i]) < len(self._ids)]
            elif self._np_mat is not None:
                sims = (self._np_mat @ q.T).flatten()
                top_idx = np.argsort(sims)[-k:][::-1]
                return [(self._ids[i], float(sims[i])) for i in top_idx]
        return []

    @property
    def count(self):
        with self._lock:
            return len(self._ids)


# =============================================================================
# FAISS Async Indexing Queue
# =============================================================================

class IndexingQueue:
    """
    Async queue for FAISS index operations.
    Embed + rebuild runs in background thread  -  never blocks callers.
    Coalesces rapid-fire inserts into single rebuild.
    """

    def __init__(self, store, coalesce_ms: float = 100):
        self._store = store
        self._coalesce_ms = coalesce_ms
        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True, name="faiss-indexer")
        self._worker.start()

    def schedule_reindex(self, entry_id: int = None):
        """Schedule a FAISS rebuild. entry_id is informational."""
        self._queue.put(("reindex", entry_id, time.monotonic()))

    def _run(self):
        while True:
            try:
                item = self._queue.get(timeout=5)
            except queue.Empty:
                continue
            # Coalesce: drain all pending reindex requests
            pending = [item]
            deadline = time.monotonic() + self._coalesce_ms / 1000.0
            while time.monotonic() < deadline:
                try:
                    pending.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            # Single rebuild for all coalesced requests
            t0 = time.perf_counter()
            try:
                self._store._do_rebuild_index()
                ms = (time.perf_counter() - t0) * 1000
                logger.info(f"FAISS rebuild: {self._store.vindex.count} vectors, "
                            f"{len(pending)} coalesced, {ms:.1f}ms")
            except Exception as e:
                logger.error(f"FAISS rebuild failed: {e}")

    @property
    def pending(self):
        return self._queue.qsize()


# =============================================================================
# Knowledge Store  -  SQLite + async FAISS
# =============================================================================

class KnowledgeStore:
    """SQLite for persistence, FAISS for vector search. Index rebuilt async."""

    def __init__(self, db_path: str = "kessie_knowledge.db", embed_dim: int = 256):
        self.db_path = db_path
        self.embed_dim = embed_dim
        self._lock = threading.Lock()
        self._embedder = None
        self.vindex = VectorIndex(dim=embed_dim)
        self._init_db()
        self._do_rebuild_index()  # sync on startup
        self.indexer = IndexingQueue(self)
        logger.info(f"KnowledgeStore: {db_path} ({self.count()} entries, {self.vindex.count} vectors)")

    def set_embedder(self, embedder_fn):
        self._embedder = embedder_fn

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                source TEXT DEFAULT '', confidence REAL DEFAULT 1.0,
                version INTEGER DEFAULT 1, embedding BLOB,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                active INTEGER DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_kt ON knowledge(topic);
            CREATE INDEX IF NOT EXISTS idx_ktk ON knowledge(topic, key);
        """)
        conn.close()

    def _conn(self):
        c = sqlite3.connect(self.db_path); c.row_factory = sqlite3.Row; return c

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self._embedder is None: return None
        try: return self._embedder(text)
        except: return None

    def _do_rebuild_index(self):
        """Actually rebuild FAISS from SQLite. Called by IndexingQueue worker."""
        conn = self._conn()
        rows = conn.execute("SELECT id, embedding FROM knowledge WHERE active=1 AND embedding IS NOT NULL").fetchall()
        conn.close()
        vectors, ids = [], []
        for r in rows:
            blob = r["embedding"]
            if blob and len(blob) == self.embed_dim * 4:
                vectors.append(np.frombuffer(blob, dtype=np.float32).copy())
                ids.append(r["id"])
        self.vindex.rebuild(vectors, ids)

    def store(self, topic, key, value, source="", confidence=1.0):
        now = datetime.now(timezone.utc).isoformat()
        vec = self._embed(f"{topic} {key} {value}")
        blob = vec.tobytes() if vec is not None else None
        with self._lock:
            conn = self._conn()
            ex = conn.execute("SELECT id,version FROM knowledge WHERE topic=? AND key=? AND active=1", (topic,key)).fetchone()
            ver = (ex["version"] + 1) if ex else 1
            if ex:
                conn.execute("UPDATE knowledge SET active=0,updated_at=? WHERE id=?", (now, ex["id"]))
            conn.execute(
                "INSERT INTO knowledge(topic,key,value,source,confidence,version,embedding,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (topic, key, value, source, confidence, ver, blob, now, now))
            conn.commit()
            eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.close()
        # Async FAISS rebuild  -  does NOT block caller
        self.indexer.schedule_reindex(eid)
        return {"id":eid,"topic":topic,"key":key,"value":value,"version":ver,"status":"stored"}

    def retrieve(self, query, topic=None, limit=5):
        results = []
        # Vector search
        qvec = self._embed(query)
        if qvec is not None and self.vindex.count > 0:
            hits = self.vindex.search(qvec, top_k=limit * 2)
            if hits:
                hit_ids = [h[0] for h in hits]
                ph = ",".join("?" * len(hit_ids))
                with self._lock:
                    conn = self._conn()
                    if topic:
                        rows = conn.execute(f"SELECT * FROM knowledge WHERE id IN ({ph}) AND active=1 AND topic=?", hit_ids+[topic]).fetchall()
                    else:
                        rows = conn.execute(f"SELECT * FROM knowledge WHERE id IN ({ph}) AND active=1", hit_ids).fetchall()
                    conn.close()
                row_map = {r["id"]: dict(r) for r in rows}
                for eid, score in hits:
                    if eid in row_map:
                        e = row_map[eid]; e["_score"] = score; e.pop("embedding", None)
                        results.append(e)
                        if len(results) >= limit: break
        # LIKE fallback
        if not results:
            words = [w.strip() for w in query.split() if len(w.strip()) > 1]
            stems = set()
            for w in words:
                stems.add(w)
                if w.endswith('s') and len(w) > 3: stems.add(w[:-1])
            terms = list(stems)[:6]
            if terms:
                conds, params = [], []
                for t in terms:
                    lq = f"%{t}%"
                    conds.append("(topic LIKE ? OR key LIKE ? OR value LIKE ?)")
                    params.extend([lq,lq,lq])
                w = " OR ".join(conds)
                with self._lock:
                    conn = self._conn()
                    if topic:
                        rows = conn.execute(f"SELECT * FROM knowledge WHERE active=1 AND topic=? AND ({w}) LIMIT ?", [topic]+params+[limit]).fetchall()
                    else:
                        rows = conn.execute(f"SELECT * FROM knowledge WHERE active=1 AND ({w}) LIMIT ?", params+[limit]).fetchall()
                    conn.close()
                results = [dict(r) for r in rows]
                for r in results: r.pop("embedding", None)
        return results

    def list_topics(self):
        c = self._conn()
        r = c.execute("SELECT topic,COUNT(*) as count FROM knowledge WHERE active=1 GROUP BY topic ORDER BY topic").fetchall()
        c.close(); return [dict(x) for x in r]

    def count(self):
        c = self._conn(); n = c.execute("SELECT COUNT(*) FROM knowledge WHERE active=1").fetchone()[0]; c.close(); return n

    def reembed_all(self):
        if self._embedder is None: return 0
        conn = self._conn()
        rows = conn.execute("SELECT id,topic,key,value FROM knowledge WHERE active=1").fetchall()
        n = 0
        for r in rows:
            vec = self._embed(f"{r['topic']} {r['key']} {r['value']}")
            if vec is not None:
                conn.execute("UPDATE knowledge SET embedding=? WHERE id=?", (vec.tobytes(), r["id"]))
                n += 1
        conn.commit(); conn.close()
        self._do_rebuild_index()
        logger.info(f"Re-embedded {n} entries")
        return n


# =============================================================================
# Tool Definitions + Executor
# =============================================================================

TOOLS_SCHEMA = []

class ToolExecutor:
    def __init__(self, store: KnowledgeStore):
        self.store = store
    def execute(self, name, args):
        try:
            if name == "store_learned":
                return json.dumps(self.store.store(args.get("topic","general"),args.get("key","unnamed"),
                    args.get("value",""),args.get("source",""),args.get("confidence",1.0)), ensure_ascii=False)
            elif name == "retrieve_learned":
                r = self.store.retrieve(args.get("query",""), args.get("topic"), 5)
                return json.dumps({"results":r,"count":len(r)}, ensure_ascii=False, default=str)
            return json.dumps({"error":f"Unknown tool: {name}"})
        except Exception as e:
            return json.dumps({"error":str(e)})


# =============================================================================
# KeSSie Cache
# =============================================================================

class KeSSieCache:
    """
    Tiered KV cache:
      GPU VRAM: hot window (clear zone)  -  active attention
      CPU RAM:  fogged/evicted KV  -  recalled on hit
      GPU:      conversation index embeddings (fast search)

    On evict: fogged KV moves to CPU RAM (not destroyed)
    On recall hit: CPU chunk loaded back to GPU, spliced into window
    """

    def __init__(self, window_size=4096, index_granularity=128, fog_alpha=0.5, fog_start=0.5,
                 kv_cache_dtype=None, max_conversation_tokens=10_000_000):
        self.window_size = window_size
        self.fog_alpha = fog_alpha; self.fog_start = fog_start
        self.index_granularity = index_granularity
        self.kv_cache_dtype = kv_cache_dtype
        self.max_conversation_tokens = max_conversation_tokens
        self.conversation_tokens: List[int] = []
        self.index_embeddings: List[np.ndarray] = []
        self.index_positions: List[int] = []

        # GPU-resident: hot window KV
        self.key_cache = []; self.value_cache = []

        # CPU-resident: evicted fogged KV chunks
        # List of (start_pos, end_pos, [(key_tensor_cpu, value_tensor_cpu), ...per layer])
        self.cpu_kv_store: List[Tuple[int, int, List[Tuple[Any, Any]]]] = []
        self.cpu_kv_bytes = 0

        self.window_start = 0; self.seen_tokens = 0; self._seen_tokens = 0
        self.total_evictions = 0; self.total_recalls = 0
        self._proj_matrix = None
        self.has_previous_state = False
        self.ssm_states = {}; self.conv_states = {}

        # Turn boundary tracking: list of (token_start_pos, role, turn_number)
        # Enables positional annotation when recalling fogged context
        self.turn_boundaries: List[Tuple[int, str, int]] = []
        self._turn_counter = 0

    def _cast_kv(self, t):
        if self.kv_cache_dtype is not None and t.dtype != self.kv_cache_dtype:
            return t.to(self.kv_cache_dtype)
        return t

    def __len__(self): return len(self.key_cache)
    def __getitem__(self, i): return (self.key_cache[i], self.value_cache[i])
    def get_seq_length(self, li=0):
        if li < len(self.key_cache) and self._valid(self.key_cache[li]):
            return self.key_cache[li].shape[-2]
        return 0
    def get_max_length(self): return self.window_size * 2
    def get_usable_length(self, n, li=0): return max(0, self.get_seq_length(li))
    def get_mask_sizes(self, cp, li): return self.get_seq_length(li)+cp.shape[0], 0
    def reset(self):
        self.key_cache.clear(); self.value_cache.clear()
        self.cpu_kv_store.clear(); self.cpu_kv_bytes = 0
        self.window_start=0; self.seen_tokens=0; self._seen_tokens=0
        self.has_previous_state=False; self.ssm_states.clear(); self.conv_states.clear()

    def full_reset(self):
        """Full eviction: clear everything including conversation store and index."""
        self.reset()
        self.conversation_tokens.clear()
        self.index_embeddings.clear()
        self.index_positions.clear()
        self.turn_boundaries.clear()
        self._turn_counter = 0
        self.total_evictions = 0
        self.total_recalls = 0
    @staticmethod
    def _valid(t):
        import torch
        return t is not None and isinstance(t, torch.Tensor) and t.numel() > 0

    def _compute_fog_mask(self, seq_len, device, dtype):
        import torch
        if self.fog_alpha <= 0 or seq_len == 0: return None
        cc = max(1, int(seq_len * self.fog_start))
        fc = seq_len - cc
        if fc <= 0: return None
        pos = torch.arange(fc, dtype=dtype, device=device)
        fw = (pos / max(fc-1, 1)).pow(self.fog_alpha).clamp(min=0.01)
        cw = torch.ones(cc, dtype=dtype, device=device)
        return torch.cat([fw, cw]).view(1, 1, seq_len, 1)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        import torch
        key_states = self._cast_kv(key_states)
        value_states = self._cast_kv(value_states)
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.tensor([], device=key_states.device))
            self.value_cache.append(torch.tensor([], device=key_states.device))
        if not self._valid(self.key_cache[layer_idx]):
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        if layer_idx == 0:
            self.seen_tokens = self.window_start + self.key_cache[0].shape[-2]
            self._seen_tokens = self.seen_tokens
        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]
        fog = self._compute_fog_mask(v.shape[-2], v.device, torch.float32)
        if fog is not None:
            v = (v.float() * fog).to(k.dtype)
        return k, v

    def evict_if_needed(self):
        """Evict fogged KV to CPU RAM. Window stays on GPU."""
        if not self.key_cache or not self._valid(self.key_cache[0]): return
        cl = self.key_cache[0].shape[-2]
        ow = self.window_size * 2
        if cl <= ow: return
        ev = cl - self.window_size
        if ev <= 0: return

        # Move evicted KV to CPU RAM
        cpu_layers = []
        chunk_bytes = 0
        for i in range(len(self.key_cache)):
            if self._valid(self.key_cache[i]):
                k_evicted = self.key_cache[i][:, :, :ev, :].to("cpu", non_blocking=True)
                v_evicted = self.value_cache[i][:, :, :ev, :].to("cpu", non_blocking=True)
                chunk_bytes += k_evicted.nelement() * k_evicted.element_size()
                chunk_bytes += v_evicted.nelement() * v_evicted.element_size()
                cpu_layers.append((k_evicted, v_evicted))
                # Trim GPU cache
                self.key_cache[i] = self.key_cache[i][:, :, ev:, :].contiguous()
                self.value_cache[i] = self.value_cache[i][:, :, ev:, :].contiguous()
            else:
                cpu_layers.append((None, None))

        start_pos = self.window_start
        end_pos = self.window_start + ev
        self.cpu_kv_store.append((start_pos, end_pos, cpu_layers))
        self.cpu_kv_bytes += chunk_bytes

        self.window_start += ev; self.total_evictions += ev
        self.seen_tokens = self.window_start + self.key_cache[0].shape[-2]
        self._seen_tokens = self.seen_tokens
        logger.info(f"KV evict: {ev} tokens -> CPU ({chunk_bytes/1048576:.1f} MB), "
                    f"{len(self.cpu_kv_store)} chunks in RAM ({self.cpu_kv_bytes/1048576:.1f} MB total)")

    def recall_from_cpu(self, position: int, device) -> bool:
        """
        Recall a CPU-stored KV chunk back to GPU and prepend to window.
        Returns True if chunk was found and loaded.
        """
        import torch
        target_chunk = None
        target_idx = None
        for idx, (start, end, layers) in enumerate(self.cpu_kv_store):
            if start <= position < end:
                target_chunk = (start, end, layers)
                target_idx = idx
                break
        if target_chunk is None:
            return False

        start, end, cpu_layers = target_chunk
        chunk_len = end - start
        recalled_bytes = 0

        # Prepend recalled KV to GPU window
        for i in range(len(self.key_cache)):
            if i < len(cpu_layers) and cpu_layers[i][0] is not None:
                k_recalled = cpu_layers[i][0].to(device, non_blocking=True)
                v_recalled = cpu_layers[i][1].to(device, non_blocking=True)
                recalled_bytes += k_recalled.nelement() * k_recalled.element_size() * 2
                if self._valid(self.key_cache[i]):
                    self.key_cache[i] = torch.cat([k_recalled, self.key_cache[i]], dim=-2)
                    self.value_cache[i] = torch.cat([v_recalled, self.value_cache[i]], dim=-2)
                else:
                    self.key_cache[i] = k_recalled
                    self.value_cache[i] = v_recalled

        # Update window tracking  -  window now starts earlier
        self.window_start = start
        self.seen_tokens = self.window_start + self.key_cache[0].shape[-2]
        self._seen_tokens = self.seen_tokens
        self.total_recalls += 1

        # Remove from CPU store
        del self.cpu_kv_store[target_idx]
        self.cpu_kv_bytes -= recalled_bytes // 2  # approximate

        logger.info(f"KV recall: pos {start}-{end} ({chunk_len} tokens) -> GPU "
                    f"({recalled_bytes/1048576:.1f} MB), window now {self.get_seq_length()} tokens")
        return True

    def append_conversation(self, tids, role="assistant"):
        """Append token IDs and record turn boundary. Enforces max_conversation_tokens."""
        # Evict oldest tokens if we'd exceed the cap
        overflow = (len(self.conversation_tokens) + len(tids)) - self.max_conversation_tokens
        if overflow > 0:
            self.conversation_tokens = self.conversation_tokens[overflow:]
            # Shift index positions and prune entries that fell off
            new_embeddings = []
            new_positions = []
            for emb, pos in zip(self.index_embeddings, self.index_positions):
                new_pos = pos - overflow
                if new_pos >= 0:
                    new_embeddings.append(emb)
                    new_positions.append(new_pos)
            self.index_embeddings = new_embeddings
            self.index_positions = new_positions
            # Shift turn boundaries
            new_boundaries = []
            for start_pos, r, turn_num in self.turn_boundaries:
                new_start = start_pos - overflow
                if new_start >= 0:
                    new_boundaries.append((new_start, r, turn_num))
            self.turn_boundaries = new_boundaries

        pos = len(self.conversation_tokens)
        self.conversation_tokens.extend(tids)
        self._turn_counter += 1
        self.turn_boundaries.append((pos, role, self._turn_counter))

    def get_turn_for_position(self, token_pos: int):
        """
        Find the turn (role, turn_number, total_turns) for a given token position.
        Returns (role, turn_number, total_turns) or ("unknown", 0, total_turns).
        """
        total = self._turn_counter
        # Binary search would be better but turns are typically <1000
        matched_role, matched_turn = "unknown", 0
        for start_pos, role, turn_num in self.turn_boundaries:
            if start_pos <= token_pos:
                matched_role = role
                matched_turn = turn_num
            else:
                break
        return matched_role, matched_turn, total

    def get_token_distance(self, token_pos: int) -> int:
        """How many tokens ago was this position relative to current end."""
        return len(self.conversation_tokens) - token_pos

    def _get_projection(self, ed, target=256):
        if self._proj_matrix is None or self._proj_matrix.shape[0] != ed:
            np.random.seed(42)
            self._proj_matrix = np.random.randn(ed, target).astype(np.float32) / np.sqrt(ed)
        return self._proj_matrix

    def get_stats(self):
        kv_gpu_bytes = 0
        kv_dtype = None
        kv_layers = len(self.key_cache)
        kv_seq = self.get_seq_length()
        try:
            for i in range(len(self.key_cache)):
                if self._valid(self.key_cache[i]):
                    kv_gpu_bytes += self.key_cache[i].nelement() * self.key_cache[i].element_size()
                    kv_gpu_bytes += self.value_cache[i].nelement() * self.value_cache[i].element_size()
                    if kv_dtype is None:
                        kv_dtype = str(self.key_cache[i].dtype)
        except:
            pass
        fog_tokens = int(kv_seq * (1.0 - self.fog_start)) if kv_seq > 0 else 0
        clear_tokens = kv_seq - fog_tokens
        # CPU store stats
        cpu_chunks = len(self.cpu_kv_store)
        cpu_tokens = sum(end - start for start, end, _ in self.cpu_kv_store)
        return {
            "conversation_tokens": len(self.conversation_tokens),
            "semantic_index_count": len(self.index_embeddings),
            "kv_seq_len": kv_seq,
            "kv_window_start": self.window_start,
            "kv_layers": kv_layers,
            "kv_gpu_bytes": kv_gpu_bytes,
            "kv_gpu_mb": round(kv_gpu_bytes / 1048576, 2),
            "kv_cpu_mb": round(self.cpu_kv_bytes / 1048576, 2),
            "kv_cpu_chunks": cpu_chunks,
            "kv_cpu_tokens": cpu_tokens,
            "kv_dtype": kv_dtype,
            "fog_tokens": fog_tokens,
            "clear_tokens": clear_tokens,
            "window_capacity": self.window_size,
            "window_fill_pct": round(kv_seq / max(self.window_size, 1) * 100, 1),
            "total_evictions": self.total_evictions,
            "total_recalls": self.total_recalls,
        }


class SessionStats:
    """Per-request generation statistics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.t_start = time.perf_counter()
        self.t_prefill_start = 0
        self.t_prefill_end = 0
        self.t_decode_start = 0
        self.t_decode_end = 0
        self.prompt_tokens = 0
        self.generated_tokens = 0
        self.tool_rounds = 0
        self.tool_calls = []
        self.kv_stats_pre = {}
        self.kv_stats_post = {}
        self.peak_gpu_mb = 0
        self.rope_positions = 0
        self.truncated = False
        self.mid_gen_recalls = 0

    @property
    def prefill_ms(self):
        if self.t_prefill_end and self.t_prefill_start:
            return (self.t_prefill_end - self.t_prefill_start) * 1000
        return 0

    @property
    def decode_ms(self):
        if self.t_decode_end and self.t_decode_start:
            return (self.t_decode_end - self.t_decode_start) * 1000
        return 0

    @property
    def total_ms(self):
        return (time.perf_counter() - self.t_start) * 1000

    @property
    def tokens_per_sec(self):
        d = self.decode_ms / 1000
        return self.generated_tokens / d if d > 0 else 0

    @property
    def prefill_tokens_per_sec(self):
        p = self.prefill_ms / 1000
        return self.prompt_tokens / p if p > 0 else 0

    def log(self, cache_stats=None, batch_stats=None, backend="hf", vllm_engine=None):
        """Log comprehensive session debug info."""
        lines = []
        lines.append(f"-- Session Debug ------------------------------")
        lines.append(f"  Prompt:     {self.prompt_tokens} tokens"
                     f"{'  [TRUNCATED to window]' if self.truncated else ''}")
        lines.append(f"  Generated:  {self.generated_tokens} tokens")
        lines.append(f"  Prefill:    {self.prefill_ms:.0f}ms"
                     f" ({self.prefill_tokens_per_sec:.0f} tok/s)" if self.prefill_ms else "  Prefill:    (fast path)")
        lines.append(f"  Decode:     {self.decode_ms:.0f}ms"
                     f" ({self.tokens_per_sec:.1f} tok/s)")
        lines.append(f"  Total:      {self.total_ms:.0f}ms")
        if self.tool_rounds:
            lines.append(f"  Tools:      {self.tool_rounds} round(s), "
                         f"{len(self.tool_calls)} call(s): "
                         f"{', '.join(self.tool_calls)}")
        if self.rope_positions:
            lines.append(f"  RoPE:       max position {self.rope_positions}")
        if self.mid_gen_recalls:
            lines.append(f"  Mid-gen:    {self.mid_gen_recalls} recall(s) during generation")

        if backend == "vllm":
            # vLLM manages KV internally  -  show what we can
            if cache_stats:
                cs = cache_stats
                lines.append(f"  Convo:      {cs.get('conversation_tokens',0)} total tokens tracked")
                idx_count = len(cs.get('index_embeddings', [])) if 'index_embeddings' in cs else 0
                lines.append(f"  Semantic:   {cs.get('semantic_index_count', 0)} indexed chunks")
                if cs.get('total_recalls', 0):
                    lines.append(f"  Recalls:    {cs['total_recalls']} semantic recalls")
            # vLLM GPU stats  -  query from engine if available
            try:
                if vllm_engine is not None:
                    # Try to get vLLM's own stats
                    import torch
                    ngpu = torch.cuda.device_count()
                    total_alloc = 0
                    for i in range(ngpu):
                        try:
                            props = torch.cuda.get_device_properties(i)
                            total_alloc += props.total_memory
                        except Exception:
                            pass
                    if total_alloc > 0:
                        lines.append(f"  GPU total:  {total_alloc/1073741824:.1f} GB across {ngpu} GPUs")
            except Exception:
                pass
            # ROCm/CUDA memory from driver (not PyTorch allocator  -  vLLM uses its own pool)
            try:
                import torch
                if torch.cuda.is_available():
                    ngpu = torch.cuda.device_count()
                    total_free = 0; total_used = 0
                    for i in range(ngpu):
                        free, total = torch.cuda.mem_get_info(i)
                        total_free += free
                        total_used += (total - free)
                    lines.append(f"  VRAM used:  {total_used/1073741824:.2f} GB "
                                 f"(free: {total_free/1073741824:.2f} GB)")
            except Exception:
                pass
        else:
            # HF backend  -  show KeSSie KV cache stats
            if cache_stats:
                cs = cache_stats
                lines.append(f"  KV GPU:     {cs.get('kv_seq_len',0)} tokens in "
                             f"{cs.get('kv_layers',0)} layers, "
                             f"{cs.get('kv_gpu_mb',0)} MB ({cs.get('kv_dtype','?')})")
                lines.append(f"  KV CPU:     {cs.get('kv_cpu_tokens',0)} tokens in "
                             f"{cs.get('kv_cpu_chunks',0)} chunks, "
                             f"{cs.get('kv_cpu_mb',0)} MB")
                lines.append(f"  Window:     {cs.get('window_fill_pct',0)}% full "
                             f"({cs.get('kv_seq_len',0)}/{cs.get('window_capacity',0)})")
                lines.append(f"  Fog:        {cs.get('fog_tokens',0)} fogged, "
                             f"{cs.get('clear_tokens',0)} clear")
                if cs.get('total_evictions', 0):
                    lines.append(f"  Evictions:  {cs['total_evictions']} tokens evicted to CPU")
                if cs.get('total_recalls', 0):
                    lines.append(f"  Recalls:    {cs['total_recalls']} chunks recalled to GPU")
                lines.append(f"  Convo:      {cs.get('conversation_tokens',0)} total tokens")
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_alloc = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))
                    gpu_resv = sum(torch.cuda.memory_reserved(i) for i in range(torch.cuda.device_count()))
                    lines.append(f"  GPU mem:    {gpu_alloc/1073741824:.2f} GB allocated, "
                                 f"{gpu_resv/1073741824:.2f} GB reserved")
            except:
                pass

        if batch_stats:
            lines.append(f"  Batch:      {batch_stats.get('active',0)} active, "
                         f"{batch_stats.get('total_requests',0)} served, "
                         f"{batch_stats.get('total_tokens',0)} total tokens")
        try:
            import psutil
            proc = psutil.Process()
            ram = proc.memory_info()
            lines.append(f"  RAM:        {ram.rss/1073741824:.2f} GB RSS, "
                         f"{ram.vms/1073741824:.2f} GB VMS")
        except ImportError:
            pass
        lines.append(f"-----------------------------------------------")
        for line in lines:
            logger.info(line)


# =============================================================================
# Batch Generation Manager
# =============================================================================

class BatchGenManager:
    """
    Concurrent generation with batch management.
    Multiple requests can be in-flight. Up to 32 concurrent connections.
    Each request gets its own generation context.

    For single-GPU single-model, we batch by running requests concurrently
    in separate threads. The model's internal CUDA stream serialization
    handles GPU scheduling. For true batch prefill, we'd need vLLM-style
    continuous batching  -  this is the pragmatic version.
    """

    def __init__(self, max_concurrent=32):
        self.max_concurrent = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._active = 0
        self._active_lock = threading.Lock()
        self._total_requests = 0
        self._total_tokens = 0

    def acquire(self, timeout=60) -> bool:
        """Try to acquire a generation slot. Returns False if full."""
        got = self._semaphore.acquire(timeout=timeout)
        if got:
            with self._active_lock:
                self._active += 1
        return got

    def release(self, tokens_generated=0):
        with self._active_lock:
            self._active -= 1
            self._total_requests += 1
            self._total_tokens += tokens_generated
        self._semaphore.release()

    @property
    def active_count(self):
        with self._active_lock:
            return self._active

    @property
    def stats(self):
        with self._active_lock:
            return {"active": self._active, "max": self.max_concurrent,
                    "total_requests": self._total_requests, "total_tokens": self._total_tokens}


# =============================================================================
# Conversation Manager  -  Multi-tenant cache isolation
# =============================================================================

class ConversationManager:
    """
    Manages per-conversation KeSSie caches with concurrency control.

    - T simultaneous conversations (--conversation-threads)
    - Q waiting in queue when all threads busy
    - Each conversation gets its own KeSSieCache (isolated fog/recall state)
    - Caches evicted on conversation end or inactivity timeout
    - When no conversations active, all caches freed

    Client sends conversation_id in request. ConversationManager routes to
    the correct cache, creating/evicting as needed. Client refills cache
    on reconnect (KeSSie cache = external context window).
    """

    def __init__(self, max_threads: int = 4, max_queue: int = 64,
                 cache_size: int = 10_000_000, window_size: int = 131072,
                 fog_alpha: float = 0.5, fog_start: float = 0.5,
                 kv_cache_dtype=None):
        self.max_threads = max_threads
        self.max_queue = max_queue
        self.cache_size = cache_size
        self.window_size = window_size
        self.fog_alpha = fog_alpha
        self.fog_start = fog_start
        self.kv_cache_dtype = kv_cache_dtype

        # Active conversation caches: conv_id -> KeSSieCache
        self._caches: Dict[str, KeSSieCache] = {}
        # Per-conversation lock: prevents eviction during inference
        self._conv_locks: Dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

        # Concurrency: T threads active, Q queued
        self._thread_sem = threading.Semaphore(max_threads)
        self._queue_sem = threading.Semaphore(max_threads + max_queue)

        # Stats
        self._total_conversations = 0
        self._total_evictions = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def acquire(self, conv_id: str, timeout: float = 120.0) -> KeSSieCache:
        """
        Get or create a cache for conv_id. Blocks if queue is full.
        Returns the KeSSieCache for this conversation.
        Raises TimeoutError if can't acquire within timeout.
        """
        # Queue gate: limits total (active + waiting) to max_threads + max_queue
        if not self._queue_sem.acquire(timeout=timeout):
            raise TimeoutError(
                f"Server at capacity: {self.max_threads} active, "
                f"{self.max_queue} queued. Try again later.")

        # Thread gate: limits active generation to max_threads
        if not self._thread_sem.acquire(timeout=timeout):
            self._queue_sem.release()
            raise TimeoutError(
                f"All {self.max_threads} conversation threads busy. "
                f"Queued request timed out after {timeout}s.")

        with self._lock:
            if conv_id in self._caches:
                self._cache_hits += 1
                cache = self._caches[conv_id]
            else:
                self._cache_misses += 1
                self._total_conversations += 1
                cache = KeSSieCache(
                    window_size=self.window_size,
                    fog_alpha=self.fog_alpha,
                    fog_start=self.fog_start,
                    kv_cache_dtype=self.kv_cache_dtype,
                    max_conversation_tokens=self.cache_size)
                self._caches[conv_id] = cache
                self._conv_locks[conv_id] = threading.Lock()
            # Hold per-conversation lock during inference
            self._conv_locks[conv_id].acquire()

        return cache

    def release(self, conv_id: str):
        """Release the thread slot after generation completes."""
        with self._lock:
            lock = self._conv_locks.get(conv_id)
            if lock:
                try:
                    lock.release()
                except RuntimeError:
                    pass  # already released
        self._thread_sem.release()
        self._queue_sem.release()

    def end_conversation(self, conv_id: str):
        """
        End a conversation. Waits for any active inference to finish,
        then evicts the cache. Client refills on reconnect.
        """
        with self._lock:
            lock = self._conv_locks.get(conv_id)

        if lock:
            # Wait for inference to finish (blocks until release() called)
            lock.acquire()
            lock.release()

        with self._lock:
            if conv_id in self._caches:
                self._caches[conv_id].full_reset()
                del self._caches[conv_id]
                del self._conv_locks[conv_id]
                self._total_evictions += 1
                logger.info(f"Conversation {conv_id[:12]}... ended, cache evicted")
            # If no conversations left, ensure everything is clean
            if not self._caches:
                self._force_cleanup()

    def _force_cleanup(self):
        """Called when no conversations active. Ensures zero memory held."""
        import gc
        self._caches.clear()
        self._conv_locks.clear()
        gc.collect()
        logger.info("All conversations ended — caches fully released")

    @property
    def stats(self) -> dict:
        with self._lock:
            active_tokens = sum(len(c.conversation_tokens) for c in self._caches.values())
            active_index = sum(len(c.index_embeddings) for c in self._caches.values())
            return {
                "active_conversations": len(self._caches),
                "max_threads": self.max_threads,
                "max_queue": self.max_queue,
                "cache_size_per_conversation": self.cache_size,
                "total_conversations": self._total_conversations,
                "total_evictions": self._total_evictions,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "active_tokens": active_tokens,
                "active_index_entries": active_index,
                "active_memory_mb": round(
                    (active_tokens * 4 + active_index * 256 * 4) / 1048576, 1),
            }

    def get_cache(self, conv_id: str) -> Optional[KeSSieCache]:
        """Peek at a cache without acquiring (for stats/debug)."""
        with self._lock:
            return self._caches.get(conv_id)

    def list_conversations(self) -> List[dict]:
        """List active conversations with stats."""
        with self._lock:
            return [{
                "conversation_id": cid,
                "tokens": len(cache.conversation_tokens),
                "turns": cache._turn_counter,
                "index_entries": len(cache.index_embeddings),
                "active": self._conv_locks.get(cid, threading.Lock()).locked(),
            } for cid, cache in self._caches.items()]


# =============================================================================
# KeSSie Attention  -  LogitsProcessor for vLLM
# =============================================================================

class KeSSieContextManager:
    """
    Prompt-level KeSSie memory management for vLLM backend.
    Since vLLM V1 doesn't support per-request logits processors,
    KeSSie manages context BEFORE inference through intelligent prompt construction.

    Fog-of-war: Older messages get progressively compressed/summarized.
    Recall: Semantically relevant past context injected into prompt.
    Window: Messages trimmed to fit context window with priority to recent.
    """

    def __init__(self, fog_alpha=0.5, fog_start=0.5, window_size=4096):
        self.fog_alpha = fog_alpha
        self.fog_start = fog_start
        self.window_size = window_size

    def apply_fog_windowing(self, messages, tokenizer, max_prompt_tokens=None):
        """
        Apply fog-of-war at the message level:
        - System message: always kept in full
        - Recent messages (within fog_start fraction): kept in full
        - Older messages (fog zone): truncated proportionally by age

        Returns: windowed message list that fits within token budget.
        """
        if not messages:
            return messages

        budget = max_prompt_tokens or self.window_size

        # Separate system messages from conversation
        system_msgs = [m for m in messages if m.get("role") == "system"]
        conv_msgs = [m for m in messages if m.get("role") != "system"]

        if not conv_msgs:
            return messages

        # Estimate system tokens
        sys_tokens = 0
        for m in system_msgs:
            c = m.get("content", "")
            if isinstance(c, str):
                sys_tokens += len(tokenizer.encode(c, add_special_tokens=False))

        remaining = budget - sys_tokens - 256  # reserve for generation prompt overhead
        if remaining <= 0:
            return system_msgs + conv_msgs[-2:]  # at minimum keep last exchange

        # Clear zone: most recent fog_start fraction of messages
        n_conv = len(conv_msgs)
        clear_count = max(2, int(n_conv * self.fog_start))  # at least last exchange
        fog_msgs = conv_msgs[:n_conv - clear_count]
        clear_msgs = conv_msgs[n_conv - clear_count:]

        # Estimate clear zone tokens
        clear_tokens = 0
        for m in clear_msgs:
            c = m.get("content", "")
            if isinstance(c, str):
                clear_tokens += len(tokenizer.encode(c, add_special_tokens=False))
            elif isinstance(c, list):
                for part in c:
                    if part.get("type") == "text":
                        clear_tokens += len(tokenizer.encode(part["text"], add_special_tokens=False))

        fog_budget = remaining - clear_tokens
        if fog_budget <= 0 or not fog_msgs:
            # No room for fog zone  -  just return system + clear
            return system_msgs + clear_msgs

        # Apply fog: older messages get progressively truncated
        # Most recent fog message keeps ~(1-fog_alpha) of content
        # Oldest fog message keeps minimal content
        fogged_result = []
        n_fog = len(fog_msgs)
        fog_tokens_used = 0

        for i, m in enumerate(fog_msgs):
            c = m.get("content", "")
            if not isinstance(c, str):
                # Multimodal  -  keep reference but drop images in fog zone
                text_parts = []
                if isinstance(c, list):
                    for part in c:
                        if part.get("type") == "text":
                            text_parts.append(part["text"])
                c = " ".join(text_parts)

            if not c:
                fogged_result.append(m)
                continue

            # Age: 0 = oldest fog message, 1 = newest fog message
            age = i / max(n_fog - 1, 1)
            # Keep fraction: older = less, newer = more
            # At age=0: keep (1 - fog_alpha) minimum
            # At age=1: keep full
            keep_frac = (1.0 - self.fog_alpha) + self.fog_alpha * age
            keep_frac = max(keep_frac, 0.1)  # always keep at least 10%

            toks = tokenizer.encode(c, add_special_tokens=False)
            keep_n = max(8, int(len(toks) * keep_frac))

            if fog_tokens_used + keep_n > fog_budget:
                keep_n = max(8, fog_budget - fog_tokens_used)

            if keep_n < len(toks):
                # Truncate: keep the LAST keep_n tokens (most recent part of message)
                truncated = tokenizer.decode(toks[-keep_n:], skip_special_tokens=True)
                fogged_result.append({**m, "content": f"[...] {truncated}"})
            else:
                fogged_result.append(m)

            fog_tokens_used += keep_n
            if fog_tokens_used >= fog_budget:
                break  # drop remaining oldest messages

        return system_msgs + fogged_result + clear_msgs


# =============================================================================
# Engine
# =============================================================================

class LibrarianEngine:
    def __init__(self, model_name="Qwen/Qwen3-0.6B", num_gpus=4, dtype="bfloat16",
                 max_new_tokens=4096, window_size=4096, fog_alpha=0.5, fog_start=0.5,
                 db_path="kessie_knowledge.db", served_model_name=None, force_tp=False,
                 kv_cache_dtype=None, backend="hf",
                 gpu_memory_utilization=0.90, max_model_len=None, enforce_eager=False,
                 enable_hip_acceleration=False, kv_gpus=None,
                 kessie_cache_size=10_000_000, conversation_threads=4,
                 conversation_queue=64):
        self.model_name = model_name
        self.served_model_name = served_model_name or model_name
        self.num_gpus = num_gpus; self.dtype_str = dtype
        self.kv_cache_dtype_str = kv_cache_dtype
        self.max_new_tokens = max_new_tokens
        self.window_size = window_size
        self.fog_alpha = fog_alpha; self.fog_start = fog_start; self.force_tp = force_tp
        self.backend = backend
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager
        self.enable_hip_acceleration = enable_hip_acceleration
        self.kv_gpus = [int(g) for g in kv_gpus.split(",")] if kv_gpus else []
        self.model = None; self.tokenizer = None
        self.vllm_engine = None
        self.store = KnowledgeStore(db_path=db_path)
        self.executor = ToolExecutor(self.store)
        self._kv_cache_dtype = self._resolve_kv_dtype(kv_cache_dtype) if backend == "hf" else None
        self.kessie_cache_size = kessie_cache_size
        # Default cache for single-conversation / CLI mode
        self.cache = KeSSieCache(window_size=window_size, fog_alpha=fog_alpha, fog_start=fog_start,
                                 kv_cache_dtype=self._kv_cache_dtype,
                                 max_conversation_tokens=kessie_cache_size)
        # Multi-conversation manager for server mode
        self.conv_mgr = ConversationManager(
            max_threads=conversation_threads,
            max_queue=conversation_queue,
            cache_size=kessie_cache_size,
            window_size=window_size,
            fog_alpha=fog_alpha,
            fog_start=fog_start,
            kv_cache_dtype=self._kv_cache_dtype)
        self.batch_mgr = BatchGenManager(max_concurrent=conversation_threads + conversation_queue)
        self._model_kind = "causal"
        # KeSSie context manager  -  prompt-level fog-of-war + recall
        self.context_manager = KeSSieContextManager(
            fog_alpha=fog_alpha, fog_start=fog_start, window_size=window_size)
        self._internal_system = ""
        # Observable event log for mid-gen recall testing
        self._mid_gen_events = []
        self._last_user_query = ""

    @staticmethod
    def _resolve_kv_dtype(dtype_str):
        """Resolve KV cache dtype string to torch dtype."""
        if dtype_str is None:
            return None
        import torch
        mapping = {
            "fp8": torch.float8_e5m2, "fp8_e5m2": torch.float8_e5m2,
            "fp8_e5m3": torch.float8_e5m2,  # closest available; actual e5m3 if torch supports
            "fp8_e4m3": torch.float8_e4m3fn, "fp8_e4m3fn": torch.float8_e4m3fn,
            "float8_e5m2": torch.float8_e5m2, "float8_e4m3fn": torch.float8_e4m3fn,
            "float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float32": torch.float32, "fp32": torch.float32,
        }
        # Try exact match first
        if dtype_str in mapping:
            return mapping[dtype_str]
        # Try torch attribute
        try:
            return getattr(torch, dtype_str)
        except AttributeError:
            pass
        # Try float8 variants that may exist in newer torch
        for attr in [dtype_str, dtype_str.replace("-","_"), f"float8_{dtype_str.split('_',1)[-1]}"]:
            try:
                return getattr(torch, attr)
            except AttributeError:
                continue
        logger.warning(f"Unknown kv-cache-dtype '{dtype_str}', using model default")
        return None

    def _make_embedder(self):
        """Create embedder function for knowledge store. Works with both backends."""
        import torch
        import numpy as np

        if self.backend == "vllm":
            # vLLM: can't access model internals  -  use token-hash projection
            tokenizer = self.tokenizer
            np.random.seed(42)
            vocab_size = tokenizer.vocab_size or 32000
            proj = np.random.randn(vocab_size, self.store.embed_dim).astype(np.float32)
            proj /= np.sqrt(vocab_size)
            def embedder(text):
                toks = tokenizer.encode(text, add_special_tokens=False)[:128]
                if not toks: return np.zeros(self.store.embed_dim, dtype=np.float32)
                # Bag-of-tokens embedding via random projection
                vec = np.zeros(self.store.embed_dim, dtype=np.float32)
                for t in toks:
                    if t < vocab_size:
                        vec += proj[t]
                vec /= len(toks)
                return vec
            return embedder

        # HF: use actual model embeddings
        inner = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(inner, 'model') and hasattr(inner.model, 'embed_tokens'):
            el = inner.model.embed_tokens
        elif hasattr(inner, 'embed_tokens'):
            el = inner.embed_tokens
        else: return None
        dev = self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device
        ed = el.weight.shape[1]
        np.random.seed(42)
        proj = np.random.randn(ed, self.store.embed_dim).astype(np.float32) / np.sqrt(ed)
        tokenizer = self.tokenizer
        def embedder(text):
            toks = tokenizer.encode(text, add_special_tokens=False)[:128]
            if not toks: return np.zeros(self.store.embed_dim, dtype=np.float32)
            tt = torch.tensor(toks, dtype=torch.long, device=dev)
            with torch.no_grad():
                return el(tt).mean(dim=0).cpu().float().numpy() @ proj
        return embedder

    def load(self):
        import torch
        from transformers import AutoTokenizer
        logger.info(f"Loading {self.model_name} | GPUs={self.num_gpus} dtype={self.dtype_str} backend={self.backend}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.processor = None

        if self.backend == "vllm":
            self._load_vllm()
        else:
            self._load_hf()

        # Embedder for knowledge store
        emb = self._make_embedder()
        if emb:
            self.store.set_embedder(emb)
            if self.store.count() > 0: self.store.reembed_all()
        logger.info("Engine ready.")

    def _load_vllm(self):
        """Load model via vLLM AsyncLLMEngine with KeSSie attention + KV connector."""
        from vllm import AsyncLLMEngine, AsyncEngineArgs

        vllm_dtype = self.dtype_str
        if vllm_dtype in ("fp8", "fp8_e5m2", "fp8_e4m3fn", "float8"):
            vllm_dtype = "auto"

        # --- Layer 1: KeSSie Attention Backend (fog-of-war + recall boost) ---
        # Controls HOW the model attends  -  decay old context, boost recalled
        self._kessie_attention_active = False
        try:
            from kessie_attention import register_kessie_attention, KESSIE_STATE
            if register_kessie_attention():
                self._kessie_attention_active = True
                self._kessie_attn_state = KESSIE_STATE
                KESSIE_STATE.update(fog_alpha=self.fog_alpha, fog_start=self.cache.fog_start)
                logger.info(f"  KeSSie attention: fog_alpha={self.fog_alpha}, "
                            f"fog_start={self.cache.fog_start}")
        except ImportError:
            logger.info("  kessie_attention.py not found  -  default attention")
        except Exception as e:
            logger.warning(f"  KeSSie attention registration failed: {e}")

        vllm_kwargs = dict(
            model=self.model_name,
            tensor_parallel_size=self.num_gpus,
            dtype=vllm_dtype,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
            enable_prefix_caching=True,
        )
        if self.max_model_len:
            vllm_kwargs["max_model_len"] = self.max_model_len
        if self.kv_cache_dtype_str:
            vllm_kwargs["kv_cache_dtype"] = self.kv_cache_dtype_str

        # --- Layer 2: KeSSie KV Connector (token-linear store + recompute on recall) ---
        # -- KeSSie KV Connector: DISABLED --
        # vLLM's KV transfer path (has_kv_transfer_group() == True) forces
        # per-step overhead in gpu_model_runner even when all connector methods
        # are pure no-ops. Measured 100x throughput degradation.
        # Token store + recall lives in kessie_exp3.py via prompt prepend.
        # vLLM prefix caching handles KV reuse with zero overhead.
        self._kessie_kv_active = False
        logger.info(f"  KeSSie: token store active, recall via prompt prepend")

        logger.info(f"  vLLM config: tp={self.num_gpus}, prefix_caching=True, "
                    f"gpu_util={self.gpu_memory_utilization}, dtype={vllm_dtype}")

        engine_args = AsyncEngineArgs(**vllm_kwargs)
        self.vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._model_kind = "vllm"

        # Initialize token store with embedder + tokenizer
        try:
            from kessie_kv_connector import get_store
            self._token_store = get_store(chunk_size=128)
            self._token_store.set_tokenizer(self.tokenizer)
            emb = self._make_embedder()
            if emb:
                self._token_store.set_embedder(emb)
            logger.info(f"  KeSSie token store: O(4B/token), semantic index active")
        except Exception as e:
            logger.debug(f"  KV Store embedder setup: {e}")

        # --- Persistent async event loop in a background thread ---
        import asyncio
        self._vllm_loop = asyncio.new_event_loop()
        self._vllm_queues: Dict[str, queue.Queue] = {}
        self._vllm_queues_lock = threading.Lock()

        def _run_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._vllm_thread = threading.Thread(target=_run_loop, args=(self._vllm_loop,), daemon=True)
        self._vllm_thread.start()
        kv_mode = "KeSSie KV Connector" if self._kessie_kv_active else "default"
        logger.info(f"  vLLM AsyncLLMEngine + persistent loop started (KV: {kv_mode})")

    def _vllm_submit_generate(self, prompt, sampling_params, req_id):
        """Submit an async generate request to the persistent vLLM loop.
        Returns a queue.Queue that will receive (kind, data, token_id) tuples.
        kind: 'token' | 'done' | 'error'
        """
        import asyncio
        out_q = queue.Queue()

        with self._vllm_queues_lock:
            self._vllm_queues[req_id] = out_q

        async def _stream():
            emitted_text = ""  # what we've actually sent to the client
            all_ids = []
            try:
                async for output in self.vllm_engine.generate(prompt, sampling_params, request_id=req_id):
                    if not output.outputs:
                        continue
                    out = output.outputs[0]
                    cur_ids = list(out.token_ids)
                    new_ids = cur_ids[len(all_ids):]
                    all_ids = cur_ids

                    if new_ids:
                        full = self.tokenizer.decode(cur_ids, skip_special_tokens=True)
                        # Only emit text beyond what we've already sent
                        candidate = full[len(emitted_text):]
                        if candidate:
                            # Strip trailing replacement chars - incomplete multi-byte
                            safe = candidate.rstrip('\ufffd')
                            if safe:
                                out_q.put(("token", safe, new_ids[-1]))
                                emitted_text += safe

                # Final flush: re-decode everything cleanly, emit remainder
                if all_ids:
                    final = self.tokenizer.decode(all_ids, skip_special_tokens=True)
                    remaining = final[len(emitted_text):]
                    if remaining:
                        out_q.put(("token", remaining, all_ids[-1]))

            except Exception as e:
                out_q.put(("error", str(e), None))
            finally:
                out_q.put(("done", None, None))
                with self._vllm_queues_lock:
                    self._vllm_queues.pop(req_id, None)

        # Schedule on the persistent loop
        asyncio.run_coroutine_threadsafe(_stream(), self._vllm_loop)
        return out_q

    def _load_hf(self):
        """Load model via HuggingFace transformers."""
        import torch
        from transformers import AutoConfig

        dm = {"float16":torch.float16, "bfloat16":torch.bfloat16, "float32":torch.float32,
              "fp16":torch.float16, "bf16":torch.bfloat16, "fp32":torch.float32}
        if self.dtype_str in dm:
            td = dm[self.dtype_str]
        elif "fp8" in self.dtype_str or "float8" in self.dtype_str:
            td = self.dtype_str
        else:
            try: td = getattr(torch, self.dtype_str)
            except AttributeError: td = torch.bfloat16

        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        config_class = type(config).__name__.lower()
        model_type = getattr(config, "model_type", "").lower()
        logger.info(f"  Config: {type(config).__name__} model_type={model_type}")

        # Determine model class from architectures
        AutoModelClass = None
        self._model_kind = "causal"
        architectures = getattr(config, "architectures", []) or []
        if architectures:
            logger.info(f"  architectures: {architectures}")
            for arch in architectures:
                try:
                    import transformers
                    cls = getattr(transformers, arch, None)
                    if cls is not None:
                        AutoModelClass = cls
                        arch_lower = arch.lower()
                        if "conditionalgener" in arch_lower or "vision" in arch_lower or "vl" in arch_lower:
                            self._model_kind = "vision"
                        elif "seq2seq" in arch_lower:
                            self._model_kind = "seq2seq"
                        logger.info(f"  Resolved architecture -> {arch}")
                        break
                except Exception:
                    pass

        if AutoModelClass is None:
            import transformers
            for auto_name, kind in [("AutoModelForVision2Seq","vision"),("AutoModelForImageTextToText","vision"),
                                     ("AutoModelForCausalLM","causal"),("AutoModelForSeq2SeqLM","seq2seq"),
                                     ("AutoModel","auto")]:
                auto_cls = getattr(transformers, auto_name, None)
                if auto_cls:
                    AutoModelClass = auto_cls; self._model_kind = kind
                    logger.info(f"  Selected -> {auto_name}"); break

        if AutoModelClass is None:
            from transformers import AutoModel
            AutoModelClass = AutoModel; self._model_kind = "auto"

        # Handle pre-quantized models
        quant_config = getattr(config, "quantization_config", None)
        kw = dict(trust_remote_code=True)
        if quant_config is not None:
            kw["torch_dtype"] = "auto"
            logger.info(f"  Pre-quantized model detected")
        else:
            kw["torch_dtype"] = td

        # Patch MoE configs
        if "moe" in model_type or "moe" in config_class:
            cd = config.to_dict() if hasattr(config, "to_dict") else {}
            flat = json.dumps(cd)
            import re as _re
            matches = _re.findall(r'"num_(?:local_)?experts"\s*:\s*(\d+)', flat)
            if matches:
                n_experts = int(matches[0])
                config_cls = type(config)
                orig_ga = getattr(config_cls, '__getattr__', None)
                def _pga(self, name, _n=n_experts, _orig=orig_ga):
                    if name in ("num_experts", "num_local_experts"):
                        for a in ("text_config", "language_config", "llm_config"):
                            nested = self.__dict__.get(a)
                            if nested and hasattr(nested, name): return getattr(nested, name)
                        return _n
                    if _orig: return _orig(self, name)
                    raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
                config_cls.__getattr__ = _pga
                logger.info(f"  Patched MoE config: num_experts={n_experts}")

        if self.force_tp and self.num_gpus > 1:
            try:
                import tensor_parallel as tp_lib
                devs = [f"cuda:{i}" for i in range(self.num_gpus)]
                self.model = AutoModelClass.from_pretrained(self.model_name, device_map="cpu", attn_implementation="eager", **kw)
                self.model.eval()
                self.model = tp_lib.tensor_parallel(self.model, devs)
                logger.info(f"  Tensor parallel: {devs}")
            except ImportError: self.force_tp = False
        if not self.force_tp:
            self.model = AutoModelClass.from_pretrained(self.model_name, device_map="auto", **kw)
            self.model.eval()
            logger.info(f"  device_map='auto' across {self.num_gpus} GPUs")

        if self._model_kind == "vision":
            try:
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
                logger.info(f"  Loaded AutoProcessor for VL model")
            except Exception as e:
                logger.warning(f"  Failed to load processor: {e}")


    def _inject_system(self, messages, extra_tools=None):
        """Prepend system context. Only advertise tools if client provides them."""
        parts = []

        # Only add tool instructions if client sent tools
        if extra_tools:
            tool_names = []
            for t in extra_tools:
                fn = t.get("function", t) if "function" in t else t
                name = fn.get("name", "")
                desc = fn.get("description", "")
                if desc and len(desc) > 60:
                    desc = desc[:57] + "..."
                if name:
                    tool_names.append(f"  - {name}: {desc}" if desc else f"  - {name}")
            if tool_names:
                parts.append(f"Available tools:\n" + "\n".join(tool_names))
                parts.append(
                    f"To call a tool: <tool_call>\n"
                    f'{{"name": "tool_name", "arguments": {{"key": "value"}}}}\n'
                    f"</tool_call>\n"
                    f"Only use <tool_call> to execute tools. Code blocks do not execute.\n"
                    f"If no tool is needed, just respond normally."
                )

        system_content = "\n\n".join(parts) if parts else ""

        if not system_content:
            return messages

        has_system = False
        result = []
        for m in messages:
            if m.get("role") == "system":
                if not has_system:
                    result.append({"role": "system",
                                   "content": system_content + "\n\n" + m["content"]})
                    has_system = True
                else:
                    result.append(m)
            else:
                result.append(m)
        if not has_system:
            result.insert(0, {"role": "system", "content": system_content})
        return result

    def _get_device(self):
        import torch
        if self.backend == "vllm":
            return torch.device("cuda:0")
        if hasattr(self.model, 'device'): return self.model.device
        try: return next(self.model.parameters()).device
        except: return torch.device('cpu')

    def _extract_images(self, messages):
        """Extract PIL images from messages. Handles:
          - {"type": "image_url", "image_url": {"url": "data:...base64..."}}
          - {"type": "image_url", "image_url": {"url": "https://..."}}
          - {"type": "image", "image": "base64string..."}
          - {"type": "image", "source": {"type": "base64", "data": "..."}}
          - {"type": "image_url", "image_url": {"url": "base64string..."}}  (raw b64, no data: prefix)
        """
        images = []
        try:
            from PIL import Image
            import base64, io
        except ImportError:
            return images

        def _decode_b64(b64str):
            """Try to decode a base64 string to PIL Image."""
            try:
                # Strip data URI prefix if present
                if "," in b64str and b64str.index(",") < 100:
                    b64str = b64str.split(",", 1)[1]
                img_bytes = base64.b64decode(b64str)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                return None

        def _fetch_url(url):
            try:
                import urllib.request
                with urllib.request.urlopen(url, timeout=10) as resp:
                    return Image.open(io.BytesIO(resp.read())).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to fetch image URL: {e}")
                return None

        for m in messages:
            content = m.get("content", "")
            if not isinstance(content, list):
                continue
            for part in content:
                ptype = part.get("type", "")
                img = None

                if ptype == "image_url":
                    url_obj = part.get("image_url", {})
                    url = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
                    if url.startswith("http"):
                        img = _fetch_url(url)
                    elif url:
                        # data: URI or raw base64
                        img = _decode_b64(url)

                elif ptype == "image":
                    # Raw base64 in "image" field
                    raw = part.get("image", "")
                    if raw:
                        img = _decode_b64(raw)
                    # Anthropic-style source block
                    source = part.get("source", {})
                    if source and source.get("type") == "base64":
                        img = _decode_b64(source.get("data", ""))

                if img:
                    images.append(img)
                elif ptype in ("image_url", "image"):
                    logger.warning(f"Failed to decode image from {ptype} part")

        return images

    def _flatten_message_content(self, messages):
        """Flatten multipart content arrays to text for chat template."""
        flat = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                text_parts = []
                has_image = False
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part["text"])
                    elif part.get("type") == "image_url":
                        has_image = True
                # For VL models using processor, inject image placeholder
                text = "\n".join(text_parts)
                if has_image and self.processor:
                    # Qwen VL models expect <|vision_start|><|image_pad|><|vision_end|> or similar
                    # The processor handles this  -  just mark where images go
                    text = "<image>\n" + text if text else "<image>"
                flat.append({**m, "content": text})
            else:
                flat.append(m)
        return flat

    def _prepare_inputs(self, prompt, messages=None):
        """
        Prepare model inputs. For VL models with processor + images, the processor
        handles chat template + vision token insertion + pixel encoding.
        Falls back to text-only tokenizer path otherwise.
        """
        import torch
        dev = self._get_device()

        if self.processor and messages:
            images = self._extract_images(messages)
            if images:
                try:
                    # Build messages in the format processor expects
                    # Qwen3-VL processor needs content as list with type/image/text items
                    proc_msgs = []
                    img_idx = 0
                    for m in messages:
                        content = m.get("content", "")
                        if isinstance(content, list):
                            new_content = []
                            for part in content:
                                ptype = part.get("type", "")
                                if ptype == "text":
                                    new_content.append({"type": "text", "text": part["text"]})
                                elif ptype in ("image_url", "image"):
                                    if img_idx < len(images):
                                        new_content.append({"type": "image", "image": images[img_idx]})
                                        img_idx += 1
                            proc_msgs.append({"role": m["role"], "content": new_content})
                        else:
                            proc_msgs.append(m)

                    # Use processor's apply_chat_template  -  it knows about vision tokens
                    text = self.processor.apply_chat_template(
                        proc_msgs, tokenize=False, add_generation_prompt=True)
                    inputs = self.processor(
                        text=[text], images=images, return_tensors="pt",
                        padding=True)
                    inputs = {k: v.to(dev) if hasattr(v, 'to') else v for k, v in inputs.items()}
                    logger.info(f"  VL inputs: {list(inputs.keys())}, "
                               f"{len(images)} image(s), "
                               f"tokens={inputs['input_ids'].shape[1]}")
                    return inputs
                except Exception as e:
                    logger.warning(f"Processor failed ({e}), falling back to text-only")

        # Text-only path
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(dev)
        attn_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attn_mask}

    def _build_prompt(self, messages, recalled="", extra_tools=None):
        build = list(messages)
        if recalled:
            # Find system message to inject after, or prepend as new system message
            has_system = any(m["role"] == "system" for m in build)
            if has_system:
                aug = []
                for m in build:
                    aug.append(m)
                    if m["role"] == "system":
                        aug.append({"role":"system","content": recalled})
                build = aug
            else:
                # No system message — prepend recalled as system
                build.insert(0, {"role":"system","content": recalled})
        # Only pass client tools to chat template if provided
        tools_for_template = extra_tools if extra_tools else None
        try:
            return self.tokenizer.apply_chat_template(
                build, tools=tools_for_template, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        except:
            return self._manual_prompt(build, extra_tools)

    def _manual_prompt(self, messages, tools=None):
        # Only inject tool block if client tools are provided
        tb = ""
        if tools:
            tb = ("To call a tool:\n<tool_call>\n"
                  '{"name":"<tool_name>","arguments":{...}}\n'
                  "</tool_call>")
        parts, hs = [], False
        for m in messages:
            r, c = m.get("role","user"), m.get("content","")
            if r=="system":
                content = f"{c}\n\n{tb}" if tb else c
                parts.append(f"<|im_start|>system\n{content}<|im_end|>"); hs=True
            elif r=="user": parts.append(f"<|im_start|>user\n{c}<|im_end|>")
            elif r=="assistant": parts.append(f"<|im_start|>assistant\n{c}<|im_end|>")
            elif r=="tool": parts.append(f"<|im_start|>user\n<tool_response>\n{c}\n</tool_response><|im_end|>")
        if not hs:
            if tb:
                parts.insert(0, f"<|im_start|>system\n{tb}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def _parse_tool_calls(self, text):
        calls = []
        for m in re.finditer(r'<tool_call>\s*(.*?)\s*(?:</tool_call>|$)', text, re.DOTALL):
            raw = m.group(1).strip()
            if not raw: continue
            try:
                obj = json.loads(raw)
                n, a = obj.get("name",""), obj.get("arguments",{})
                if isinstance(a, str): a = json.loads(a)
                if n: calls.append({"id":f"call_{uuid.uuid4().hex[:8]}","name":n,"arguments":a})
            except: continue
        return calls

    def _annotate_recall(self, text: str, token_pos: int) -> str:
        """
        Annotate recalled text with positional context so the model
        understands where this content originally appeared in the conversation.
        
        Produces output like:
          [Recalled from turn 3/47 (user), ~2800 tokens ago:]
          <content>
          [End recalled]
        """
        role, turn_num, total_turns = self.cache.get_turn_for_position(token_pos)
        distance = self.cache.get_token_distance(token_pos)

        # Human-readable distance
        if distance < 500:
            dist_str = f"~{distance} tokens ago"
        elif distance < 5000:
            dist_str = f"~{distance // 100 * 100} tokens ago"
        else:
            dist_str = f"~{distance // 1000}k tokens ago"

        header = f"[Recalled from turn {turn_num}/{total_turns} ({role}), {dist_str}:]"
        return f"{header}\n{text}\n[End recalled]"

    def _auto_recall(self, user_message):
        """
        Semantic recall: search conversation history for relevant context.
        Works with both HF (embed_tokens) and vLLM (token-hash embedder) backends.
        Returns decoded text for injection into the prompt.
        """
        if not self.cache.index_embeddings:
            return ""

        # For vLLM: use the embedder function (token-hash projection)
        # For HF: use embed_tokens if available, else embedder
        embedder = getattr(self.store, '_embedder', None)
        qe = None

        if self.backend == "hf" and self.model is not None:
            try:
                import torch
                toks = self.tokenizer.encode(user_message, add_special_tokens=False)
                dev = self._get_device()
                tt = torch.tensor(toks, dtype=torch.long, device=dev)
                inner = self.model.module if hasattr(self.model, 'module') else self.model
                with torch.no_grad():
                    if hasattr(inner, 'model') and hasattr(inner.model, 'embed_tokens'):
                        emb = inner.model.embed_tokens(tt)
                        qe = emb.mean(dim=0).cpu().float().numpy()
                        if qe.shape[0] != 256:
                            qe = qe @ self.cache._get_projection(qe.shape[0])
            except Exception:
                pass

        if qe is None and embedder is not None:
            try:
                qe = embedder(user_message)
            except Exception:
                pass

        if qe is None:
            return ""

        # Search conversation index
        mat = np.stack(self.cache.index_embeddings)
        q = qe / (np.linalg.norm(qe) + 1e-8)
        norms = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        sims = norms @ q
        idx = np.argsort(sims)[-5:][::-1]
        results = [(self.cache.index_positions[i], float(sims[i])) for i in idx]

        # For HF: try to recall from CPU KV cache
        if self.backend == "hf":
            kl = self.cache.get_seq_length()
            cs = self.cache.window_start + int(kl * (1.0 - self.cache.fog_start))
            we = self.cache.window_start + kl
            fogged = [(p, s) for p, s in results if not (cs <= p < we) and s > 0.1]
            if not fogged:
                return ""
            has_cpu_kv = len(self.cache.cpu_kv_store) > 0
            if has_cpu_kv:
                dev = self._get_device()
                for pos, sc in fogged[:2]:
                    self.cache.recall_from_cpu(pos, dev)
                self.cache.evict_if_needed()
        else:
            # vLLM: no KV manipulation  -  just filter by similarity
            fogged = [(p, s) for p, s in results if s > 0.1]
            if not fogged:
                return ""

        # Return decoded text for prompt injection with positional annotation
        parts, total, mx = [], 0, self.window_size // 8
        for pos, sc in fogged[:2]:
            end = min(pos + self.cache.index_granularity, len(self.cache.conversation_tokens))
            chunk = self.cache.conversation_tokens[pos:end]
            if total + len(chunk) > mx:
                break
            text = self.tokenizer.decode(chunk, skip_special_tokens=True).strip()
            parts.append(self._annotate_recall(text, pos))
            total += len(chunk)
        return "\n".join(parts)

    # ----- Generation (concurrent, called from HTTP threads) -----

    def generate(self, messages: List[Dict], stream: bool = False, max_tool_rounds=3, extra_tools=None, sampling=None):
        """
        Generate response. Called concurrently from HTTP handler threads.
        BatchGenManager semaphore controls concurrency (up to 32).
        """
        import torch

        if not self.batch_mgr.acquire(timeout=60):
            return self._error_response("Server busy  -  max concurrent requests reached")

        try:
            return self._generate_inner(messages, stream, max_tool_rounds, extra_tools, sampling)
        finally:
            self.batch_mgr.release()

    def generate_stream(self, messages: List[Dict], max_tool_rounds=3, extra_tools=None, sampling=None):
        """Streaming generator. Yields SSE chunks for ALL rounds."""
        if not self.batch_mgr.acquire(timeout=60):
            yield self._sse_chunk(None, finish_reason="error")
            return
        try:
            yield from self._generate_stream_inner(messages, max_tool_rounds, extra_tools, sampling)
        finally:
            self.batch_mgr.release()

    # Known tools that server handles internally
    KNOWN_TOOLS = set()

    def _generate_stream_inner(self, messages, max_tool_rounds, extra_tools=None, sampling=None):
        """
        Stream ALL rounds. Server filters <tool_call> blocks for KNOWN tools only.
        """
        import torch
        stats = SessionStats()
        conversation = list(messages)
        conversation = self._inject_system(conversation, extra_tools)

        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        for rnd in range(max_tool_rounds + 1):
            user_msgs = [m["content"] for m in conversation if m["role"]=="user"]
            recalled = self._auto_recall(user_msgs[-1]) if user_msgs else ""
            self._last_user_query = user_msgs[-1] if user_msgs else ""

            # KeSSie context management: fog-of-war message windowing
            if self.backend == "vllm":
                windowed = self.context_manager.apply_fog_windowing(
                    conversation, self.tokenizer, max_prompt_tokens=self.window_size)
            else:
                windowed = conversation

            prompt = self._build_prompt(windowed, recalled, extra_tools)

            # Track user turn in conversation store for positional awareness
            if user_msgs:
                user_toks = self.tokenizer.encode(user_msgs[-1], add_special_tokens=False)
                if user_toks:
                    self.cache.append_conversation(user_toks, role="user")

            full_text = ""
            generated_ids = []
            buf = ""
            in_tool_call = False
            tool_call_buf = ""  # accumulates content inside <tool_call>...</tool_call>

            for token_text, token_id in self._do_generate_streaming(prompt, stats, conversation, sampling, recalled=recalled):
                full_text += token_text
                generated_ids.append(token_id)
                buf += token_text

                if in_tool_call:
                    tc_end = buf.find("</tool_call>")
                    if tc_end >= 0:
                        # Complete tool_call block captured
                        tool_call_buf += buf[:tc_end]
                        after = buf[tc_end + len("</tool_call>"):]
                        # Parse and check if it's a known tool
                        is_known = False
                        try:
                            obj = json.loads(tool_call_buf.strip())
                            if obj.get("name", "") in self.KNOWN_TOOLS:
                                is_known = True
                        except (json.JSONDecodeError, AttributeError):
                            pass
                        if not is_known:
                            # Unknown tool  -  emit as OpenAI-format tool_call
                            try:
                                tc_obj = json.loads(tool_call_buf.strip())
                                yield self._sse_tool_call_chunk(
                                    tc_obj.get("name", "unknown"),
                                    tc_obj.get("arguments", {}),
                                    req_id=req_id)
                            except (json.JSONDecodeError, AttributeError):
                                # Malformed  -  pass raw as content
                                yield self._sse_chunk(tool_call_buf, req_id=req_id)
                        # Reset
                        buf = after
                        in_tool_call = False
                        tool_call_buf = ""
                    else:
                        # Still accumulating tool_call content
                        tool_call_buf += buf
                        buf = ""
                    continue

                # Check for tool_call start
                tc_start = buf.find("<tool_call>")
                if tc_start >= 0:
                    # Flush content before the tag
                    before = buf[:tc_start]
                    if before:
                        yield self._sse_chunk(before, req_id=req_id)
                    rest = buf[tc_start + len("<tool_call>"):]
                    # Check if closing tag already in buffer
                    tc_end = rest.find("</tool_call>")
                    if tc_end >= 0:
                        # Complete block in one go
                        tool_call_content = rest[:tc_end]
                        after = rest[tc_end + len("</tool_call>"):]
                        is_known = False
                        try:
                            obj = json.loads(tool_call_content.strip())
                            if obj.get("name", "") in self.KNOWN_TOOLS:
                                is_known = True
                        except (json.JSONDecodeError, AttributeError):
                            pass
                        if not is_known:
                            try:
                                tc_obj = json.loads(tool_call_content.strip())
                                yield self._sse_tool_call_chunk(
                                    tc_obj.get("name", "unknown"),
                                    tc_obj.get("arguments", {}),
                                    req_id=req_id)
                            except (json.JSONDecodeError, AttributeError):
                                yield self._sse_chunk(tool_call_content, req_id=req_id)
                        buf = after
                    else:
                        tool_call_buf = rest
                        buf = ""
                        in_tool_call = True
                    continue

                # No tool_call tag  -  buffer to avoid splitting tags
                last_lt = buf.rfind('<')
                if last_lt > 0:
                    yield self._sse_chunk(buf[:last_lt], req_id=req_id)
                    buf = buf[last_lt:]
                elif last_lt < 0 and len(buf) > 20:
                    yield self._sse_chunk(buf, req_id=req_id)
                    buf = ""

            # Flush remaining
            if in_tool_call and tool_call_buf:
                # Incomplete tool_call at end  -  try to parse and emit properly
                try:
                    tc_obj = json.loads(tool_call_buf.strip())
                    yield self._sse_tool_call_chunk(
                        tc_obj.get("name", "unknown"),
                        tc_obj.get("arguments", {}),
                        req_id=req_id)
                except (json.JSONDecodeError, AttributeError):
                    yield self._sse_chunk(tool_call_buf, req_id=req_id)
            elif buf:
                yield self._sse_chunk(buf, req_id=req_id)

            self.cache.append_conversation(generated_ids)

            # Check for tool calls  -  only execute known ones
            calls = self._parse_tool_calls(full_text)
            known_calls = [c for c in calls if c["name"] in self.KNOWN_TOOLS]
            unknown_calls = [c for c in calls if c["name"] not in self.KNOWN_TOOLS]
            if not known_calls:
                fr = "tool_calls" if unknown_calls else "stop"
                yield self._sse_chunk(None, finish_reason=fr, req_id=req_id)
                yield "data: [DONE]\n\n"
                stats.log(self.cache.get_stats(), self.batch_mgr.stats, backend=self.backend, vllm_engine=self.vllm_engine)
                return

            stats.tool_rounds += 1
            stats.tool_calls.extend(tc["name"] for tc in known_calls)
            logger.info(f"Stream round {rnd+1}: {len(known_calls)} known tool call(s)")
            conversation.append({"role":"assistant","content":full_text})
            for tc in known_calls:
                logger.info(f"  -> {tc['name']}({json.dumps(tc['arguments'])[:80]})")
                result = self.executor.execute(tc["name"], tc["arguments"])
                logger.info(f"  <- {result[:200]}")
                conversation.append({"role":"tool","content":result})

        yield self._sse_chunk(None, finish_reason="stop", req_id=req_id)
        yield "data: [DONE]\n\n"
        stats.log(self.cache.get_stats(), self.batch_mgr.stats, backend=self.backend, vllm_engine=self.vllm_engine)

    def _generate_inner(self, messages, stream, max_tool_rounds, extra_tools=None, sampling=None):
        """Non-streaming generation with tool loop."""
        import torch
        stats = SessionStats()
        conversation = list(messages)
        conversation = self._inject_system(conversation, extra_tools)

        for rnd in range(max_tool_rounds + 1):
            user_msgs = [m["content"] for m in conversation if m["role"]=="user"]
            recalled = self._auto_recall(user_msgs[-1]) if user_msgs else ""
            self._last_user_query = user_msgs[-1] if user_msgs else ""

            # KeSSie context management: fog-of-war message windowing
            if self.backend == "vllm":
                windowed = self.context_manager.apply_fog_windowing(
                    conversation, self.tokenizer, max_prompt_tokens=self.window_size)
            else:
                windowed = conversation

            prompt = self._build_prompt(windowed, recalled, extra_tools)

            # Track user turn in conversation store
            if user_msgs:
                user_toks = self.tokenizer.encode(user_msgs[-1], add_special_tokens=False)
                if user_toks:
                    self.cache.append_conversation(user_toks, role="user")

            generated, prompt_len = self._do_generate(prompt, stats, conversation, sampling, recalled=recalled)
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            self.cache.append_conversation(generated)
            calls = self._parse_tool_calls(text)
            ci = text.find("<tool_call>")
            content = text[:ci].strip() if ci >= 0 else text.strip()
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            if not calls:
                stats.log(self.cache.get_stats(), self.batch_mgr.stats, backend=self.backend, vllm_engine=self.vllm_engine)
                return self._ok_response(content, prompt_len, len(generated))

            known_calls = [c for c in calls if c["name"] in self.KNOWN_TOOLS]
            unknown_calls = [c for c in calls if c["name"] not in self.KNOWN_TOOLS]

            # If there are unknown/client tool calls, return them in OpenAI format
            if unknown_calls:
                oai_tool_calls = []
                for i, tc in enumerate(unknown_calls):
                    oai_tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"], ensure_ascii=False)
                        }
                    })
                if not known_calls:
                    # Only client tools  -  return immediately for client to handle
                    stats.log(self.cache.get_stats(), self.batch_mgr.stats, backend=self.backend, vllm_engine=self.vllm_engine)
                    return self._ok_response(content, prompt_len, len(generated), oai_tool_calls)

            # Execute known/server tools and continue loop
            stats.tool_rounds += 1
            stats.tool_calls.extend(tc["name"] for tc in known_calls)
            logger.info(f"Round {rnd+1}: {len(known_calls)} known tool call(s)")
            conversation.append({"role":"assistant","content":text})
            for tc in known_calls:
                logger.info(f"  -> {tc['name']}({json.dumps(tc['arguments'])[:80]})")
                r = self.executor.execute(tc["name"], tc["arguments"])
                logger.info(f"  <- {r[:200]}")
                conversation.append({"role":"tool","content":r})
        stats.log(self.cache.get_stats(), self.batch_mgr.stats, backend=self.backend, vllm_engine=self.vllm_engine)
        return self._ok_response(content or "Done.", 0, 0)

    def _do_generate(self, prompt, stats: SessionStats = None, messages=None, sampling=None, recalled="") -> Tuple[List[int], int]:
        """Run generation and return (token_ids, prompt_len). Dispatches to backend."""
        if self.backend == "vllm":
            return self._do_generate_vllm(prompt, stats, sampling, recalled=recalled)
        return self._do_generate_hf(prompt, stats, messages, sampling)

    def _build_vllm_params(self, sampling, prompt_len=0):
        """Build vLLM SamplingParams  -  V1 compatible, no per-request logits_processors."""
        from vllm import SamplingParams
        sp = sampling or {}
        temp = sp.get("temperature", 0.7)
        greedy = sp.get("greedy", False)

        # Build stop token IDs  -  include all special stop tokens
        stop_token_ids = []
        for tok_name in ["eos_token_id", "eot_token_id"]:
            tid = getattr(self.tokenizer, tok_name, None)
            if tid is not None:
                stop_token_ids.append(tid)
        # Also get <|im_end|> token ID explicitly
        try:
            im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end_id is not None and im_end_id != self.tokenizer.unk_token_id:
                stop_token_ids.append(im_end_id)
        except Exception:
            pass

        return SamplingParams(
            temperature=0.0 if greedy else max(temp, 0.01),
            top_p=sp.get("top_p", 0.8),
            top_k=sp.get("top_k", 20) if not greedy else -1,
            max_tokens=sp.get("max_tokens") or self.max_new_tokens,
            repetition_penalty=sp.get("repetition_penalty", 1.0),
            presence_penalty=sp.get("presence_penalty", 0.0),
            stop_token_ids=stop_token_ids if stop_token_ids else None,
        )

    def _update_kessie_attention(self, prompt_len, recalled="", knowledge="", prompt_text=""):
        """
        Update KeSSie attention state before generation.
        
        When prompt_text is provided, finds actual token positions of recalled
        text so fog-of-war fully un-fogs + amplifies those positions.
        """
        if not getattr(self, '_kessie_attention_active', False):
            return
        try:
            recall_positions = set()
            if recalled:
                if prompt_text:
                    # Find recalled text in the actual prompt string
                    char_offset = prompt_text.find("[Recalled from turn")
                    if char_offset < 0:
                        char_offset = prompt_text.find(recalled[:50])

                    if char_offset >= 0:
                        # Encode prefix to get token offset
                        prefix_toks = self.tokenizer.encode(
                            prompt_text[:char_offset], add_special_tokens=False)
                        recall_toks = self.tokenizer.encode(
                            recalled, add_special_tokens=False)
                        start = len(prefix_toks)
                        end = min(start + len(recall_toks), prompt_len)
                        recall_positions = set(range(start, end))
                    else:
                        # Can't find — mark from position 0
                        recall_toks = self.tokenizer.encode(recalled, add_special_tokens=False)
                        recall_positions = set(range(0, min(len(recall_toks), prompt_len)))
                else:
                    # No prompt text — approximate from position 0
                    recall_toks = self.tokenizer.encode(recalled, add_special_tokens=False)
                    recall_positions = set(range(0, min(len(recall_toks), prompt_len)))

            self._kessie_attn_state.update(
                prompt_len=prompt_len,
                recall_positions=recall_positions,
            )
        except Exception as e:
            logger.debug(f"KeSSie attention state update: {e}")

    def _do_generate_vllm(self, prompt, stats, sampling, recalled="", knowledge="") -> Tuple[List[int], int]:
        """Generate via vLLM engine (non-streaming)  -  collect from async stream."""
        input_ids = self.tokenizer.encode(prompt)
        pl = len(input_ids)
        if pl > self.window_size:
            prompt = self.tokenizer.decode(input_ids[-self.window_size:], skip_special_tokens=False)
            pl = self.window_size
            if stats: stats.truncated = True

        vllm_params = self._build_vllm_params(sampling, pl)
        req_id = f"kessie-{uuid.uuid4().hex[:12]}"

        # Update KeSSie attention state (fog-of-war + recall boost)
        self._update_kessie_attention(pl, recalled, knowledge, prompt_text=prompt)

        if stats:
            stats.prompt_tokens = pl
            stats.rope_positions = pl
            stats.t_prefill_start = time.perf_counter()

        # Submit to persistent async loop and drain all tokens
        out_q = self._vllm_submit_generate(prompt, vllm_params, req_id)
        full_text = ""
        while True:
            try:
                kind, data, tid = out_q.get(timeout=300)
            except queue.Empty:
                break
            if kind == "done":
                break
            elif kind == "error":
                logger.warning(f"vLLM generate error: {data}")
                break
            elif kind == "token":
                full_text += data

        gen_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        if stats:
            stats.t_prefill_end = stats.t_decode_end = time.perf_counter()
            stats.generated_tokens += len(gen_ids)
            stats.rope_positions = pl + len(gen_ids)

        return gen_ids, pl

    def _do_generate_hf(self, prompt, stats: SessionStats = None, messages=None, sampling=None) -> Tuple[List[int], int]:
        """Run HF model.generate() and return (token_ids, prompt_len)."""
        import torch
        sp = sampling or {}
        temp = sp.get("temperature", 0.7)
        top_p = sp.get("top_p", 0.8)
        top_k = sp.get("top_k", 20)
        rep_pen = sp.get("repetition_penalty", 1.0)
        pres_pen = sp.get("presence_penalty", 1.5)
        max_tok = sp.get("max_tokens") or self.max_new_tokens
        greedy = sp.get("greedy", False)
        do_sample = not greedy and temp > 0

        with torch.inference_mode():
            inputs = self._prepare_inputs(prompt, messages)
            input_ids = inputs["input_ids"]
            pl = input_ids.shape[1]
            if pl > self.window_size:
                input_ids = input_ids[:, -self.window_size:]
                inputs["input_ids"] = input_ids
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -self.window_size:]
                if stats: stats.truncated = True
                pl = input_ids.shape[1]

            if stats:
                stats.prompt_tokens = pl
                stats.rope_positions = pl
                stats.t_prefill_start = time.perf_counter()

            if not self.force_tp:
                try:
                    gen_kwargs = dict(
                        max_new_tokens=max_tok,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.pad_token_id)
                    if do_sample:
                        gen_kwargs.update(temperature=temp, top_p=top_p, top_k=top_k)
                    if rep_pen != 1.0:
                        gen_kwargs["repetition_penalty"] = rep_pen
                    # HF doesn't have presence_penalty natively  -  fold into repetition_penalty
                    # presence_penalty adds a flat penalty to seen tokens vs multiplicative
                    # We combine: effective_rep = repetition_penalty + (presence_penalty - 1.0) * 0.1
                    if pres_pen and pres_pen != 0:
                        effective_rep = max(rep_pen + (pres_pen - 1.0) * 0.1, 1.0)
                        gen_kwargs["repetition_penalty"] = effective_rep
                    out = self.model.generate(**inputs, **gen_kwargs)
                    gen_ids = out[0][pl:].tolist()
                    if stats:
                        stats.t_prefill_end = stats.t_decode_end = time.perf_counter()
                        stats.generated_tokens += len(gen_ids)
                        stats.rope_positions = pl + len(gen_ids)
                    return gen_ids, pl
                except Exception as e:
                    logger.warning(f"model.generate() failed ({e}), manual decode")

            if stats:
                stats.t_prefill_start = time.perf_counter()
            gen_ids = self._manual_decode(input_ids, pl)
            if stats:
                stats.t_decode_end = time.perf_counter()
                stats.generated_tokens += len(gen_ids)
                stats.rope_positions = pl + len(gen_ids)
            return gen_ids, pl

    def _do_generate_streaming(self, prompt, stats: SessionStats = None, messages=None, sampling=None, recalled=""):
        """Streaming generation. Dispatches to backend."""
        if self.backend == "vllm":
            yield from self._do_generate_streaming_vllm(prompt, stats, sampling, recalled=recalled)
        else:
            yield from self._do_generate_streaming_hf(prompt, stats, messages, sampling)

    def _mid_gen_recall(self, generated_text: str, recalled_already: str = "") -> str:
        """
        Mid-generation semantic recall. Embeds the text generated so far,
        searches the conversation store for relevant fogged context.
        Returns recalled text or "" if nothing useful found.
        Lightweight: uses cached embedder, no GPU ops.
        """
        if not self.cache.index_embeddings:
            return ""

        embedder = getattr(self.store, '_embedder', None)
        if embedder is None:
            return ""

        # Strip hedge phrases from search text — they add noise, not signal.
        # We want to search with the topical content (e.g. "Korthax pipeline 
        # configuration shard count port compression failover timeout")
        search_text = generated_text.lower()
        for pattern in self._HEDGE_PATTERNS:
            search_text = search_text.replace(pattern, "")
        # Also strip common filler
        for filler in ["regarding", "about the", "from our", "earlier conversation",
                        "let me", "see if", "i can", "...", "the exact", "we discussed"]:
            search_text = search_text.replace(filler, "")
        search_text = " ".join(search_text.split())  # collapse whitespace

        # Also include the user's original query — best keywords are there
        user_query = getattr(self, '_last_user_query', '')
        if user_query:
            combined_search = search_text + " " + user_query
        else:
            combined_search = search_text
        if not combined_search.strip():
            combined_search = generated_text  # fallback to original

        try:
            qe = embedder(combined_search)
        except Exception:
            return ""

        if qe is None:
            return ""

        # Search conversation index
        mat = np.stack(self.cache.index_embeddings)
        q = qe / (np.linalg.norm(qe) + 1e-8)
        norms = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        sims = norms @ q
        idx = np.argsort(sims)[-5:][::-1]
        results = [(self.cache.index_positions[i], float(sims[i])) for i in idx]

        # Filter: only fogged positions with high similarity
        fogged = [(p, s) for p, s in results if s > 0.1]
        if not fogged:
            return ""

        # Decode recalled chunks with positional annotation
        parts, total, mx = [], 0, self.window_size // 8
        for pos, sc in fogged[:2]:
            end = min(pos + self.cache.index_granularity, len(self.cache.conversation_tokens))
            chunk = self.cache.conversation_tokens[pos:end]
            if total + len(chunk) > mx:
                break
            text = self.tokenizer.decode(chunk, skip_special_tokens=True).strip()
            if text and text not in recalled_already:
                parts.append(self._annotate_recall(text, pos))
                total += len(chunk)

        return "\n".join(parts)

    def _abort_vllm_request(self, req_id: str):
        """Abort an in-flight vLLM generation request."""
        import asyncio
        try:
            engine = self.vllm_engine
            if hasattr(engine, 'abort'):
                asyncio.run_coroutine_threadsafe(
                    engine.abort(req_id), self._vllm_loop).result(timeout=5)
            elif hasattr(engine, 'engine') and hasattr(engine.engine, 'abort_request'):
                asyncio.run_coroutine_threadsafe(
                    engine.engine.abort_request(req_id), self._vllm_loop).result(timeout=5)
        except Exception as e:
            logger.debug(f"vLLM abort for {req_id}: {e}")

    def _rebuild_prompt_with_recall(self, original_prompt: str, generated_so_far: str,
                                     recalled_text: str) -> str:
        """
        Rebuild prompt for resume-after-recall:
        [original prompt with recalled context injected] + [cleaned partial output]
        
        The partial assistant output has hedge phrases stripped so the model 
        continues from topical content, not from "let me see if I can remember..."
        The partial output is appended WITHOUT a closing tag so the model
        continues generating from where it left off.
        """
        # Strip hedge phrases from partial output
        # Model said: "Regarding the Korthax pipeline... let me see if I can remember..."
        # We want: "Regarding the Korthax pipeline..."
        cleaned_partial = generated_so_far
        lower = cleaned_partial.lower()
        # Find the earliest hedge phrase and truncate there
        earliest_pos = len(cleaned_partial)
        for pattern in self._HEDGE_PATTERNS:
            pos = lower.find(pattern)
            if pos >= 0 and pos < earliest_pos:
                earliest_pos = pos
        if earliest_pos < len(cleaned_partial):
            cleaned_partial = cleaned_partial[:earliest_pos].rstrip(". ,;:\n")
            if cleaned_partial:
                cleaned_partial += "\n"  # clean break before model continues
            logger.info(f"  Stripped hedge from partial: '{generated_so_far[-60:]}' -> '{cleaned_partial[-60:]}'")

        # Inject recalled context into the prompt
        recall_block = f"\n[Recalled context:]\n{recalled_text}\n[End recalled]\n"

        # Insert before the final assistant generation prompt
        # Look for the assistant start marker
        markers = ["<|im_start|>assistant\n", "<|start_header_id|>assistant<|end_header_id|>\n"]
        for marker in markers:
            last_pos = original_prompt.rfind(marker)
            if last_pos >= 0:
                insert_at = last_pos
                return (original_prompt[:insert_at] +
                        f"<|im_start|>system\n{recall_block}<|im_end|>\n" +
                        original_prompt[insert_at:] +
                        cleaned_partial)

        # Fallback: just append
        return original_prompt + recall_block + cleaned_partial

    # --- Uncertainty detection for mid-generation recall ---

    # Hedging phrases that suggest the model needs context it doesn't have
    _HEDGE_PATTERNS = [
        "as i mentioned", "as we discussed", "you mentioned earlier",
        "i think you said", "if i recall", "i believe you",
        "previously", "earlier in our conversation",
        "i'm not sure if", "i don't have enough context",
        "could you remind me", "can you clarify",
        "based on what you told me", "from what i remember",
        "let me see if i can remember", "i don't remember the exact",
    ]

    def _check_uncertainty(self, generated_text: str, token_count: int) -> bool:
        """
        Check if generated text shows signs of needing fogged context.
        Called periodically during streaming decode.
        
        Returns True if mid-generation recall should be triggered.
        """
        # Don't check too early (need a few tokens to detect patterns)
        if token_count < 8:
            return False

        # Check every 8 tokens (was 16 — too sparse for short hedges)
        if token_count % 8 != 0:
            return False

        # Check last ~100 chars for hedging patterns
        tail = generated_text[-150:].lower()
        for pattern in self._HEDGE_PATTERNS:
            if pattern in tail:
                logger.info(f"  Mid-gen recall triggered: hedge pattern '{pattern}' at token {token_count}")
                self._mid_gen_events.append({
                    "event": "uncertainty_detected",
                    "pattern": pattern,
                    "token_count": token_count,
                    "tail": tail[-80:],
                })
                return True

        # Check for repetition (model spinning = lost context)
        if token_count >= 60:
            last_50 = generated_text[-200:]
            prev_50 = generated_text[-400:-200]
            if prev_50 and last_50:
                # Simple overlap check: if >50% of recent bigrams appeared in previous window
                def bigrams(s):
                    words = s.lower().split()
                    return set(zip(words, words[1:])) if len(words) > 1 else set()
                recent = bigrams(last_50)
                previous = bigrams(prev_50)
                if recent and len(recent & previous) / len(recent) > 0.5:
                    logger.info(f"  Mid-gen recall triggered: repetition at token {token_count}")
                    self._mid_gen_events.append({
                        "event": "repetition_detected",
                        "token_count": token_count,
                        "overlap": len(recent & previous) / len(recent),
                    })
                    return True

        return False

    def _do_generate_streaming_vllm(self, prompt, stats, sampling, recalled="", knowledge=""):
        """
        True streaming via persistent vLLM async loop with KeSSie attention.
        
        Includes mid-generation recall: if the model shows uncertainty signals
        (hedging, repetition), generation pauses, relevant fogged context is
        recalled from the store, and generation resumes with the recalled
        context injected and partial output preserved.
        
        Max 1 recall per generation to prevent loops.
        """
        input_ids = self.tokenizer.encode(prompt)
        pl = len(input_ids)
        if pl > self.window_size:
            prompt = self.tokenizer.decode(input_ids[-self.window_size:], skip_special_tokens=False)
            pl = self.window_size
            if stats: stats.truncated = True

        vllm_params = self._build_vllm_params(sampling, pl)
        req_id = f"kessie-{uuid.uuid4().hex[:12]}"

        # Update KeSSie attention state (fog-of-war + recall boost)
        self._update_kessie_attention(pl, recalled, knowledge, prompt_text=prompt)

        if stats:
            stats.prompt_tokens = pl
            stats.rope_positions = pl
            stats.t_prefill_start = time.perf_counter()

        # Submit to persistent async loop
        out_q = self._vllm_submit_generate(prompt, vllm_params, req_id)

        gen_count = 0
        first_token = True
        generated_text = ""
        recall_used = False  # max 1 recall per generation
        original_prompt = prompt

        while True:
            try:
                kind, data, tid = out_q.get(timeout=300)
            except queue.Empty:
                logger.warning(f"vLLM stream timeout for {req_id}")
                break

            if kind == "done":
                # Before exiting: if model hit EOS with a hedge in output
                # and we haven't recalled yet, this is the mid-gen opportunity.
                # Do recall, rebuild prompt with partial hedge + recalled, resume.
                if (not recall_used and
                        self.cache.index_embeddings and
                        gen_count >= 4):
                    # Force check regardless of token alignment
                    forced_count = (gen_count // 8) * 8 if gen_count % 8 != 0 else gen_count
                    if forced_count < 8:
                        forced_count = 8
                    if self._check_uncertainty(generated_text, forced_count):
                        mid_recalled = self._mid_gen_recall(generated_text, recalled)
                        if mid_recalled:
                            logger.info(f"  Mid-gen recall at EOS: {len(mid_recalled)} chars")
                            recall_used = True
                            self._mid_gen_events.append({
                                "event": "recall_found",
                                "token_count": gen_count,
                                "recalled_chars": len(mid_recalled),
                                "recalled_preview": mid_recalled[:120],
                                "generated_before_recall": generated_text[:200],
                                "trigger": "eos_hedge",
                            })

                            new_prompt = self._rebuild_prompt_with_recall(
                                original_prompt, generated_text, mid_recalled)
                            new_ids = self.tokenizer.encode(new_prompt)
                            new_pl = len(new_ids)
                            if new_pl > self.window_size:
                                new_prompt = self.tokenizer.decode(
                                    new_ids[-self.window_size:], skip_special_tokens=False)
                                new_pl = self.window_size

                            combined_recalled = (recalled + "\n" + mid_recalled).strip()
                            self._update_kessie_attention(new_pl, combined_recalled, knowledge,
                                                           prompt_text=new_prompt)

                            new_req_id = f"kessie-{uuid.uuid4().hex[:12]}"
                            new_params = self._build_vllm_params(sampling, new_pl)
                            out_q = self._vllm_submit_generate(
                                new_prompt, new_params, new_req_id)
                            req_id = new_req_id

                            if stats:
                                stats.prompt_tokens = new_pl
                                stats.rope_positions = new_pl
                                stats.mid_gen_recalls = getattr(stats, 'mid_gen_recalls', 0) + 1

                            logger.info(f"  Mid-gen resume after EOS: new prompt {new_pl} tokens")
                            self._mid_gen_events.append({
                                "event": "resume",
                                "new_prompt_tokens": new_pl,
                                "new_req_id": new_req_id,
                                "trigger": "eos_hedge",
                            })
                            first_token = True
                            continue  # Don't break — read from new request
                        else:
                            self._mid_gen_events.append({
                                "event": "recall_empty",
                                "token_count": gen_count,
                                "generated_preview": generated_text[-100:],
                                "trigger": "eos_hedge",
                            })
                break
            elif kind == "error":
                logger.warning(f"vLLM stream error: {data}")
                break
            elif kind == "token":
                if first_token and stats:
                    stats.t_prefill_end = time.perf_counter()
                    stats.t_decode_start = time.perf_counter()
                    first_token = False
                gen_count += 1
                generated_text += data
                yield (data, tid)

                # --- Mid-generation recall check ---
                if (not recall_used and
                        self.cache.index_embeddings and
                        self._check_uncertainty(generated_text, gen_count)):

                    # Attempt recall using generated text as query
                    mid_recalled = self._mid_gen_recall(generated_text, recalled)

                    if mid_recalled:
                        logger.info(f"  Mid-gen recall: {len(mid_recalled)} chars at token {gen_count}")
                        recall_used = True
                        self._mid_gen_events.append({
                            "event": "recall_found",
                            "token_count": gen_count,
                            "recalled_chars": len(mid_recalled),
                            "recalled_preview": mid_recalled[:120],
                            "generated_before_recall": generated_text[:200],
                        })

                        # 1. Abort current generation
                        self._abort_vllm_request(req_id)
                        # Drain remaining tokens from the queue
                        while True:
                            try:
                                k, _, _ = out_q.get(timeout=2)
                                if k == "done" or k == "error":
                                    break
                            except queue.Empty:
                                break

                        # 2. Rebuild prompt with recalled context + partial output
                        new_prompt = self._rebuild_prompt_with_recall(
                            original_prompt, generated_text, mid_recalled)

                        new_ids = self.tokenizer.encode(new_prompt)
                        new_pl = len(new_ids)
                        if new_pl > self.window_size:
                            new_prompt = self.tokenizer.decode(
                                new_ids[-self.window_size:], skip_special_tokens=False)
                            new_pl = self.window_size

                        # 3. Update fog state with recall positions
                        combined_recalled = (recalled + "\n" + mid_recalled).strip()
                        self._update_kessie_attention(new_pl, combined_recalled, knowledge,
                                                       prompt_text=new_prompt)

                        # 4. Re-submit with new prompt
                        new_req_id = f"kessie-{uuid.uuid4().hex[:12]}"
                        new_params = self._build_vllm_params(sampling, new_pl)
                        out_q = self._vllm_submit_generate(
                            new_prompt, new_params, new_req_id)
                        req_id = new_req_id

                        if stats:
                            stats.prompt_tokens = new_pl
                            stats.rope_positions = new_pl
                            stats.mid_gen_recalls = getattr(stats, 'mid_gen_recalls', 0) + 1

                        logger.info(f"  Mid-gen resume: new prompt {new_pl} tokens, "
                                    f"continuing from '{generated_text[-50:]}'")
                        self._mid_gen_events.append({
                            "event": "resume",
                            "new_prompt_tokens": new_pl,
                            "new_req_id": new_req_id,
                        })
                        # Continue the while loop — next tokens come from new request
                        first_token = True  # will get new prefill timing

                    else:
                        # Uncertainty detected but recall found nothing
                        self._mid_gen_events.append({
                            "event": "recall_empty",
                            "token_count": gen_count,
                            "generated_preview": generated_text[-100:],
                        })

        if stats:
            stats.t_decode_end = time.perf_counter()
            stats.generated_tokens += gen_count
            stats.rope_positions = pl + gen_count


    def _do_generate_streaming_hf(self, prompt, stats: SessionStats = None, messages=None, sampling=None):
        """Token-by-token generation yielding decoded text chunks."""
        import torch
        sp = sampling or {}
        temp = max(sp.get("temperature", 0.7), 0.01)  # avoid div by zero
        rep_pen = sp.get("repetition_penalty", 1.0)
        pres_pen = sp.get("presence_penalty", 1.5)
        max_tok = sp.get("max_tokens") or self.max_new_tokens
        greedy = sp.get("greedy", False)
        # Combine presence_penalty into effective repetition penalty
        effective_rep = rep_pen
        if pres_pen and pres_pen != 0:
            effective_rep = max(rep_pen + (pres_pen - 1.0) * 0.1, 1.0)
        with torch.inference_mode():
            inputs = self._prepare_inputs(prompt, messages)
            input_ids = inputs["input_ids"]
            pl = input_ids.shape[1]
            if pl > self.window_size:
                input_ids = input_ids[:, -self.window_size:]
                inputs["input_ids"] = input_ids
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -self.window_size:]
                if stats: stats.truncated = True
                pl = input_ids.shape[1]

            if stats:
                stats.prompt_tokens = pl
                stats.rope_positions = pl
                stats.t_prefill_start = time.perf_counter()

            # Prefill  -  pass all inputs including pixel_values etc.
            try:
                prefill_kwargs = {k: v for k, v in inputs.items()}
                prefill_kwargs["use_cache"] = True
                prefill_kwargs["return_dict"] = True
                out = self.model(**prefill_kwargs)
            except Exception as e:
                yield (f"[Error: {e}]", -1)
                return

            if stats:
                stats.t_prefill_end = time.perf_counter()
                stats.t_decode_start = time.perf_counter()

            past = out.past_key_values
            logits = out.logits[:, -1, :]
            gen = []
            # Incremental decode: track what we've already decoded to avoid
            # garbled multi-byte characters from single-token decode
            decoded_so_far = ""

            for step in range(max_tok):
                if gen and effective_rep > 1.0:
                    for t in set(gen[-256:]):
                        if logits[0,t] > 0: logits[0,t] /= effective_rep
                        else: logits[0,t] *= effective_rep
                if greedy:
                    nt = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temp
                    probs = torch.softmax(logits, dim=-1)
                    nt = torch.multinomial(probs, num_samples=1)
                tid = nt.item()
                gen.append(tid)
                if tid == self.tokenizer.eos_token_id: break

                # Decode all generated tokens so far, emit only the new text
                full_decoded = self.tokenizer.decode(gen, skip_special_tokens=True)
                new_text = full_decoded[len(decoded_so_far):]
                decoded_so_far = full_decoded
                if new_text:
                    # Hold back trailing replacement chars (incomplete multi-byte)
                    safe = new_text
                    while safe and safe[-1] == '\ufffd':
                        safe = safe[:-1]
                    if safe:
                        yield (safe, tid)

                try:
                    step_out = self.model(input_ids=nt.view(1,1), past_key_values=past,
                                          use_cache=True, return_dict=True)
                except:
                    break
                past = step_out.past_key_values
                logits = step_out.logits[:, -1, :]

            if stats:
                stats.t_decode_end = time.perf_counter()
                stats.generated_tokens += len(gen)
                stats.rope_positions = pl + len(gen)

    def _manual_decode(self, input_ids, prompt_len):
        """Fallback token-by-token decode for TP mode."""
        import torch
        try:
            out = self.model(input_ids=input_ids, use_cache=True, return_dict=True)
        except Exception as e:
            logger.error(f"Prefill failed: {e}")
            return []
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        gen = []
        for step in range(self.max_new_tokens):
            if gen:
                for t in set(gen[-256:]):
                    if logits[0,t] > 0: logits[0,t] /= 1.15
                    else: logits[0,t] *= 1.15
            logits = logits / 0.7
            probs = torch.softmax(logits, dim=-1)
            nt = torch.multinomial(probs, num_samples=1)
            tid = nt.item()
            gen.append(tid)
            if tid == self.tokenizer.eos_token_id: break
            try:
                so = self.model(input_ids=nt.view(1,1), past_key_values=past, use_cache=True, return_dict=True)
            except: break
            past = so.past_key_values
            logits = so.logits[:, -1, :]
        return gen

    def _sse_chunk(self, content, finish_reason=None, req_id=None):
        rid = req_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"
        delta = {}
        if content is not None: delta["content"] = content
        if finish_reason: delta["finish_reason"] = finish_reason
        chunk = {"id":rid,"object":"chat.completion.chunk","created":int(time.time()),
                 "model":self.served_model_name,
                 "choices":[{"index":0,"delta":delta,
                             "finish_reason":finish_reason}]}
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    def _sse_tool_call_chunk(self, tool_name, tool_args, tool_index=0, req_id=None):
        """Emit a tool call in OpenAI streaming format."""
        rid = req_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        args_str = json.dumps(tool_args, ensure_ascii=False) if isinstance(tool_args, dict) else str(tool_args)
        delta = {
            "tool_calls": [{
                "index": tool_index,
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": args_str
                }
            }]
        }
        chunk = {"id":rid,"object":"chat.completion.chunk","created":int(time.time()),
                 "model":self.served_model_name,
                 "choices":[{"index":0,"delta":delta,"finish_reason":None}]}
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    def _ok_response(self, content, pt, ct, tool_calls=None):
        msg = {"role":"assistant","content":content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return {"id":f"chatcmpl-{uuid.uuid4().hex[:12]}","object":"chat.completion",
                "created":int(time.time()),"model":self.served_model_name,
                "choices":[{"index":0,"message":msg,
                            "finish_reason":"tool_calls" if tool_calls else "stop"}],
                "usage":{"prompt_tokens":pt,"completion_tokens":ct,"total_tokens":pt+ct},
                "kessie":self.cache.get_stats(), "batch":self.batch_mgr.stats}

    def _error_response(self, msg):
        return {"error":{"message":msg,"type":"server_error","code":503}}


# =============================================================================
# HTTP Server  -  threaded, up to 32 concurrent
# =============================================================================

_engine: Optional[LibrarianEngine] = None

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    request_queue_size = 64
    timeout = 300

class Handler(BaseHTTPRequestHandler):
    timeout = 300
    protocol_version = "HTTP/1.1"
    def log_message(self, fmt, *args): logger.debug(f"HTTP: {fmt % args}")

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(body)))
        self.send_header("Connection","close")
        self.send_header("Access-Control-Allow-Origin","*")
        self.end_headers(); self.wfile.write(body); self.wfile.flush()

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type","text/event-stream")
        self.send_header("Cache-Control","no-cache")
        self.send_header("Connection","keep-alive")
        self.send_header("Access-Control-Allow-Origin","*")
        self.end_headers()

    def _sse_write(self, data):
        try:
            self.wfile.write(data.encode() if isinstance(data, str) else data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers","*")
        self.send_header("Connection","close")
        self.end_headers()

    def do_GET(self):
        if self.path == "/v1/models":
            self._json({"object":"list","data":[{
                "id":_engine.served_model_name,
                "object":"model",
                "max_model_len": _engine.kessie_cache_size,
                "window": _engine.window_size,
            }]})
        elif self.path == "/health":
            self._json({"status":"ok","knowledge":_engine.store.count(),
                         "kessie":_engine.cache.get_stats(),"batch":_engine.batch_mgr.stats,
                         "conversations": _engine.conv_mgr.stats,
                         "faiss_pending":_engine.store.indexer.pending,
                         "faiss_vectors":_engine.store.vindex.count})
        elif self.path == "/v1/knowledge/topics":
            self._json({"topics":_engine.store.list_topics()})
        else:
            self._json({"error":"not found"},404)

    def do_POST(self):
        try:
            cl = int(self.headers.get("Content-Length",0))
            body = self.rfile.read(cl) if cl > 0 else b""
            req = json.loads(body) if body else {}
        except Exception as e:
            self._json({"error":str(e)},400); return

        if self.path == "/v1/chat/completions":
            t0 = time.perf_counter()
            msgs = req.get("messages",[])
            do_stream = req.get("stream", False)
            client_tools = req.get("tools", None)
            conv_id = req.get("conversation_id", "default")
            # Sampling parameters from request
            sampling = {
                "temperature": req.get("temperature", 0.7),
                "top_p": req.get("top_p", 0.8),
                "top_k": req.get("top_k", 20),
                "repetition_penalty": req.get("repetition_penalty", 1.0),
                "presence_penalty": req.get("presence_penalty", 1.5),
                "max_tokens": req.get("max_tokens") or req.get("out_seq_length"),
                "greedy": req.get("greedy", False),
            }

            # Acquire conversation cache (blocks if queue full)
            try:
                conv_cache = _engine.conv_mgr.acquire(conv_id)
            except TimeoutError as e:
                self._json({"error": str(e)}, 503); return

            # Swap engine cache to this conversation's cache
            original_cache = _engine.cache
            _engine.cache = conv_cache

            try:
                if do_stream:
                    self._sse_start()
                    try:
                        for chunk in _engine.generate_stream(msgs, extra_tools=client_tools, sampling=sampling):
                            self._sse_write(chunk)
                    except Exception as e:
                        traceback.print_exc()
                        self._sse_write(f"data: {json.dumps({'error':str(e)})}\n\n")
                    logger.info(f"Stream [{conv_id[:8]}]: {(time.perf_counter()-t0)*1000:.0f}ms")
                else:
                    try:
                        result = _engine.generate(msgs, extra_tools=client_tools, sampling=sampling)
                        logger.info(f"Generate [{conv_id[:8]}]: {(time.perf_counter()-t0)*1000:.0f}ms")
                        self._json(result)
                    except Exception as e:
                        traceback.print_exc()
                        try: self._json({"error":str(e)},500)
                        except: pass
            finally:
                # Always restore and release
                _engine.cache = original_cache
                _engine.conv_mgr.release(conv_id)

        elif self.path == "/v1/conversations/end":
            conv_id = req.get("conversation_id")
            if not conv_id:
                self._json({"error": "conversation_id required"}, 400); return
            _engine.conv_mgr.end_conversation(conv_id)
            self._json({"status": "ended", "conversation_id": conv_id})

        elif self.path == "/v1/conversations/list":
            self._json({
                "conversations": _engine.conv_mgr.list_conversations(),
                "stats": _engine.conv_mgr.stats,
            })

        elif self.path == "/v1/knowledge/store":
            self._json(_engine.store.store(
                req.get("topic","general"),req.get("key",""),req.get("value",""),req.get("source","api")))
        elif self.path == "/v1/knowledge/search":
            r = _engine.store.retrieve(req.get("query",""),req.get("topic"),req.get("limit",10))
            self._json({"results":r,"count":len(r)})
        else:
            self._json({"error":"not found"},404)


# =============================================================================
# CLI
# =============================================================================

def run_chat(engine):
    print(f"\n{'='*60}")
    print(f"KeSSie Experiment 3  -  KeSSie Engine")
    print(f"Model: {engine.model_name} | Window: {engine.window_size} | Knowledge: {engine.store.count()}")
    print(f"{'='*60}")
    print("Commands: /topics /store <t> <k> <v> /search <q> /stats /quit\n")
    messages = []
    while True:
        try: user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt): print("\nBye."); break
        if not user: continue
        if user == "/quit": break
        if user == "/topics":
            for t in engine.store.list_topics(): print(f"  {t['topic']}: {t['count']}"); continue
        if user == "/stats":
            for k,v in engine.cache.get_stats().items(): print(f"  {k}: {v}")
            print(f"  knowledge: {engine.store.count()}")
            print(f"  faiss_vectors: {engine.store.vindex.count}")
            b = engine.batch_mgr.stats
            print(f"  batch: active={b['active']} total={b['total_requests']}"); continue
        if user.startswith("/store "):
            p = user[7:].split(" ", 2)
            if len(p) < 3: print("  Usage: /store <topic> <key> <value>"); continue
            r = engine.store.store(p[0], p[1], p[2], source="cli")
            print(f"  Stored: [{p[0]}] {p[1]} v{r['version']}"); continue
        if user.startswith("/search "):
            for r in engine.store.retrieve(user[8:]): print(f"  [{r['topic']}] {r['key']}: {r['value'][:120]}")
            continue
        messages.append({"role":"user","content":user})
        result = engine.generate(messages)
        if "error" in result:
            print(f"\nError: {result['error']}"); continue
        reply = result["choices"][0]["message"]["content"]
        print(f"\nKeSSie> {reply}\n")
        messages.append({"role":"assistant","content":reply})


def _seed_reference_data(store):
    entries = [
        ("math","quadratic_formula","x = (-b +/- sqrt(b^2-4ac)) / 2a"),
        ("math","pythagorean_theorem","a^2 + b^2 = c^2"),
        ("math","euler_identity","e^(ipi) + 1 = 0"),
        ("math","derivative_rules","Power: d/dx[x^n]=nx^(n-1). Product: (fg)'=f'g+fg'. Chain: f(g(x))'=f'(g(x))g'(x)"),
        ("math","integral_rules","integralx^n dx = x^(n+1)/(n+1)+C. By parts: integralu dv = uv - integralv du"),
        ("math","trig_identities","sin^2theta+cos^2theta=1. sin(2theta)=2sinthetacostheta. tan=sin/cos"),
        ("logic","modus_ponens","If P then Q. P is true. Therefore Q. [(P->Q)ANDP]->Q"),
        ("logic","modus_tollens","If P then Q. Q is false. Therefore P is false. [(P->Q)ANDNOTQ]->NOTP"),
        ("logic","de_morgans","NOT(AANDB)=NOTAORNOTB. NOT(AORB)=NOTAANDNOTB"),
        ("logic","syllogism","All A are B. All B are C. Therefore all A are C."),
        ("reasoning","bayes_theorem","P(A|B) = P(B|A).P(A) / P(B)"),
        ("reasoning","occams_razor","Among competing hypotheses, prefer the one with fewest assumptions."),
        ("reasoning","scientific_method","Observe->Hypothesize->Predict->Experiment->Analyze->Conclude->Replicate"),
        ("definitions","algorithm","Finite sequence of well-defined instructions for solving a class of problems."),
        ("definitions","turing_complete","System that can simulate any Turing machine. Requires: conditional branching + arbitrary memory."),
        ("definitions","halting_problem","Undecidable: no general algorithm can determine if an arbitrary program halts on arbitrary input."),
    ]
    for t,k,v in entries: store.store(t,k,v,source="seed")
    print(f"Seeded {len(entries)} reference entries.")


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="KeSSie Exp3  -  KeSSie Engine")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("serve")
    s.add_argument("--model", default="Qwen/Qwen3-0.6B")
    s.add_argument("--served-model-name", default=None)
    s.add_argument("--port", type=int, default=8200)
    s.add_argument("--gpus", type=int, default=4)
    s.add_argument("--dtype", default="bfloat16")
    s.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                   help="Inference backend: hf (transformers) or vllm")
    s.add_argument("--window", type=int, default=4096)
    s.add_argument("--max-generation", type=int, default=4096, help="Max tokens per response")
    s.add_argument("--kv-cache-dtype", default=None, help="KV cache dtype (e.g. fp8_e5m2, fp8_e4m3fn, float16)")
    s.add_argument("--kessie-cache-size", type=int, default=10_000_000,
                   help="Max conversation tokens per KeSSie cache (default: 10M)")
    s.add_argument("--conversation-threads", type=int, default=4,
                   help="Max simultaneous active conversations (default: 4)")
    s.add_argument("--conversation-queue", type=int, default=64,
                   help="Max queued conversations waiting for a thread (default: 64)")
    s.add_argument("--fog-alpha", type=float, default=0.5)
    s.add_argument("--db", default="kessie_knowledge.db")
    s.add_argument("--seed", action="store_true")
    s.add_argument("--force-tp", action="store_true")
    s.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                   help="vLLM GPU memory utilization (0.0-1.0)")
    s.add_argument("--max-model-len", type=int, default=None,
                   help="vLLM max model length (context window)")
    s.add_argument("--enforce-eager", action="store_true",
                   help="vLLM: disable CUDA graphs")
    s.add_argument("--enable-hip-acceleration", action="store_true",
                   help="Use HIP kernels for KV recompute on dedicated KV-GPUs + P2P DMA inject")
    s.add_argument("--kv-gpus", default=None,
                   help="Comma-separated GPU IDs for KV recompute (e.g. 4,5,6,7). Requires --enable-hip-acceleration")

    c = sub.add_parser("chat")
    c.add_argument("--model", default="Qwen/Qwen3-0.6B")
    c.add_argument("--served-model-name", default=None)
    c.add_argument("--gpus", type=int, default=4)
    c.add_argument("--dtype", default="bfloat16")
    c.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    c.add_argument("--window", type=int, default=4096)
    c.add_argument("--max-generation", type=int, default=4096)
    c.add_argument("--kv-cache-dtype", default=None)
    c.add_argument("--kessie-cache-size", type=int, default=10_000_000)
    c.add_argument("--fog-alpha", type=float, default=0.5)
    c.add_argument("--db", default="kessie_knowledge.db")
    c.add_argument("--seed", action="store_true")
    c.add_argument("--force-tp", action="store_true")
    c.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    c.add_argument("--max-model-len", type=int, default=None)
    c.add_argument("--enforce-eager", action="store_true")
    c.add_argument("--enable-hip-acceleration", action="store_true")
    c.add_argument("--kv-gpus", default=None)

    st = sub.add_parser("store")
    st_sub = st.add_subparsers(dest="store_cmd")
    st_sub.add_parser("list"); st_sub.add_parser("seed")
    sa = st_sub.add_parser("add"); sa.add_argument("--topic",required=True); sa.add_argument("--key",required=True); sa.add_argument("--value",required=True)
    ss = st_sub.add_parser("search"); ss.add_argument("query")
    st.add_argument("--db", default="kessie_knowledge.db")

    args = p.parse_args()

    if args.cmd == "store":
        store = KnowledgeStore(db_path=args.db)
        if args.store_cmd == "list":
            for t in store.list_topics(): print(f"  {t['topic']:20s} {t['count']:>5d}")
        elif args.store_cmd == "seed": _seed_reference_data(store)
        elif args.store_cmd == "add": store.store(args.topic, args.key, args.value, source="cli"); print("Stored.")
        elif args.store_cmd == "search":
            for r in store.retrieve(args.query): print(f"  [{r['topic']}] {r['key']}: {r['value'][:120]}")
        return

    if args.cmd in ("serve","chat"):
        global _engine
        _engine = LibrarianEngine(
            model_name=args.model, num_gpus=args.gpus, dtype=args.dtype,
            max_new_tokens=getattr(args, 'max_generation', 0) or None,
            window_size=args.window, fog_alpha=args.fog_alpha, db_path=args.db,
            served_model_name=getattr(args, 'served_model_name', None),
            force_tp=getattr(args, 'force_tp', False),
            kv_cache_dtype=getattr(args, 'kv_cache_dtype', None),
            backend=getattr(args, 'backend', 'hf'),
            gpu_memory_utilization=getattr(args, 'gpu_memory_utilization', 0.90),
            max_model_len=getattr(args, 'max_model_len', None),
            enforce_eager=getattr(args, 'enforce_eager', False),
            enable_hip_acceleration=getattr(args, 'enable_hip_acceleration', False),
            kv_gpus=getattr(args, 'kv_gpus', None),
            kessie_cache_size=getattr(args, 'kessie_cache_size', 10_000_000),
            conversation_threads=getattr(args, 'conversation_threads', 4),
            conversation_queue=getattr(args, 'conversation_queue', 64))
        _engine.load()
        if getattr(args,'seed',False): _seed_reference_data(_engine.store)

        if args.cmd == "serve":
            srv = ThreadedHTTPServer(("127.0.0.1",args.port), Handler)
            print(f"\n{'='*60}")
            print(f"KeSSie Experiment 3  -  KeSSie Engine")
            print(f"  Model:     {args.model} ({args.gpus} GPUs, {_engine._model_kind}, {_engine.backend})")
            if _engine.served_model_name != args.model:
                print(f"  Served as: {_engine.served_model_name}")
            print(f"  Window:    {args.window} tokens (fog alpha={args.fog_alpha})")
            print(f"  Max gen:   {_engine.max_new_tokens} tokens")
            if _engine.backend == "vllm":
                attn = "KeSSie fog-of-war" if getattr(_engine, '_kessie_attention_active', False) else "default"
                kv = "KeSSie lossless recall" if getattr(_engine, '_kessie_kv_active', False) else "prefix only"
                hip = f" [GPU accel: KV-GPUs {_engine.kv_gpus}]" if _engine.enable_hip_acceleration else ""
                print(f"  vLLM:      async streaming, attention={attn}")
                print(f"             KV={kv}{hip}, gpu_util={_engine.gpu_memory_utilization}")
                if _engine.kv_cache_dtype_str:
                    print(f"  KV cache:  {_engine.kv_cache_dtype_str} (vLLM managed)")
                else:
                    print(f"  KV cache:  auto (vLLM managed)")
            else:
                kv_info = str(_engine._kv_cache_dtype) if _engine._kv_cache_dtype else "same as model"
                print(f"  KV cache:  {kv_info} (KeSSie managed, GPU<->CPU tiered)")
            print(f"  Memory:    semantic index + knowledge store + auto-recall")
            print(f"  KeSSie:    {_engine.kessie_cache_size:,} tokens/conversation cache")
            ct = getattr(args, 'conversation_threads', 4)
            cq = getattr(args, 'conversation_queue', 64)
            print(f"  Threads:   {ct} active conversations, {cq} queued, evict-on-close")
            print(f"  Knowledge: {_engine.store.count()} entries ({_engine.store.vindex.count} vectors)")
            print(f"  FAISS:     {'native' if HAS_FAISS else 'numpy fallback'}")
            print(f"  Concurrency: 32 max concurrent requests")
            print(f"  Streaming: SSE on stream:true")
            print(f"  API:       http://127.0.0.1:{args.port}/v1/chat/completions")
            print(f"{'='*60}\n")
            try: srv.serve_forever()
            except KeyboardInterrupt: srv.shutdown()
        else:
            run_chat(_engine)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
