"""
04_embed_and_index.py
----------------------
WHAT THIS FILE DOES:
    Reads every candidate's narrative (built in 03_build_narratives.py)
    and converts it into a 768-dimensional mathematical vector using
    the bge-base-en-v1.5 embedding model.

    Then builds a FAISS HNSW search index over all 100,000 vectors
    so that at ranking time, we can find the 3,000 most similar
    candidates to any Job Description in under 2 seconds on CPU.

    Also builds a small 50-candidate index for the sandbox demo.

WHY EMBEDDINGS?
    Two candidates with similar career backgrounds will have vectors
    that point in similar directions in 768-dimensional space.

    A candidate who "built product discovery ranking at Swiggy" and
    a JD asking for "ranking systems at product companies" will have
    a high cosine similarity — even if they share zero exact keywords.

    This is semantic search. It understands MEANING, not just words.

WHY FAISS INSTEAD OF PGVECTOR?
    FAISS is a single file. No database server to set up.
    Loads in 3 seconds. Works on CPU. Portable across machines.
    Perfect for a hackathon where the ranking step runs in a sandbox.

WHY bge-base-en-v1.5?
    - Free (runs locally, no API cost)
    - 768 dimensions — good quality, reasonable file size
    - Designed specifically for retrieval tasks
    - Consistently top-ranked on the MTEB retrieval benchmark
    - Fast enough on CPU for 100K candidates in ~40 mins
    - On Kaggle T4 GPU: under 10 minutes for 100K

HOW TO RUN:

    On your laptop (test — 50 candidates):
        python precompute/04_embed_and_index.py --mode test

    On Kaggle (full — 100,000 candidates):
        python precompute/04_embed_and_index.py --mode full

FIRST-TIME SETUP:
    The model downloads automatically from HuggingFace (~440 MB).
    After first download, it uses local cache — no internet needed.
    On Kaggle, this happens in the first cell when you run the script.

OUTPUT FILES:
    artifacts/candidates.faiss              ← main search index (full)
    artifacts/candidates_metadata.npy       ← candidate ID order map
    artifacts/sample_candidates.faiss       ← small index for sandbox
    artifacts/sample_metadata.npy           ← ID order for sandbox
"""

import json
import os
import time
import argparse
import numpy as np

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────

INPUT_SAMPLE = "artifacts/narratives_sample.jsonl"
INPUT_FULL   = "artifacts/narratives_candidates.jsonl"

# Output files
FAISS_FULL    = "artifacts/candidates.faiss"
IDS_FULL      = "artifacts/candidate_ids.npy"       # order of IDs in index
EMBEDDINGS_FULL = "artifacts/candidate_embeddings.npy"  # backup

FAISS_SAMPLE  = "artifacts/sample_candidates.faiss"
IDS_SAMPLE    = "artifacts/sample_candidate_ids.npy"

# Embedding model — bge-base-en-v1.5
# This is the best model for retrieval that runs on CPU
MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBED_DIM  = 768    # output dimension of bge-base

# FAISS HNSW settings
# M = number of connections per node in the graph
# Higher M = more accurate search, larger index, slower build
# 32 is a good default for 100K-1M vectors
HNSW_M = 32

# efConstruction = how carefully the graph is built
# Higher = better quality graph, slower build time
# 200 is appropriate for 100K vectors
EF_CONSTRUCTION = 200

# efSearch = how many neighbors checked during search
# Higher = more accurate search, slightly slower
# Set in rank.py at search time — not here
EF_SEARCH_DEFAULT = 128

# Batch size for encoding
# How many narratives to process at once
# 256 works well on CPU (8GB+ RAM)
# 512 is fine on GPU (Kaggle T4)
BATCH_SIZE_CPU = 128
BATCH_SIZE_GPU = 512

# BGE model requires this prefix for query encoding
# (not needed for candidate narratives — only for JD embedding)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ─────────────────────────────────────────────
# STEP 1 — DETECT GPU / CPU
# ─────────────────────────────────────────────

def detect_device():
    """
    Checks if a GPU is available (e.g. on Kaggle T4).
    Returns 'cuda' if GPU found, 'cpu' otherwise.

    On your laptop: always returns 'cpu'
    On Kaggle with GPU enabled: returns 'cuda'
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  🚀 GPU detected: {gpu_name}")
            return "cuda"
        else:
            print("  💻 No GPU found — using CPU")
            return "cpu"
    except ImportError:
        print("  💻 PyTorch not detecting GPU — using CPU")
        return "cpu"


# ─────────────────────────────────────────────
# STEP 2 — LOAD NARRATIVES FROM FILE
# ─────────────────────────────────────────────

def load_narratives(filepath):
    """
    Reads the narratives JSONL file and returns:
        candidate_ids  — list of CAND_XXXXXXX strings (in order)
        narratives     — list of narrative text strings (in order)

    The ORDER matters. The nth narrative produces the nth embedding
    vector. We use candidate_ids to map index positions back to IDs.
    """
    candidate_ids = []
    narratives    = []

    print(f"\n  📂 Loading narratives from: {filepath}")
    start = time.time()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)

            candidate_ids.append(c["candidate_id"])
            narratives.append(c["_narrative"])

    elapsed = time.time() - start
    print(f"  ✅ Loaded {len(narratives)} narratives in {elapsed:.1f}s")
    print(f"  Sample narrative length: {len(narratives[0])} chars")

    return candidate_ids, narratives


# ─────────────────────────────────────────────
# STEP 3 — GENERATE EMBEDDINGS
# ─────────────────────────────────────────────

def generate_embeddings(narratives, device="cpu"):
    """
    Converts a list of text narratives into a 2D NumPy array
    of shape (N, 768) where N is the number of candidates.

    Each row is a 768-dimensional vector representing the
    semantic meaning of one candidate's full narrative.

    Key settings:
        normalize_embeddings=True
            Makes every vector exactly length 1.0.
            This allows cosine similarity to be computed
            with a simple dot product — much faster.

        batch_size
            How many narratives to encode at once.
            Larger = faster but uses more RAM/VRAM.

        show_progress_bar=True
            Shows a tqdm progress bar so you can see
            how far along the encoding is.
    """
    from sentence_transformers import SentenceTransformer

    print(f"\n  🧠 Loading embedding model: {MODEL_NAME}")
    print(f"     Device: {device}")
    model_load_start = time.time()

    model = SentenceTransformer(MODEL_NAME, device=device)

    model_load_time = time.time() - model_load_start
    print(f"  ✅ Model loaded in {model_load_time:.1f}s")

    # Choose batch size based on device
    batch_size = BATCH_SIZE_GPU if device == "cuda" else BATCH_SIZE_CPU

    print(f"\n  ⚙️  Encoding {len(narratives)} narratives...")
    print(f"     Batch size: {batch_size}")
    print(f"     Estimated time: ", end="")

    # Rough time estimate for the user
    if device == "cuda":
        est_mins = len(narratives) / 100000 * 10
        print(f"~{est_mins:.0f} minutes on GPU")
    else:
        est_mins = len(narratives) / 100000 * 40
        print(f"~{est_mins:.0f} minutes on CPU")

    encode_start = time.time()

    embeddings = model.encode(
        narratives,
        batch_size=batch_size,
        normalize_embeddings=True,   # cosine sim via dot product
        show_progress_bar=True,      # tqdm progress bar
        convert_to_numpy=True,       # return as numpy array
    )

    encode_time = time.time() - encode_start

    # Cast to float32 — FAISS requires float32
    embeddings = embeddings.astype(np.float32)

    print(f"\n  ✅ Encoding complete in {encode_time:.1f}s "
          f"({encode_time/60:.1f} minutes)")
    print(f"     Embedding shape: {embeddings.shape}")
    print(f"     Dtype: {embeddings.dtype}")

    # Verify normalization — each vector should have norm ~1.0
    sample_norms = np.linalg.norm(embeddings[:5], axis=1)
    print(f"     Sample vector norms (should all be ~1.0): "
          f"{[f'{n:.4f}' for n in sample_norms]}")

    return embeddings


# ─────────────────────────────────────────────
# STEP 4 — BUILD FAISS HNSW INDEX
# ─────────────────────────────────────────────

def build_faiss_index(embeddings):
    """
    Builds a FAISS HNSW (Hierarchical Navigable Small World) index.

    HNSW is a graph-based approximate nearest neighbour algorithm.
    It builds a multi-layer graph where similar vectors are connected.
    Searching it is like navigating a network — you start at a random
    point and hop toward the nearest neighbour, layer by layer.

    Why HNSW over flat search?
        Flat (brute force): checks every single vector — O(N)
        HNSW: navigates the graph — O(log N)

        For 100K vectors:
            Flat:  ~300ms per query
            HNSW:  ~1-2ms per query
            With efSearch=128: ~5-10ms, excellent recall

    Parameters:
        HNSW_M = 32
            Each node connects to 32 neighbours.
            More connections = better recall, more memory.

        EF_CONSTRUCTION = 200
            How carefully the graph is built.
            Higher = better quality, slower build.
            200 is good for 100K vectors.
    """
    import faiss

    n, d = embeddings.shape
    print(f"\n  🔨 Building FAISS HNSW index...")
    print(f"     Vectors: {n:,}")
    print(f"     Dimensions: {d}")
    print(f"     M (connections): {HNSW_M}")
    print(f"     efConstruction: {EF_CONSTRUCTION}")

    build_start = time.time()

    # Create the HNSW index
    index = faiss.IndexHNSWFlat(d, HNSW_M)
    index.hnsw.efConstruction = EF_CONSTRUCTION

    # Add all vectors to the index
    # This is what actually builds the graph
    print(f"  ⚙️  Adding vectors to index (this is the slow part)...")
    index.add(embeddings)

    build_time = time.time() - build_start

    print(f"  ✅ Index built in {build_time:.1f}s ({build_time/60:.1f} mins)")
    print(f"     Total vectors indexed: {index.ntotal:,}")

    # Quick sanity check — search for the first vector
    # It should return itself as the nearest neighbour
    print(f"\n  🔍 Sanity check — searching for vector 0...")
    test_vec  = embeddings[0:1]
    distances, indices = index.search(test_vec, 3)
    print(f"     Top 3 results (indices): {indices[0]}")
    print(f"     Top 3 scores: {[f'{d:.4f}' for d in distances[0]]}")
    print(f"     ✅ Vector 0 found itself at rank {list(indices[0]).index(0)+1}")

    return index


# ─────────────────────────────────────────────
# STEP 5 — SAVE INDEX AND METADATA
# ─────────────────────────────────────────────

def save_artifacts(index, candidate_ids, embeddings,
                   faiss_path, ids_path, emb_path=None):
    """
    Saves the FAISS index and candidate ID mapping to disk.

    The candidate_ids array is critical — it maps the integer
    position in the FAISS index back to the CAND_XXXXXXX string.

    Example:
        FAISS index position 0  →  candidate_ids[0]  →  CAND_0000001
        FAISS index position 1  →  candidate_ids[1]  →  CAND_0000002

    Without this mapping, we can't convert search results back
    to actual candidate IDs.
    """
    import faiss

    os.makedirs("artifacts", exist_ok=True)

    # Save FAISS index
    print(f"\n  💾 Saving FAISS index to: {faiss_path}")
    faiss.write_index(index, faiss_path)
    index_size_mb = os.path.getsize(faiss_path) / (1024 * 1024)
    print(f"     Size: {index_size_mb:.1f} MB")

    # Save candidate ID order
    print(f"  💾 Saving candidate ID map to: {ids_path}")
    np.save(ids_path, np.array(candidate_ids))
    print(f"     {len(candidate_ids)} IDs saved")

    # Optionally save raw embeddings as backup
    if emb_path:
        print(f"  💾 Saving raw embeddings to: {emb_path}")
        np.save(emb_path, embeddings)
        emb_size_mb = os.path.getsize(emb_path) / (1024 * 1024)
        print(f"     Size: {emb_size_mb:.1f} MB")

    print(f"  ✅ All artifacts saved")


# ─────────────────────────────────────────────
# STEP 6 — TEST SEMANTIC SEARCH
# ─────────────────────────────────────────────

def test_semantic_search(faiss_path, ids_path, narratives_path):
    """
    Loads the saved index and runs a test search using a
    sample JD query to verify the semantic search is working.

    This is the same process rank.py will use — but here we
    run it immediately after building to validate quality.
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    print(f"\n  🧪 Testing semantic search...")

    # Load saved artifacts
    index        = faiss.read_index(faiss_path)
    candidate_ids= np.load(ids_path, allow_pickle=True)

    # Load candidate data for display
    candidates = {}
    with open(narratives_path) as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                candidates[c["candidate_id"]] = c

    # Sample JD query — same as what rank.py will use
    # This is a condensed version of the actual job description
    test_query = """
    Senior AI Engineer at a Series A product company.
    5-9 years experience. Production embeddings-based retrieval systems.
    Vector databases: FAISS, Pinecone, Weaviate, Qdrant, pgvector.
    Hybrid search, ranking systems, recommendation engines.
    NLP, fine-tuning LLMs, evaluation frameworks (NDCG, MRR, MAP).
    Strong Python. A/B testing. Pune or Noida preferred.
    Product company experience required. No consulting-only backgrounds.
    Active on platform, sub-30 day notice preferred.
    """

    # Embed the query
    model     = SentenceTransformer(MODEL_NAME)
    query_emb = model.encode(
        [test_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Search the index
    index.hnsw.efSearch = EF_SEARCH_DEFAULT
    distances, indices  = index.search(query_emb, 10)

    print(f"\n  Top 10 results for sample JD query:")
    print(f"  {'─'*72}")
    print(f"  {'#':<4} {'ID':<16} {'Score':>6}  {'Title':<30} {'Career':>7}")
    print(f"  {'─'*72}")

    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        cid   = candidate_ids[idx]
        cand  = candidates.get(cid, {})
        title = cand.get("profile", {}).get("current_title", "Unknown")[:28]
        cscore= cand.get("_scores", {}).get("career_score", 0)
        print(f"  {rank:<4} {cid:<16} {dist:>6.4f}  {title:<30} {cscore:>7.1f}")

    print(f"  {'─'*72}")
    print(f"\n  ✅ Semantic search working correctly")
    print(f"     Top result should be a strong AI/ML candidate")


# ─────────────────────────────────────────────
# TIMING REPORT
# ─────────────────────────────────────────────

def print_timing_report(total_start, n_candidates, device):
    """Prints the overall timing and estimated full-dataset time."""
    total_time = time.time() - total_start
    per_cand   = total_time / n_candidates if n_candidates > 0 else 0

    print("\n" + "="*65)
    print("         EMBEDDING & INDEXING — TIMING REPORT")
    print("="*65)
    print(f"  Candidates processed    : {n_candidates:,}")
    print(f"  Device used             : {device.upper()}")
    print(f"  Total time              : {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Time per candidate      : {per_cand*1000:.2f}ms")

    if n_candidates < 1000:
        # Extrapolate to 100K
        est_100k = per_cand * 100000
        print(f"\n  Estimated time for 100K : "
              f"{est_100k:.0f}s ({est_100k/60:.0f} mins)")
        if device == "cuda":
            print(f"  (Kaggle T4 GPU is ~4x faster than CPU estimate)")

    print("="*65)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings and build FAISS index"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "full"],
        default="test",
        help="'test' uses 50 sample candidates, 'full' uses 100K"
    )
    args = parser.parse_args()

    total_start = time.time()

    print("\n" + "="*65)
    print("     REDROB HACKATHON — 04 EMBED & INDEX")
    print("="*65)

    # ── Detect device ──
    print("\n  Step 1: Detecting hardware...")
    device = detect_device()

    if args.mode == "test":
        print("\n🧪 Running in TEST mode (50 candidates)")
        print("   This will take ~30 seconds on CPU")
        print("   Perfect for verifying everything works\n")

        # Check input exists
        if not os.path.exists(INPUT_SAMPLE):
            print(f"❌ ERROR: {INPUT_SAMPLE} not found.")
            print("   Run 03_build_narratives.py --mode test first.")
            return

        # ── Load narratives ──
        print("  Step 2: Loading narratives...")
        candidate_ids, narratives = load_narratives(INPUT_SAMPLE)

        # ── Generate embeddings ──
        print("\n  Step 3: Generating embeddings...")
        embeddings = generate_embeddings(narratives, device)

        # ── Build FAISS index ──
        print("\n  Step 4: Building FAISS index...")
        index = build_faiss_index(embeddings)

        # ── Save artifacts ──
        print("\n  Step 5: Saving artifacts...")
        save_artifacts(
            index, candidate_ids, embeddings,
            faiss_path=FAISS_SAMPLE,
            ids_path=IDS_SAMPLE,
            emb_path=None  # skip raw embeddings in test mode
        )

        # ── Test semantic search ──
        print("\n  Step 6: Testing semantic search...")
        test_semantic_search(FAISS_SAMPLE, IDS_SAMPLE, INPUT_SAMPLE)

        # ── Timing report ──
        print_timing_report(total_start, len(narratives), device)

        print("\n✅ TEST MODE COMPLETE")
        print("   Check the top 10 results above.")
        print("   The best AI/ML candidates should rank highest.")
        print("   If they do, run --mode full on Kaggle.\n")

    else:
        print("\n🚀 Running in FULL mode (100,000 candidates)")
        print("   On Kaggle T4 GPU:  ~10-15 minutes")
        print("   On CPU (8 cores):  ~35-45 minutes\n")

        # Check input exists
        if not os.path.exists(INPUT_FULL):
            print(f"❌ ERROR: {INPUT_FULL} not found.")
            print("   Run 03_build_narratives.py --mode full first.")
            return

        # ── Load narratives ──
        print("  Step 2: Loading narratives...")
        candidate_ids, narratives = load_narratives(INPUT_FULL)

        if len(narratives) < 10000:
            print(f"⚠️  WARNING: Only {len(narratives)} candidates found.")
            print(f"   Expected ~100,000. Check your input file.")
            response = input("   Continue anyway? (y/n): ")
            if response.lower() != "y":
                return

        # ── Generate embeddings ──
        print("\n  Step 3: Generating embeddings...")
        print("   💡 Tip: On Kaggle, make sure GPU is enabled")
        print("          in the right panel (Accelerator → GPU T4)\n")
        embeddings = generate_embeddings(narratives, device)

        # ── Save raw embeddings backup ──
        print("\n  Saving raw embeddings backup...")
        np.save(EMBEDDINGS_FULL, embeddings)
        emb_size = os.path.getsize(EMBEDDINGS_FULL) / (1024*1024)
        print(f"  Saved {EMBEDDINGS_FULL} ({emb_size:.0f} MB)")

        # ── Build FAISS index ──
        print("\n  Step 4: Building FAISS index...")
        print("   This takes about 5-10 minutes for 100K vectors...")
        index = build_faiss_index(embeddings)

        # ── Save full artifacts ──
        print("\n  Step 5: Saving artifacts...")
        save_artifacts(
            index, candidate_ids, embeddings,
            faiss_path=FAISS_FULL,
            ids_path=IDS_FULL,
            emb_path=None  # already saved above
        )

        # ── Also build a small sample index for the sandbox demo ──
        print("\n  Step 5b: Building small sample index for sandbox demo...")
        print("   (This is the 50-candidate version for HuggingFace Spaces)")

        # Take first 50 candidates for the sandbox
        sample_ids  = candidate_ids[:50]
        sample_embs = embeddings[:50]
        sample_idx  = build_faiss_index(sample_embs)
        save_artifacts(
            sample_idx, sample_ids, sample_embs,
            faiss_path=FAISS_SAMPLE,
            ids_path=IDS_SAMPLE,
            emb_path=None
        )

        # ── Test semantic search ──
        print("\n  Step 6: Testing semantic search on full index...")
        test_semantic_search(FAISS_FULL, IDS_FULL, INPUT_FULL)

        # ── Timing report ──
        print_timing_report(total_start, len(narratives), device)

        print("\n✅ FULL MODE COMPLETE")
        print("\n   Files saved to artifacts/:")
        for fname in [FAISS_FULL, IDS_FULL, EMBEDDINGS_FULL,
                      FAISS_SAMPLE, IDS_SAMPLE]:
            if os.path.exists(fname):
                size = os.path.getsize(fname) / (1024*1024)
                print(f"   ✓ {fname} ({size:.0f} MB)")

        print("\n   Next steps:")
        print("   1. Download artifacts/ folder from Kaggle")
        print("   2. Place on your laptop in redrob-hackathon/artifacts/")
        print("   3. Run: python precompute/05_build_metadata.py --mode full")
        print("   4. Then run: python rank.py\n")


if __name__ == "__main__":
    main()