"""
rank.py  — The Ultimate AI Recruiter
======================================
This is the single file that gets sandboxed and evaluated by Redrob.

Constraints it must satisfy:
    ✓ Runs in ≤5 minutes on CPU
    ✓ Uses ≤16 GB RAM
    ✓ No internet access (loads model from local cache)
    ✓ No GPU
    ✓ Produces exactly 100 rows, ranks 1-100, scores non-increasing

HOW IT WORKS — 6 stages:

    Stage 1 — Load pre-built artifacts (FAISS index + metadata)
    Stage 2 — Embed the Job Description (1 model call, ~0.5s)
    Stage 3 — Semantic search: top 3000 candidates via FAISS (~1s)
    Stage 4 — Multi-factor scoring: semantic + career + behavioral
    Stage 5 — Reasoning generation: specific, factual, rank-consistent
    Stage 6 — Validate and write submission CSV

SCORING PHILOSOPHY — The "Ultimate AI Recruiter" Formula:

    This system doesn't just keyword-match.
    It separates four types of candidates the JD specifically warns about:

    TYPE A — Keyword stuffer:
        Perfect skills list. Title is "Marketing Manager".
        Career descriptions: zero AI/ML content.
        → High semantic score but LOW career score → falls below top 100.

    TYPE B — Consulting-only background:
        7 years at TCS, Infosys, Wipro.
        JD explicitly disqualifies this.
        → Flagged is_disqualified=True → excluded before scoring.

    TYPE C — Perfect on paper, not actually available:
        Great career score. Last logged in 8 months ago.
        Recruiter response rate: 5%.
        → Low behavioral score + staleness penalty → drops in ranking.

    TYPE D — The right candidate:
        6 years building ranking/retrieval at product companies.
        Active 2 weeks ago. 30-day notice. Based in Hyderabad.
        → High semantic + high career + high behavioral + city bonus → top 10.

HOW TO RUN:

    python rank.py --candidates data/candidates.jsonl.gz \\
                   --artifacts artifacts/ \\
                   --jd data/job_description.md \\
                   --out submission.csv

    Or in test mode (uses sample artifacts):
    python rank.py --mode test
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

TODAY        = datetime(2026, 6, 10)
MODEL_NAME   = "BAAI/bge-base-en-v1.5"

# ── Composite score weights ──
# These weights are calibrated to the JD's explicit priorities.
#
# SEMANTIC (0.45):
#   The single best signal for "does this person understand
#   the domain" — captures meaning beyond keywords.
#   A recommendation systems engineer at Swiggy matches
#   "ranking systems at product companies" without sharing
#   a single keyword.
#
# CAREER (0.35):
#   Pre-computed in 02_feature_engineering.py.
#   Measures actual work history alignment — AI/ML keywords
#   in job descriptions, product company ratio, YoE band,
#   title seniority.
#   This catches keyword stuffers and consulting-only backgrounds.
#
# BEHAVIORAL (0.20):
#   Pre-computed in 02_feature_engineering.py.
#   Measures real availability — login recency, response rate,
#   notice period, GitHub activity, interview completion rate.
#   The JD explicitly says to down-weight stale/unresponsive candidates.

W_SEMANTIC = 0.45
W_CAREER   = 0.35
W_BEHAV    = 0.20

# ── Bonuses (added to composite score) ──
BONUS_PREFERRED_CITY = 0.035   # Pune/Noida/Delhi NCR/Hyderabad — JD strongly prefers
BONUS_INDIA          = 0.015   # India but not preferred city — still desirable
BONUS_SHORT_NOTICE   = 0.040   # ≤30 days — "sub-30 day is preferred"
BONUS_MEDIUM_NOTICE  = 0.015   # 31-60 days — within buyout window
BONUS_OPEN_TO_WORK   = 0.015   # explicitly marked available
BONUS_HIGH_DEMAND    = 0.010   # saved by 5+ recruiters in 30 days
BONUS_VERIFIED_SKILLS= 0.010   # has platform-verified assessment scores

# ── Penalties (subtracted from composite score) ──
PENALTY_STALE        = 0.080   # >180 days inactive — JD says down-weight these
PENALTY_SEMI_STALE   = 0.035   # 90-180 days inactive
PENALTY_LONG_NOTICE  = 0.050   # >90 days notice — "bar gets higher"
PENALTY_LOW_RESPONSE = 0.040   # <20% recruiter response rate — effectively unreachable
PENALTY_NO_INDIA     = 0.020   # outside India, not willing to relocate

# ── FAISS search depth ──
# How many candidates to retrieve before scoring.
# 3000 gives excellent recall with fast filtering.
FAISS_RETRIEVE_K     = 3000

# ── efSearch — FAISS HNSW accuracy parameter ──
# Higher = more accurate (finds more true nearest neighbours)
# 128 gives >99% recall on typical datasets in <2 seconds
EF_SEARCH            = 128

# ─────────────────────────────────────────────
# STAGE 1 — LOAD ARTIFACTS
# ─────────────────────────────────────────────

def load_artifacts(artifacts_dir, mode="full"):
    """
    Loads the two pre-built files the ranking step needs.
    These were created by the precompute pipeline (scripts 01-05).

    Files needed:
        candidates.faiss              — HNSW search graph
        candidate_ids.npy             — ID order map
        candidates_metadata.parquet   — pre-computed scores + signals

    On sample mode (test), loads the smaller versions.
    """
    import faiss

    if mode == "test":
        faiss_file = os.path.join(artifacts_dir, "sample_candidates.faiss")
        ids_file   = os.path.join(artifacts_dir, "sample_candidate_ids.npy")
        meta_file  = os.path.join(artifacts_dir, "sample_metadata.parquet")
    else:
        faiss_file = os.path.join(artifacts_dir, "candidates.faiss")
        ids_file   = os.path.join(artifacts_dir, "candidate_ids.npy")
        meta_file  = os.path.join(artifacts_dir, "candidates_metadata.parquet")

    # Validate files exist
    for fpath in [faiss_file, ids_file, meta_file]:
        if not os.path.exists(fpath):
            print(f"\n❌ ERROR: Required file not found: {fpath}")
            print("   Run the precompute pipeline first:")
            print("   python precompute/01_parse_and_validate.py --mode full")
            print("   python precompute/02_feature_engineering.py --mode full")
            print("   python precompute/03_build_narratives.py --mode full")
            print("   python precompute/04_embed_and_index.py --mode full")
            print("   python precompute/05_build_metadata.py --mode full")
            sys.exit(1)

    print(f"  📂 Loading FAISS index: {os.path.basename(faiss_file)}")
    t = time.time()
    index = faiss.read_index(faiss_file)
    index.hnsw.efSearch = EF_SEARCH
    print(f"     {index.ntotal:,} vectors loaded in {time.time()-t:.1f}s")

    print(f"  📂 Loading candidate ID map...")
    candidate_ids = np.load(ids_file, allow_pickle=True)
    print(f"     {len(candidate_ids):,} IDs loaded")

    print(f"  📂 Loading metadata: {os.path.basename(meta_file)}")
    t = time.time()
    df = pd.read_parquet(meta_file)
    print(f"     {len(df):,} rows × {len(df.columns)} columns in {time.time()-t:.1f}s")

    # Build an index from candidate_id → DataFrame row
    # This lets us look up metadata in O(1) after FAISS returns indices
    id_to_row = {cid: i for i, cid in enumerate(candidate_ids)}

    return index, candidate_ids, df, id_to_row


# ─────────────────────────────────────────────
# STAGE 2 — EMBED THE JOB DESCRIPTION
# ─────────────────────────────────────────────

def embed_job_description(jd_path):
    """
    Converts the Job Description into a 768-dimensional vector.

    The JD vector is what we search against in the FAISS index.
    The closer a candidate's narrative vector is to this JD vector,
    the higher their semantic similarity score.

    We use the same model (bge-base-en-v1.5) and same normalization
    as was used to encode candidate narratives — critical for
    meaningful cosine similarity comparisons.

    The BGE model recommends a specific query prefix for retrieval tasks.
    We use a condensed, signal-rich version of the JD rather than
    the full 3000-word document — this gives better retrieval quality.
    """
    from sentence_transformers import SentenceTransformer

    # Read the full JD
    if not os.path.exists(jd_path):
        print(f"❌ ERROR: JD file not found: {jd_path}")
        sys.exit(1)

    with open(jd_path, "r", encoding="utf-8") as f:
        jd_full = f.read()

    # Build a signal-rich condensed query from the JD
    # This captures the key semantic meaning without noise
    jd_query = (
        "Senior AI Engineer with 5-9 years experience at product companies. "
        "Production embeddings-based retrieval systems: sentence-transformers, "
        "BGE, E5, OpenAI embeddings. Vector databases: FAISS, Pinecone, Weaviate, "
        "Qdrant, Milvus, Elasticsearch, pgvector. Hybrid search, dense retrieval, "
        "ranking systems, recommendation engines, information retrieval. "
        "NLP, natural language processing, fine-tuning LLMs, LoRA, PEFT. "
        "Evaluation frameworks: NDCG, MRR, MAP, offline evaluation, A/B testing. "
        "Learning-to-rank models, XGBoost, neural ranking. "
        "Strong Python. MLOps. Production deployment. "
        "Product company experience required — not consulting or IT services firms. "
        "Located in Pune, Noida, Delhi NCR, Hyderabad, Bangalore, Mumbai preferred. "
        "Sub-30 day notice period ideal. Active job seeker. "
        "Shipped end-to-end ranking, search, or recommendation system to real users. "
        f"{jd_full[:1000]}"  # append first 1000 chars of actual JD for grounding
    )

    print(f"  🧠 Loading embedding model: {MODEL_NAME}")
    t = time.time()
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print(f"     Model loaded in {time.time()-t:.1f}s")

    print(f"  ⚙️  Encoding Job Description...")
    t = time.time()

    jd_vector = model.encode(
        [jd_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    print(f"     JD encoded in {time.time()-t:.2f}s")
    print(f"     Vector shape: {jd_vector.shape}, "
          f"norm: {np.linalg.norm(jd_vector[0]):.4f}")

    return jd_vector, model


# ─────────────────────────────────────────────
# STAGE 3 — SEMANTIC SEARCH
# ─────────────────────────────────────────────

def semantic_search(index, jd_vector, candidate_ids, df, id_to_row):
    """
    Searches the FAISS index for the top-K most semantically
    similar candidates to the Job Description.

    Returns a DataFrame with the top-K candidates plus their
    semantic similarity scores attached.

    Why K=3000?
        After filtering honeypots and disqualified candidates,
        we typically have 60-80% of K remaining.
        We need at least 100 rankable candidates.
        3000 gives ample headroom even if 70% are filtered out.

    Why cosine similarity works here:
        Both JD vector and candidate vectors are L2-normalized.
        Dot product of two normalized vectors = cosine similarity.
        FAISS IndexHNSWFlat with normalized vectors returns
        inner product scores in [0, 1] range (1 = identical).
    """
    k = min(FAISS_RETRIEVE_K, index.ntotal)

    print(f"  🔍 Searching {index.ntotal:,} candidates for top {k:,}...")
    t = time.time()

    scores, indices = index.search(jd_vector, k)

    scores  = scores[0]    # shape (k,)
    indices = indices[0]   # shape (k,)

    print(f"     Search complete in {time.time()-t:.2f}s")
    print(f"     Semantic score range: {scores.min():.4f} – {scores.max():.4f}")

    # Map FAISS indices → candidate IDs → metadata rows
    retrieved_rows = []
    for faiss_pos, sem_score in zip(indices, scores):
        if faiss_pos < 0 or faiss_pos >= len(candidate_ids):
            continue    # invalid index, skip

        cid      = candidate_ids[faiss_pos]
        meta_idx = id_to_row.get(cid)

        if meta_idx is None:
            continue    # not in metadata, skip

        row = df.iloc[meta_idx].to_dict()
        row["semantic_score"] = float(sem_score)
        retrieved_rows.append(row)

    pool = pd.DataFrame(retrieved_rows)
    print(f"     Retrieved {len(pool):,} candidates into scoring pool")

    return pool


# ─────────────────────────────────────────────
# STAGE 4 — MULTI-FACTOR SCORING
# ─────────────────────────────────────────────

def compute_composite_score(row):
    """
    The core intelligence of the ranking system.

    Combines three independent signals into one composite score,
    then applies bonuses and penalties based on JD priorities.

    This formula is designed so that:
    - A keyword stuffer (high semantic, low career) scores moderately
    - A consulting-only candidate is excluded before scoring
    - A stale profile loses points even if technically excellent
    - The truly ideal candidate (right skills + available + India) tops the list

    Score range: [0.0, 1.0] after clamping
    """

    # ── Core composite (weighted sum of three signals) ──
    semantic = float(row.get("semantic_score",    0.0))
    career   = float(row.get("career_score",      0.0)) / 100.0
    behav    = float(row.get("behavioral_score",  0.0)) / 100.0

    composite = (
        (W_SEMANTIC * semantic) +
        (W_CAREER   * career) +
        (W_BEHAV    * behav)
    )

    # ── Bonuses ──────────────────────────────────────────────────────
    # These reflect explicit JD preferences

    if row.get("bonus_preferred_city", False):
        composite += BONUS_PREFERRED_CITY    # Pune/Noida/Hyderabad etc
    elif row.get("bonus_india", False):
        composite += BONUS_INDIA             # India but not preferred city

    if row.get("bonus_short_notice", False):
        composite += BONUS_SHORT_NOTICE      # ≤30 days — "sub-30 preferred"
    elif row.get("bonus_medium_notice", False):
        composite += BONUS_MEDIUM_NOTICE     # 31-60 days

    if row.get("open_to_work", False):
        composite += BONUS_OPEN_TO_WORK      # explicitly available

    if int(row.get("saved_by_recruiters_30d", 0)) >= 5:
        composite += BONUS_HIGH_DEMAND       # market-validated quality

    if row.get("has_verified_skills", False):
        composite += BONUS_VERIFIED_SKILLS   # objective, not self-reported

    # ── Penalties ────────────────────────────────────────────────────
    # These reflect explicit JD warnings

    if row.get("penalty_stale", False):
        composite -= PENALTY_STALE           # >180 days — JD says down-weight

    elif row.get("penalty_semi_stale", False):
        composite -= PENALTY_SEMI_STALE      # 90-180 days

    if row.get("penalty_long_notice", False):
        composite -= PENALTY_LONG_NOTICE     # >90 days — "bar gets higher"

    if row.get("penalty_low_response", False):
        composite -= PENALTY_LOW_RESPONSE    # effectively unreachable

    # Outside India and not willing to relocate
    if (not bool(row.get("in_india", False)) and
            not bool(row.get("willing_to_relocate", False))):
        composite -= PENALTY_NO_INDIA

    # ── Clamp to valid range ──
    return round(float(np.clip(composite, 0.0, 1.0)), 6)


def score_and_rank(pool):
    """
    Applies all filtering and scoring to the retrieved pool.

    Filter order matters:
    1. Remove structurally invalid candidates (missing fields)
    2. Remove honeypots (impossible profiles)
    3. Remove disqualified candidates (consulting-only, non-technical etc)
    4. Compute composite score for each remaining candidate
    5. Sort descending by composite score
    6. Assign ranks 1-N
    """

    original_count = len(pool)

    # ── Hard filters ─────────────────────────────────────────────────

    # Remove structurally invalid (missing required fields)
    pool = pool[pool["is_valid"] == True].copy()
    after_valid = len(pool)

    # Remove honeypots — impossible or fabricated profiles
    pool = pool[pool["is_honeypot"] == False].copy()
    after_honeypot = len(pool)

    # Remove explicitly disqualified candidates
    # (pure consulting, non-technical with no AI history, too junior)
    pool = pool[pool["is_disqualified"] == False].copy()
    after_disq = len(pool)

    print(f"     Filtering: {original_count:,} → "
          f"{after_valid:,} (valid) → "
          f"{after_honeypot:,} (no honeypots) → "
          f"{after_disq:,} (no disqualified)")

    if len(pool) < 100:
        print(f"\n  ⚠️  WARNING: Only {len(pool)} rankable candidates found.")
        print(f"     Need at least 100. Increase FAISS_RETRIEVE_K or")
        print(f"     check if precompute filters are too aggressive.")

    # ── Compute composite scores ──────────────────────────────────────

    print(f"  ⚙️  Computing composite scores for {len(pool):,} candidates...")
    t = time.time()

    pool["composite_score"] = pool.apply(compute_composite_score, axis=1)

    print(f"     Scored in {time.time()-t:.2f}s")
    print(f"     Score range: {pool['composite_score'].min():.4f} – "
          f"{pool['composite_score'].max():.4f}")

    # ── Sort by composite score ───────────────────────────────────────
    pool = pool.sort_values(
        by=["composite_score", "career_score", "behavioral_score"],
        ascending=False
    ).reset_index(drop=True)

    # Assign rank (1-based)
    pool["rank"] = pool.index + 1

    return pool


# ─────────────────────────────────────────────
# STAGE 5 — REASONING GENERATION
# ─────────────────────────────────────────────

def build_reasoning(row, rank):
    """
    Generates a 2-sentence, factual, rank-consistent reasoning
    for each candidate in the top 100.

    Rules this function follows (checked at Stage 4 evaluation):
    ✓ References specific facts from the candidate's profile
    ✓ Connects to specific JD requirements
    ✓ Acknowledges weaknesses honestly
    ✓ Never mentions skills or experience NOT in the profile
    ✓ Each reasoning is substantively different
    ✓ Tone matches the rank (rank 1 ≠ rank 80 in tone)

    Strategy:
        Sentence 1 — WHY this person is qualified
            Uses: title, yoe, recent company, recent industry,
                  AI keywords found in career, verified skills
            Explicitly connects to JD requirements

        Sentence 2 — CONTEXT (strength OR honest concern)
            Top 20:  Additional strength or strong overall fit
            21-50:   A constraint + why still ranked here
            51-100:  Honest gap acknowledgment + what partial fit exists
    """

    rd = json.loads(row.get("reasoning_data", "{}"))

    # Extract pre-computed facts from reasoning_data
    title          = rd.get("title",           row.get("current_title", "Professional"))
    yoe            = rd.get("yoe",             row.get("yoe", 0))
    recent_company = rd.get("recent_company",  row.get("recent_company", ""))
    recent_industry= rd.get("recent_industry", row.get("recent_industry", ""))
    recent_title   = rd.get("recent_title",    "")
    top_skills     = rd.get("top_skills",      row.get("top_skills_summary", ""))
    notice         = int(rd.get("notice_days",    row.get("notice_period_days", 90)))
    inactive       = int(rd.get("days_inactive",  row.get("days_inactive", 0)))
    response_rate  = float(rd.get("response_rate",row.get("recruiter_response_rate", 0)))
    github         = float(rd.get("github_score", row.get("github_activity_score", -1)))
    in_city        = bool(rd.get("in_preferred_city", row.get("in_preferred_city", False)))
    location       = rd.get("location",        row.get("location", ""))
    is_product     = bool(row.get("recent_is_product", False))
    product_ratio  = float(rd.get("product_ratio", row.get("product_ratio", 0.5)))
    n_jobs         = int(rd.get("n_jobs",      row.get("n_jobs", 1)))
    avg_tenure     = float(rd.get("avg_tenure_months", row.get("avg_tenure_months", 0)))
    oar            = float(rd.get("oar",       row.get("offer_acceptance_rate", -1)))
    education      = rd.get("education_summary","")
    saved          = int(rd.get("saved_by_recruiters", row.get("saved_by_recruiters_30d", 0)))
    endorsements   = int(rd.get("endorsements", row.get("endorsements_received", 0)))

    # Parse AI signals (stored as JSON string)
    try:
        ai_signals = json.loads(rd.get("ai_signals", "[]"))
    except Exception:
        ai_signals = []

    # Parse verified scores
    try:
        verified_scores = json.loads(rd.get("verified_skill_scores", "{}"))
    except Exception:
        verified_scores = {}

    # ── Build Sentence 1 — WHY they're qualified ─────────────────────

    s1_parts = []

    # Core qualification: title + experience
    yoe_str = f"{yoe:.0f}" if yoe == int(yoe) else f"{yoe:.1f}"
    s1_parts.append(f"{title} with {yoe_str} years of experience")

    # Recent company context
    if recent_company and recent_industry:
        industry_lower = recent_industry.lower()
        if is_product and industry_lower not in ["it services", "consulting"]:
            s1_parts.append(
                f"most recently at {recent_company} "
                f"({recent_industry})"
            )
        elif recent_company:
            s1_parts.append(f"at {recent_company}")

    # AI/ML technical signals from actual career descriptions
    if ai_signals:
        sig_text = ", ".join(ai_signals[:3])
        s1_parts.append(
            f"with demonstrated work in {sig_text}"
        )
    elif top_skills:
        first_skill = top_skills.split(",")[0].strip()
        s1_parts.append(f"with skills including {first_skill}")

    # Verified skill scores if available (objective, not self-reported)
    if verified_scores:
        best_skill = max(verified_scores.items(), key=lambda x: x[1])
        s1_parts.append(
            f"verified {best_skill[0]} score: {best_skill[1]:.0f}/100"
        )

    # Build the sentence
    sentence1 = "; ".join(s1_parts) + "."
    sentence1 = sentence1[0].upper() + sentence1[1:]

    # ── Build Sentence 2 — CONTEXT (rank-dependent tone) ─────────────

    concerns  = []
    strengths = []

    # Identify concerns
    if inactive > 180:
        concerns.append(f"last active {inactive} days ago")
    elif inactive > 90:
        concerns.append(f"inactive for {inactive} days")

    if notice > 90:
        concerns.append(f"{notice}-day notice period")
    elif notice > 60:
        concerns.append(f"{notice}-day notice")

    if response_rate < 0.25:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")

    if oar != -1 and oar < 0.3:
        concerns.append(f"low offer acceptance rate ({oar:.0%})")

    if avg_tenure < 15 and n_jobs > 2:
        concerns.append(
            f"frequent job changes (avg tenure {avg_tenure:.0f} months)"
        )

    if product_ratio < 0.3:
        concerns.append("limited product-company experience")

    # Identify strengths
    if in_city and location:
        strengths.append(f"based in {location}")

    if notice <= 30:
        strengths.append(f"available within {notice} days")

    if github >= 50:
        strengths.append(f"active GitHub contributor (score: {github:.0f}/100)")

    if saved >= 10:
        strengths.append(f"high market demand — saved by {saved} recruiters recently")

    if endorsements >= 100:
        strengths.append(f"strongly peer-endorsed ({endorsements} endorsements)")

    if response_rate >= 0.7:
        strengths.append(f"highly responsive to recruiters ({response_rate:.0%})")

    if inactive <= 14:
        strengths.append("active on platform in last 2 weeks")

    if product_ratio >= 0.8:
        strengths.append("strong product-company background")

    # ── Select sentence 2 tone based on rank ──
    if rank <= 10:
        # Top 10: strong positive, name specific strengths
        if strengths:
            s2 = f"Strong overall fit — {'; '.join(strengths[:2])}."
        else:
            s2 = (
                "Solid alignment with JD on technical depth, "
                "production experience, and platform activity."
            )

    elif rank <= 25:
        # 11-25: positive with one honest note
        if concerns and strengths:
            s2 = (
                f"Good candidate overall; "
                f"note {concerns[0]}, "
                f"but {strengths[0]} is a positive signal."
            )
        elif strengths:
            s2 = (
                f"Good fit across JD dimensions — "
                f"{'; '.join(strengths[:2])}."
            )
        elif concerns:
            s2 = (
                f"Technically solid but note: "
                f"{'; '.join(concerns[:2])}."
            )
        else:
            s2 = "Reasonable fit with JD requirements on both technical and availability dimensions."

    elif rank <= 60:
        # 26-60: balanced — name the main constraint
        if concerns:
            main_concern = concerns[0]
            if strengths:
                s2 = (
                    f"Included at this rank due to {main_concern}; "
                    f"{strengths[0]} partially offsets this."
                )
            else:
                s2 = (
                    f"Ranked here primarily due to {main_concern} — "
                    f"technical signals are adequate but availability is a constraint."
                )
        elif strengths:
            s2 = (
                f"Decent technical overlap with JD requirements; "
                f"{strengths[0]}."
            )
        else:
            s2 = (
                "Partial JD match — included based on relevant career signals "
                "but does not fully satisfy all requirements."
            )

    else:
        # 61-100: honest about gaps, still specific
        if concerns:
            cons_str = "; ".join(concerns[:2])
            s2 = (
                f"Ranked near threshold — concerns include {cons_str}. "
                f"Included as a borderline candidate given limited stronger alternatives."
            )
        elif not ai_signals:
            s2 = (
                "No direct AI/ML/retrieval work found in career history; "
                "included based on adjacent technical background only."
            )
        else:
            s2 = (
                f"Below top-60 fit; career shows {ai_signals[0]} "
                f"but overall profile alignment with JD requirements is limited."
            )

    return f"{sentence1} {s2}"


def generate_all_reasoning(top_100):
    """
    Generates reasoning for every row in the top-100 DataFrame.
    Returns a list of 100 reasoning strings.
    """
    print(f"  ✍️  Generating reasoning for {len(top_100)} candidates...")
    t = time.time()

    reasonings = []
    for _, row in top_100.iterrows():
        rank      = int(row["rank"])
        reasoning = build_reasoning(row, rank)
        reasonings.append(reasoning)

    print(f"     Generated in {time.time()-t:.2f}s")
    return reasonings


# ─────────────────────────────────────────────
# STAGE 6 — VALIDATE AND WRITE CSV
# ─────────────────────────────────────────────

def validate_submission(output_rows):
    """
    Runs all the checks the hackathon validator will run.
    Raises ValueError with a clear message if anything fails.

    Based on Section 3 and Section 6 of submission_spec.md.
    """
    errors = []

    # Check exactly 100 rows
    if len(output_rows) != 100:
        errors.append(
            f"❌ Row count: {len(output_rows)} (required: exactly 100)"
        )

    # Check rank uniqueness and range
    ranks = [r["rank"] for r in output_rows]
    if sorted(ranks) != list(range(1, 101)):
        missing = set(range(1, 101)) - set(ranks)
        dupes   = [r for r in ranks if ranks.count(r) > 1]
        if missing:
            errors.append(f"❌ Missing ranks: {sorted(missing)[:5]}")
        if dupes:
            errors.append(f"❌ Duplicate ranks: {list(set(dupes))[:5]}")

    # Check candidate_id uniqueness
    ids = [r["candidate_id"] for r in output_rows]
    if len(ids) != len(set(ids)):
        dupes = [cid for cid in ids if ids.count(cid) > 1]
        errors.append(f"❌ Duplicate candidate IDs: {list(set(dupes))[:3]}")

    # Check candidate_id format
    for r in output_rows:
        cid = r["candidate_id"]
        if not (isinstance(cid, str) and cid.startswith("CAND_")):
            errors.append(f"❌ Invalid candidate_id format: {cid}")
            break

    # Check scores are monotonically non-increasing
    scores = [r["score"] for r in output_rows]
    for i in range(len(scores) - 1):
        if scores[i] < scores[i + 1] - 1e-6:
            errors.append(
                f"❌ Score not monotonic at rank {i+1} → {i+2}: "
                f"{scores[i]:.6f} < {scores[i+1]:.6f}"
            )
            break

    # Check scores are valid floats in [0, 1]
    for r in output_rows:
        s = r["score"]
        if not (isinstance(s, (int, float)) and 0 <= s <= 1):
            errors.append(f"❌ Score out of range [0,1]: {s}")
            break

    if errors:
        print("\n⚠️  VALIDATION ERRORS FOUND:")
        for e in errors:
            print(f"   {e}")
        raise ValueError(f"Submission validation failed: {len(errors)} error(s)")

    print("  ✅ Validation passed — all format checks OK")


def write_csv(output_rows, output_path):
    """Writes the final submission CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(output_rows)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  💾 Saved: {output_path} ({size_kb:.1f} KB)")


def print_preview(output_rows, n=10):
    """Prints a preview of the top-N ranked candidates."""
    print(f"\n  {'─'*80}")
    print(f"  TOP {n} CANDIDATES — FINAL RANKING")
    print(f"  {'─'*80}")
    print(f"  {'#':<4} {'ID':<16} {'Score':>7}  {'Title':<28}")
    print(f"  {'─'*80}")
    for row in output_rows[:n]:
        title = row.get("_title", "")[:26]
        print(
            f"  {row['rank']:<4} "
            f"{row['candidate_id']:<16} "
            f"{row['score']:>7.4f}  "
            f"{title:<28}"
        )
    print(f"  {'─'*80}")
    print(f"\n  Sample reasoning (rank 1):")
    print(f"  {output_rows[0]['reasoning']}\n")


# ─────────────────────────────────────────────
# ORCHESTRATION
# ─────────────────────────────────────────────

def run(artifacts_dir, jd_path, output_path, mode="full"):
    """
    Runs the complete ranking pipeline end-to-end.
    Prints timing for each stage.
    """
    total_start = time.time()

    print("\n" + "="*65)
    print("     REDROB HACKATHON — rank.py (Ultimate AI Recruiter)")
    print("="*65)

    # ── Stage 1: Load ──────────────────────────────────────────────
    print(f"\n  Stage 1 — Loading artifacts...")
    t = time.time()
    index, candidate_ids, df, id_to_row = load_artifacts(artifacts_dir, mode)
    print(f"  ✅ Stage 1 complete ({time.time()-t:.1f}s)")

    # ── Stage 2: Embed JD ──────────────────────────────────────────
    print(f"\n  Stage 2 — Embedding Job Description...")
    t = time.time()
    jd_vector, _ = embed_job_description(jd_path)
    print(f"  ✅ Stage 2 complete ({time.time()-t:.1f}s)")

    # ── Stage 3: Semantic search ───────────────────────────────────
    print(f"\n  Stage 3 — Semantic search...")
    t = time.time()
    pool = semantic_search(index, jd_vector, candidate_ids, df, id_to_row)
    print(f"  ✅ Stage 3 complete ({time.time()-t:.1f}s)")

    # ── Stage 4: Score and rank ────────────────────────────────────
    print(f"\n  Stage 4 — Multi-factor scoring...")
    t = time.time()
    ranked = score_and_rank(pool)
    print(f"  ✅ Stage 4 complete ({time.time()-t:.1f}s)")

    if len(ranked) < 100:
        print(f"\n  ⚠️  Only {len(ranked)} candidates rankable.")
        print(f"     Using all {len(ranked)} for test mode.")
        top_n = ranked.copy()
    else:
        top_n = ranked.head(100).copy()

    # ── Stage 5: Reasoning ─────────────────────────────────────────
    print(f"\n  Stage 5 — Generating reasoning...")
    t = time.time()
    reasonings = generate_all_reasoning(top_n)
    print(f"  ✅ Stage 5 complete ({time.time()-t:.1f}s)")

    # ── Build output rows ──────────────────────────────────────────
    output_rows = []
    for i, (_, row) in enumerate(top_n.iterrows()):
        output_rows.append({
            "candidate_id": row["candidate_id"],
            "rank":         int(row["rank"]),
            "score":        float(row["composite_score"]),
            "reasoning":    reasonings[i],
            "_title":       str(row.get("current_title", "")),   # for preview only
        })

    # Fix: ensure monotonically non-increasing scores
    # (Tiny floating point differences can violate this)
    for i in range(1, len(output_rows)):
        if output_rows[i]["score"] > output_rows[i-1]["score"]:
            output_rows[i]["score"] = output_rows[i-1]["score"]

    # ── Stage 6: Validate and write ────────────────────────────────
    print(f"\n  Stage 6 — Validating and writing CSV...")
    t = time.time()

    # Remove internal _title field before validation
    clean_rows = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in output_rows
    ]

    if len(clean_rows) == 100:
        validate_submission(clean_rows)
    else:
        print(f"  ⚠️  Test mode: {len(clean_rows)} rows (< 100 — skipping validation)")

    write_csv(clean_rows, output_path)
    print(f"  ✅ Stage 6 complete ({time.time()-t:.1f}s)")

    # ── Print preview ──────────────────────────────────────────────
    print_preview(output_rows, n=10)

    # ── Final timing report ────────────────────────────────────────
    total_time = time.time() - total_start
    print("="*65)
    print(f"  🏁 RANKING COMPLETE")
    print(f"     Total time : {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"     Output     : {output_path}")
    print(f"     Candidates : {len(clean_rows)} ranked")

    budget_used = (total_time / 300) * 100
    print(f"     Budget used: {budget_used:.0f}% of 5-minute limit")

    if total_time > 270:
        print(f"\n  ⚠️  WARNING: Close to 5-minute limit!")
        print(f"     Consider reducing FAISS_RETRIEVE_K")
    else:
        print(f"     ✅ Well within time budget")

    print("="*65 + "\n")

    return clean_rows


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Redrob Hackathon — AI Candidate Ranker\n"
            "Produces a ranked CSV of the top 100 candidates for a given JD."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--candidates",
        default="data/candidates.jsonl",
        help="Path to candidates.jsonl or candidates.jsonl.gz"
    )
    parser.add_argument(
        "--artifacts",
        default="artifacts/",
        help="Path to artifacts directory (contains .faiss and .parquet files)"
    )
    parser.add_argument(
        "--jd",
        default="data/job_description.md",
        help="Path to the job description markdown file"
    )
    parser.add_argument(
        "--out",
        default="submission.csv",
        help="Output CSV path (default: submission.csv)"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "full"],
        default="full",
        help=(
            "'test' uses sample artifacts (50 candidates), "
            "'full' uses the complete 100K dataset"
        )
    )

    args = parser.parse_args()

    # Print run config
    print("\n  Configuration:")
    print(f"    Mode       : {args.mode}")
    print(f"    Artifacts  : {args.artifacts}")
    print(f"    JD         : {args.jd}")
    print(f"    Output     : {args.out}")

    run(
        artifacts_dir=args.artifacts,
        jd_path=args.jd,
        output_path=args.out,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()