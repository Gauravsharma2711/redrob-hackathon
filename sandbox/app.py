"""
app.py — Redrob Hackathon Sandbox Demo
========================================
This is the recruiter-facing demo that satisfies the hackathon's
mandatory "sandbox / demo link" requirement (Section 10.5 of the spec).

WHAT THIS IS FOR:
    A working hosted environment where organizers (and you) can verify
    your ranking system runs reproducibly on a small sample.

    This is NOT the official ranking step that gets scored — that's
    rank.py, run separately on the full 100K dataset in a sandboxed
    Docker container at Stage 3.

    This file exists purely as a sanity-check demo:
    - Loads the small 50-candidate sample artifacts
    - Lets a recruiter paste a JD and see ranked results instantly
    - Proves your ranking logic actually works end-to-end

HOW IT WORKS:
    It reuses the exact same scoring formula and reasoning generation
    as rank.py — just wrapped in an interactive web UI instead of
    writing a CSV file. This guarantees the demo behaves identically
    to your real ranking pipeline.

HOW TO RUN LOCALLY:
    streamlit run app.py

HOW TO DEPLOY TO HUGGINGFACE SPACES:
    1. Create a new Space (type: Streamlit) on huggingface.co
    2. Upload this file, plus:
         artifacts/sample_candidates.faiss
         artifacts/sample_candidate_ids.npy
         artifacts/sample_metadata.parquet
         requirements.txt
    3. HuggingFace installs requirements.txt and launches automatically
    4. Copy the public URL — that's your sandbox link for submission
"""

import json
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG — must be the first Streamlit call
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Redrob AI Recruiter — Sandbox Demo",
    page_icon="🎯",
    layout="wide",
)

# ─────────────────────────────────────────────
# CONSTANTS — identical to rank.py
# Keeping these in sync ensures the demo matches
# the real ranking step exactly.
# ─────────────────────────────────────────────

MODEL_NAME = "BAAI/bge-base-en-v1.5"

W_SEMANTIC = 0.45
W_CAREER   = 0.35
W_BEHAV    = 0.20

BONUS_PREFERRED_CITY = 0.035
BONUS_INDIA          = 0.015
BONUS_SHORT_NOTICE   = 0.040
BONUS_MEDIUM_NOTICE  = 0.015
BONUS_OPEN_TO_WORK    = 0.015
BONUS_HIGH_DEMAND     = 0.010
BONUS_VERIFIED_SKILLS = 0.010

PENALTY_STALE         = 0.080
PENALTY_SEMI_STALE    = 0.035
PENALTY_LONG_NOTICE   = 0.050
PENALTY_LOW_RESPONSE  = 0.040
PENALTY_NO_INDIA      = 0.020

ARTIFACTS_DIR = "artifacts"
FAISS_FILE    = os.path.join(ARTIFACTS_DIR, "sample_candidates.faiss")
IDS_FILE      = os.path.join(ARTIFACTS_DIR, "sample_candidate_ids.npy")
META_FILE     = os.path.join(ARTIFACTS_DIR, "sample_metadata.parquet")

# Default JD shown when the demo first loads — makes it easy
# for organizers to test with one click instead of typing
SAMPLE_JD = """Senior AI Engineer — Founding Team at Redrob AI.
5-9 years experience. Production embeddings-based retrieval systems
(sentence-transformers, BGE, E5). Vector databases: FAISS, Pinecone,
Weaviate, Qdrant, Milvus, Elasticsearch. Strong Python. Hands-on
experience designing evaluation frameworks for ranking systems
(NDCG, MRR, MAP). Product company experience required — not pure
consulting or IT services backgrounds. Pune/Noida preferred, open
to Hyderabad, Mumbai, Bangalore, Delhi NCR. Sub-30-day notice
period ideal. Active job seeker preferred."""


# ─────────────────────────────────────────────
# CACHED LOADERS
# Streamlit's @st.cache_resource keeps these loaded across
# user interactions — the model and index are only loaded ONCE
# per server session, not on every search.
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """Loads the embedding model once and keeps it in memory."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, device="cpu")


@st.cache_resource(show_spinner=False)
def load_index_and_metadata():
    """
    Loads the FAISS index, candidate ID map, and metadata DataFrame.
    Cached so this only happens once when the app starts, not on
    every search the recruiter makes.
    """
    import faiss

    if not os.path.exists(FAISS_FILE):
        return None, None, None

    index = faiss.read_index(FAISS_FILE)
    index.hnsw.efSearch = 64   # smaller search depth — only 50 candidates total

    candidate_ids = np.load(IDS_FILE, allow_pickle=True)
    df            = pd.read_parquet(META_FILE)

    id_to_row = {cid: i for i, cid in enumerate(candidate_ids)}

    return index, candidate_ids, df, id_to_row


# ─────────────────────────────────────────────
# SCORING — identical formula to rank.py
# ─────────────────────────────────────────────

def compute_composite_score(row):
    """
    Same three-signal composite scoring as rank.py.
    Kept identical so the demo's behaviour matches the real
    ranking step exactly — no surprises at Stage 3 reproduction.
    """
    semantic = float(row.get("semantic_score",   0.0))
    career   = float(row.get("career_score",     0.0)) / 100.0
    behav    = float(row.get("behavioral_score", 0.0)) / 100.0

    composite = (
        (W_SEMANTIC * semantic) +
        (W_CAREER   * career) +
        (W_BEHAV    * behav)
    )

    if row.get("bonus_preferred_city", False):
        composite += BONUS_PREFERRED_CITY
    elif row.get("bonus_india", False):
        composite += BONUS_INDIA

    if row.get("bonus_short_notice", False):
        composite += BONUS_SHORT_NOTICE
    elif row.get("bonus_medium_notice", False):
        composite += BONUS_MEDIUM_NOTICE

    if row.get("open_to_work", False):
        composite += BONUS_OPEN_TO_WORK

    if int(row.get("saved_by_recruiters_30d", 0)) >= 5:
        composite += BONUS_HIGH_DEMAND

    if row.get("has_verified_skills", False):
        composite += BONUS_VERIFIED_SKILLS

    if row.get("penalty_stale", False):
        composite -= PENALTY_STALE
    elif row.get("penalty_semi_stale", False):
        composite -= PENALTY_SEMI_STALE

    if row.get("penalty_long_notice", False):
        composite -= PENALTY_LONG_NOTICE

    if row.get("penalty_low_response", False):
        composite -= PENALTY_LOW_RESPONSE

    if (not bool(row.get("in_india", False)) and
            not bool(row.get("willing_to_relocate", False))):
        composite -= PENALTY_NO_INDIA

    return round(float(np.clip(composite, 0.0, 1.0)), 6)


# ─────────────────────────────────────────────
# REASONING GENERATION — identical to rank.py
# ─────────────────────────────────────────────

def build_reasoning(row, rank):
    """
    Generates the same 2-sentence factual reasoning as rank.py.
    Sentence 1 = qualification facts. Sentence 2 = rank-appropriate
    strength or honest concern.
    """
    rd = json.loads(row.get("reasoning_data", "{}"))

    title          = rd.get("title", row.get("current_title", "Professional"))
    yoe            = rd.get("yoe", row.get("yoe", 0))
    recent_company = rd.get("recent_company", "")
    recent_industry= rd.get("recent_industry", "")
    notice         = int(rd.get("notice_days", 90))
    inactive       = int(rd.get("days_inactive", 0))
    response_rate  = float(rd.get("response_rate", 0))
    github         = float(rd.get("github_score", -1))
    in_city        = bool(rd.get("in_preferred_city", False))
    location       = rd.get("location", "")
    is_product     = bool(row.get("recent_is_product", False))
    product_ratio  = float(rd.get("product_ratio", 0.5))
    n_jobs         = int(rd.get("n_jobs", 1))
    avg_tenure     = float(rd.get("avg_tenure_months", 0))
    oar            = float(rd.get("oar", -1))
    saved          = int(rd.get("saved_by_recruiters", 0))
    endorsements   = int(rd.get("endorsements", 0))

    try:
        ai_signals = json.loads(rd.get("ai_signals", "[]"))
    except Exception:
        ai_signals = []

    try:
        verified_scores = json.loads(rd.get("verified_skill_scores", "{}"))
    except Exception:
        verified_scores = {}

    # ── Sentence 1 ──
    s1_parts = []
    yoe_str = f"{yoe:.0f}" if yoe == int(yoe) else f"{yoe:.1f}"
    s1_parts.append(f"{title} with {yoe_str} years of experience")

    if recent_company and recent_industry:
        if is_product and recent_industry.lower() not in ["it services", "consulting"]:
            s1_parts.append(f"most recently at {recent_company} ({recent_industry})")
        else:
            s1_parts.append(f"at {recent_company}")

    if ai_signals:
        s1_parts.append(f"with demonstrated work in {', '.join(ai_signals[:3])}")

    if verified_scores:
        best_skill = max(verified_scores.items(), key=lambda x: x[1])
        s1_parts.append(f"verified {best_skill[0]} score: {best_skill[1]:.0f}/100")

    sentence1 = "; ".join(s1_parts) + "."
    sentence1 = sentence1[0].upper() + sentence1[1:]

    # ── Sentence 2 ──
    concerns, strengths = [], []

    if inactive > 180:
        concerns.append(f"last active {inactive} days ago")
    elif inactive > 90:
        concerns.append(f"inactive for {inactive} days")
    if notice > 90:
        concerns.append(f"{notice}-day notice period")
    if response_rate < 0.25:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")
    if oar != -1 and oar < 0.3:
        concerns.append(f"low offer acceptance rate ({oar:.0%})")
    if avg_tenure < 15 and n_jobs > 2:
        concerns.append(f"frequent job changes (avg tenure {avg_tenure:.0f} months)")
    if product_ratio < 0.3:
        concerns.append("limited product-company experience")

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
    if product_ratio >= 0.8:
        strengths.append("strong product-company background")

    if rank <= 10:
        s2 = (f"Strong overall fit — {'; '.join(strengths[:2])}."
              if strengths else
              "Solid alignment with JD on technical depth and availability.")
    elif rank <= 25:
        if concerns and strengths:
            s2 = f"Good candidate overall; note {concerns[0]}, but {strengths[0]} is a positive signal."
        elif strengths:
            s2 = f"Good fit across JD dimensions — {'; '.join(strengths[:2])}."
        else:
            s2 = "Reasonable fit with JD requirements on technical and availability dimensions."
    elif rank <= 60:
        if concerns:
            s2 = (f"Ranked here primarily due to {concerns[0]}; "
                  f"{strengths[0] if strengths else 'technical signals are adequate'}.")
        else:
            s2 = f"Decent technical overlap with JD requirements; {strengths[0] if strengths else 'included on relevant signals'}."
    else:
        if concerns:
            s2 = f"Ranked near threshold — concerns include {'; '.join(concerns[:2])}."
        elif not ai_signals:
            s2 = "No direct AI/ML/retrieval work found in career history; included on adjacent background only."
        else:
            s2 = f"Career shows {ai_signals[0]} but overall JD alignment is limited."

    return f"{sentence1} {s2}"


# ─────────────────────────────────────────────
# CORE SEARCH FUNCTION
# ─────────────────────────────────────────────

def run_search(jd_text, index, candidate_ids, df, id_to_row, top_n=20):
    """
    Embeds the JD, searches the sample FAISS index, scores every
    result, and returns a ranked DataFrame.

    This mirrors Stages 2-5 of rank.py exactly, just scoped to
    the 50-candidate sample instead of the full 100K dataset.
    """
    model = load_model()

    # Build the same condensed, signal-rich query rank.py uses
    jd_query = (
        "Senior AI Engineer with 5-9 years experience at product companies. "
        "Production embeddings-based retrieval systems, vector databases "
        "(FAISS, Pinecone, Weaviate, Qdrant), ranking systems, recommendation "
        "engines, NLP, evaluation frameworks (NDCG, MRR, MAP). Strong Python. "
        "Product company experience required, not consulting. "
        f"{jd_text[:800]}"
    )

    jd_vector = model.encode(
        [jd_query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    k = min(50, index.ntotal)
    scores, indices = index.search(jd_vector, k)

    rows = []
    for faiss_pos, sem_score in zip(indices[0], scores[0]):
        if faiss_pos < 0:
            continue
        cid      = candidate_ids[faiss_pos]
        meta_idx = id_to_row.get(cid)
        if meta_idx is None:
            continue
        row = df.iloc[meta_idx].to_dict()
        row["semantic_score"] = float(sem_score)
        rows.append(row)

    pool = pd.DataFrame(rows)

    # Filter honeypots, invalid, and disqualified candidates
    pool = pool[pool["is_valid"] == True]
    pool = pool[pool["is_honeypot"] == False]
    pool = pool[pool["is_disqualified"] == False]

    if len(pool) == 0:
        return pool

    pool["composite_score"] = pool.apply(compute_composite_score, axis=1)
    pool = pool.sort_values("composite_score", ascending=False).reset_index(drop=True)
    pool["rank"] = pool.index + 1

    return pool.head(top_n)


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

def main():

    # ── Header ──
    st.title("🎯 Redrob AI Recruiter — Sandbox Demo")
    st.caption(
        "Intelligent Candidate Discovery & Ranking · Hackathon Sandbox · "
        "Runs on a 50-candidate sample for fast verification"
    )

    st.info(
        "💡 This sandbox demonstrates the ranking logic on a small sample "
        "(50 candidates). The full system ranks the complete 100,000-candidate "
        "pool using the identical scoring formula in `rank.py`.",
        icon="ℹ️",
    )

    # ── Load artifacts ──
    loaded = load_index_and_metadata()
    if loaded[0] is None:
        st.error(
            "❌ Sample artifacts not found. Make sure these files exist:\n\n"
            f"- `{FAISS_FILE}`\n- `{IDS_FILE}`\n- `{META_FILE}`\n\n"
            "Run the precompute pipeline in `--mode test` first."
        )
        return

    index, candidate_ids, df, id_to_row = loaded

    st.success(f"✅ Loaded {len(df)} candidates into the search index", icon="✅")

    # ── Input section ──
    st.subheader("📋 Job Description")

    jd_text = st.text_area(
        "Paste a job description, or use the pre-filled sample:",
        value=SAMPLE_JD,
        height=180,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        top_n = st.number_input(
            "Show top N", min_value=5, max_value=50, value=10, step=5
        )
    with col2:
        run_button = st.button("🔍 Find Candidates", type="primary", use_container_width=False)

    # ── Run search ──
    if run_button:
        if not jd_text.strip():
            st.warning("Please enter a job description first.")
            return

        with st.spinner("Embedding JD, searching candidates, scoring, generating reasoning..."):
            start = time.time()
            ranked = run_search(jd_text, index, candidate_ids, df, id_to_row, top_n=top_n)
            elapsed = time.time() - start

        if len(ranked) == 0:
            st.warning(
                "No rankable candidates found after filtering honeypots and "
                "disqualified profiles. Try a broader JD or increase sample size."
            )
            return

        st.success(f"✅ Ranked {len(ranked)} candidates in {elapsed:.2f} seconds")

        st.subheader(f"🏆 Top {len(ranked)} Candidates")

        # ── Render each candidate as a card ──
        for _, row in ranked.iterrows():
            rank      = int(row["rank"])
            score     = float(row["composite_score"])
            title     = row.get("current_title", "Unknown")
            company   = row.get("current_company", "")
            location  = row.get("location", "")
            yoe       = row.get("yoe", 0)
            notice    = int(row.get("notice_period_days", 0))
            reasoning = build_reasoning(row, rank)

            with st.container(border=True):
                c1, c2 = st.columns([5, 1])

                with c1:
                    st.markdown(
                        f"**#{rank} — {title}** at {company}  \n"
                        f"📍 {location} · 💼 {yoe:.0f} yrs experience · "
                        f"⏱️ {notice}-day notice"
                    )
                    st.caption(reasoning)

                with c2:
                    st.metric("Match", f"{score:.0%}")

        # ── Download as CSV ──
        st.divider()
        csv_data = ranked[["candidate_id", "rank", "composite_score"]].rename(
            columns={"composite_score": "score"}
        )
        csv_data["reasoning"] = [
            build_reasoning(row, int(row["rank"])) for _, row in ranked.iterrows()
        ]

        st.download_button(
            "⬇️ Download results as CSV",
            data=csv_data.to_csv(index=False),
            file_name="sandbox_results.csv",
            mime="text/csv",
        )

    else:
        st.caption("👆 Click **Find Candidates** to run the ranking pipeline.")

    # ── Footer ──
    st.divider()
    st.caption(
        "Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge · "
        "Sandbox demo running on sample data · Full ranking logic in `rank.py`"
    )


if __name__ == "__main__":
    main()