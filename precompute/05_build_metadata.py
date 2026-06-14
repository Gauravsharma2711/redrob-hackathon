"""
05_build_metadata.py
---------------------
WHAT THIS FILE DOES:
    Reads every candidate (with scores + narratives) and extracts
    ALL the structured data rank.py needs into a single fast-loading
    Parquet file.

    Think of this as building the "brain" of the ranking system.
    Every calculation that CAN be done before ranking IS done here —
    so rank.py only needs to load, filter, score, and write CSV.

WHY THIS MATTERS:
    rank.py runs inside a sandboxed environment:
        - ≤5 minutes wall-clock
        - CPU only
        - No internet

    Every millisecond saved in pre-computation here = more time
    for semantic search and reasoning in rank.py.

WHAT GETS PRE-COMPUTED:
    1. All _scores (career, behavioral, disqualifiers)
    2. All 23 redrob_signals as flat columns
    3. Derived signals (days_inactive, india_flag, preferred_city_flag)
    4. Career AI keyword matches (for reasoning generation)
    5. Social proof signals (connections, endorsements, recruiter saves)
    6. Pre-built reasoning_data JSON (all facts needed for 2-sentence reasoning)
    7. Top skills summary (for reasoning)
    8. Recent company + industry (for reasoning)
    9. All disqualifier and honeypot flags

    rank.py reads this file and immediately has everything it needs.
    No re-parsing JSON. No re-computing dates. No re-scanning careers.

HOW TO RUN:

    Test mode (50 candidates):
        python precompute/05_build_metadata.py --mode test

    Full mode (100,000 candidates):
        python precompute/05_build_metadata.py --mode full

OUTPUT:
    artifacts/candidates_metadata.parquet    ← used by rank.py
    artifacts/sample_metadata.parquet        ← used by sandbox demo
"""

import json
import os
import argparse
from datetime import datetime
import pandas as pd

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────

INPUT_SAMPLE  = "artifacts/narratives_sample.jsonl"
INPUT_FULL    = "artifacts/narratives_candidates.jsonl"

OUTPUT_SAMPLE = "artifacts/sample_metadata.parquet"
OUTPUT_FULL   = "artifacts/candidates_metadata.parquet"

TODAY = datetime(2026, 6, 10)
TODAY_STR     = "2026-06-10"

# ─────────────────────────────────────────────
# CONSTANTS — same as feature_engineering.py
# Repeated here so this script is self-contained
# ─────────────────────────────────────────────

# AI/ML keywords to scan in career descriptions
AI_CAREER_KEYWORDS = [
    "embedding", "embeddings", "vector search", "vector database",
    "semantic search", "retrieval", "ranking", "reranking",
    "recommendation", "recommendation system", "search engine",
    "information retrieval", "hybrid search", "dense retrieval",
    "language model", "large language model", "llm", "nlp",
    "natural language processing", "fine-tun", "fine tuning",
    "rag", "retrieval augmented", "transformer", "bert",
    "machine learning", "deep learning", "neural network",
    "model serving", "model deployment", "inference", "mlops",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "elasticsearch", "opensearch",
    "ndcg", "mrr", "map", "evaluation framework",
    "a/b test", "a/b testing", "experimentation",
    "xgboost", "learning-to-rank", "feature store",
]

# Consulting companies (for disqualifier and product ratio)
CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis", "hexaware",
}

CONSULTING_INDUSTRIES = {"it services", "consulting", "bpo"}

PRODUCT_INDUSTRIES = {
    "software", "fintech", "e-commerce", "food delivery",
    "transportation", "saas", "ai/ml", "edtech", "healthtech",
    "media", "gaming",
}

# India cities the JD prefers
PREFERRED_CITIES = [
    "pune", "noida", "delhi", "ncr", "gurgaon", "gurugram",
    "hyderabad", "mumbai", "bangalore", "bengaluru", "chennai",
]

# Non-technical titles that are disqualifiers
NON_TECH_TITLES = [
    "marketing manager", "operations manager", "accountant",
    "hr manager", "customer support", "sales manager",
    "civil engineer", "mechanical engineer", "graphic designer",
    "content writer", "business development",
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def parse_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None


def days_inactive(date_str):
    d = parse_date(date_str)
    return (TODAY - d).days if d else 999


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def safe_bool(val, default=False):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return default


# ─────────────────────────────────────────────
# CAREER ANALYSIS HELPERS
# ─────────────────────────────────────────────

def get_career_text(career):
    """Full text of all career descriptions combined."""
    return " ".join([
        j.get("description", "") for j in career
    ]).lower()


def get_ai_keywords_found(career_text):
    """
    Returns list of AI/ML keywords found in career descriptions.
    Used in reasoning generation to name specific technologies.
    """
    found = [kw for kw in AI_CAREER_KEYWORDS if kw in career_text]
    return found


def get_product_ratio(career):
    """
    Returns the fraction of career months spent at product companies.
    0.0 = entirely consulting, 1.0 = entirely product companies.
    """
    product_months    = 0
    consulting_months = 0

    for job in career:
        months  = safe_int(job.get("duration_months", 0))
        company = job.get("company", "").lower()
        ind     = job.get("industry", "").lower()

        is_cons = (
            any(f in company for f in CONSULTING_COMPANIES)
            or any(i in ind for i in CONSULTING_INDUSTRIES)
        )
        is_prod = ind in PRODUCT_INDUSTRIES and not is_cons

        if is_cons:
            consulting_months += months
        if is_prod:
            product_months    += months

    total = product_months + consulting_months
    if total == 0:
        return 0.5   # unknown — neutral
    return round(product_months / total, 4)


def get_avg_tenure(career):
    """Average months per job — used to detect job-hoppers."""
    if not career:
        return 0
    tenures = [safe_int(j.get("duration_months", 0)) for j in career]
    return round(sum(tenures) / len(tenures), 1) if tenures else 0


def get_recent_job_info(career):
    """
    Returns structured info about the most recent job.
    Used directly in reasoning generation.
    """
    if not career:
        return {
            "recent_title":    "",
            "recent_company":  "",
            "recent_industry": "",
            "recent_duration": 0,
            "recent_is_product": False,
        }

    sorted_jobs = sorted(
        career,
        key=lambda j: j.get("start_date", "2000-01-01"),
        reverse=True
    )
    recent = sorted_jobs[0]
    ind    = recent.get("industry", "").lower()

    return {
        "recent_title":      recent.get("title", ""),
        "recent_company":    recent.get("company", ""),
        "recent_industry":   recent.get("industry", ""),
        "recent_duration":   safe_int(recent.get("duration_months", 0)),
        "recent_is_product": ind in PRODUCT_INDUSTRIES,
    }


def get_top_skills_summary(skills, assessment_scores):
    """
    Returns a formatted string of the top 5 skills with their
    proficiency levels and any verified assessment scores.
    Used in reasoning generation.
    """
    PROF_ORDER = {"advanced": 3, "intermediate": 2, "beginner": 1}

    scored = sorted(
        skills,
        key=lambda s: (
            PROF_ORDER.get(s.get("proficiency", "beginner"), 1) * 10
            + s.get("endorsements", 0)
        ),
        reverse=True
    )[:5]

    parts = []
    for s in scored:
        name = s.get("name", "")
        prof = s.get("proficiency", "beginner")
        if name in assessment_scores:
            score = assessment_scores[name]
            parts.append(f"{name} (verified {score:.0f}/100)")
        else:
            parts.append(f"{name} ({prof})")

    return ", ".join(parts)


def get_education_summary(education):
    """Returns highest degree info as a readable string."""
    if not education:
        return ""
    sorted_edu = sorted(
        education,
        key=lambda e: e.get("end_year", 0),
        reverse=True
    )
    best  = sorted_edu[0]
    deg   = best.get("degree", "")
    field = best.get("field_of_study", "")
    tier  = best.get("tier", "tier_4")
    return f"{deg} in {field} ({tier})" if deg else ""


# ─────────────────────────────────────────────
# REASONING DATA PRE-BUILDER
# ─────────────────────────────────────────────

def build_reasoning_data(candidate, ai_keywords, recent_job, top_skills_str):
    """
    Pre-extracts all the specific facts needed to generate
    the 2-sentence reasoning in rank.py.

    By pre-building this here, rank.py never needs to re-parse
    the full JSON — it just reads this compact dict.

    Structure:
    {
        "title": current title,
        "yoe": years of experience,
        "company": current company,
        "country": country,
        "location": city,
        "recent_title": most recent job title,
        "recent_company": most recent company,
        "recent_industry": most recent industry,
        "top_skills": "FAISS (verified 68/100), Python (advanced)...",
        "ai_signals": ["ranking", "retrieval", "a/b test"],
        "notice_days": 60,
        "days_inactive": 17,
        "response_rate": 0.91,
        "github_score": 32.6,
        "offer_acceptance_rate": 0.38,
        "is_open_to_work": True,
        "willing_to_relocate": True,
        "preferred_mode": "flexible",
        "salary_min": 27.3,
        "salary_max": 60.2,
        "in_india": True,
        "in_preferred_city": True,
        "has_verified_skills": True,
        "n_jobs": 4,
        "avg_tenure_months": 17.5
    }
    """
    p      = candidate["profile"]
    sig    = candidate["redrob_signals"]
    career = candidate.get("career_history", [])

    loc         = p.get("location", "").lower()
    country     = p.get("country", "").lower()
    in_india    = country == "india"
    in_city     = in_india and any(c in loc for c in PREFERRED_CITIES)
    inactive    = days_inactive(sig.get("last_active_date", "2020-01-01"))
    salary      = sig.get("expected_salary_range_inr_lpa", {})
    has_verified= len(sig.get("skill_assessment_scores", {})) > 0

    return {
        # Identity
        "title":             p.get("current_title", ""),
        "yoe":               safe_float(p.get("years_of_experience", 0)),
        "company":           p.get("current_company", ""),
        "country":           p.get("country", ""),
        "location":          p.get("location", ""),

        # Most recent job
        "recent_title":      recent_job["recent_title"],
        "recent_company":    recent_job["recent_company"],
        "recent_industry":   recent_job["recent_industry"],
        "recent_duration_months": recent_job["recent_duration"],

        # Skills
        "top_skills":        top_skills_str,
        "has_verified_skills": has_verified,
        "verified_skill_scores": json.dumps(
            sig.get("skill_assessment_scores", {})
        ),

        # AI signal keywords found in career
        "ai_signals":        json.dumps(ai_keywords[:5]),

        # Availability
        "notice_days":       safe_int(sig.get("notice_period_days", 90)),
        "days_inactive":     inactive,
        "response_rate":     safe_float(sig.get("recruiter_response_rate", 0)),
        "avg_response_hours":safe_float(sig.get("avg_response_time_hours", 0)),
        "github_score":      safe_float(sig.get("github_activity_score", -1)),
        "icr":               safe_float(sig.get("interview_completion_rate", 0)),
        "oar":               safe_float(sig.get("offer_acceptance_rate", -1)),
        "is_open_to_work":   safe_bool(sig.get("open_to_work_flag", False)),
        "willing_to_relocate": safe_bool(sig.get("willing_to_relocate", False)),
        "preferred_mode":    sig.get("preferred_work_mode", ""),

        # Salary
        "salary_min":        safe_float(salary.get("min", 0)),
        "salary_max":        safe_float(salary.get("max", 0)),

        # Location
        "in_india":          in_india,
        "in_preferred_city": in_city,

        # Social proof
        "connections":       safe_int(sig.get("connection_count", 0)),
        "endorsements":      safe_int(sig.get("endorsements_received", 0)),
        "saved_by_recruiters": safe_int(sig.get("saved_by_recruiters_30d", 0)),
        "search_appearances": safe_int(sig.get("search_appearance_30d", 0)),

        # Career shape
        "n_jobs":            len(career),
        "avg_tenure_months": get_avg_tenure(career),
        "product_ratio":     get_product_ratio(career),

        # Education
        "education_summary": get_education_summary(
            candidate.get("education", [])
        ),
    }


# ─────────────────────────────────────────────
# MAIN ROW BUILDER
# ─────────────────────────────────────────────

def build_row(candidate, idx):
    """
    Converts one candidate dict into a flat row dict
    ready to be added to the DataFrame.

    Every field rank.py needs is extracted here.
    rank.py never needs to look at the raw JSON.
    """

    p       = candidate["profile"]
    sig     = candidate["redrob_signals"]
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])
    scores  = candidate.get("_scores", {})
    valid   = candidate.get("_validation", {})
    disq    = scores.get("disqualifiers", {})
    asmt    = sig.get("skill_assessment_scores", {})
    salary  = sig.get("expected_salary_range_inr_lpa", {})

    # Pre-compute derived values
    career_text     = get_career_text(career)
    ai_keywords     = get_ai_keywords_found(career_text)
    recent_job      = get_recent_job_info(career)
    top_skills_str  = get_top_skills_summary(skills, asmt)
    product_ratio   = get_product_ratio(career)
    avg_tenure      = get_avg_tenure(career)
    inactive_days   = days_inactive(sig.get("last_active_date", "2020-01-01"))

    loc     = p.get("location", "").lower()
    country = p.get("country", "").lower()
    in_india= country == "india"
    in_city = in_india and any(c in loc for c in PREFERRED_CITIES)

    # Pre-build reasoning data dict
    reasoning_data = build_reasoning_data(
        candidate, ai_keywords, recent_job, top_skills_str
    )

    return {

        # ── IDENTITY ───────────────────────────────────────
        "faiss_idx":          idx,          # position in FAISS index
        "candidate_id":       candidate["candidate_id"],

        # ── PRE-COMPUTED SCORES ─────────────────────────────
        # These are the core inputs to rank.py's scoring formula
        "career_score":       safe_float(scores.get("career_score", 0)),
        "behavioral_score":   safe_float(scores.get("behavioral_score", 0)),

        # ── FILTER FLAGS ────────────────────────────────────
        # rank.py uses these to immediately exclude candidates
        "is_honeypot":        safe_bool(valid.get("is_honeypot", False)),
        "is_valid":           safe_bool(valid.get("is_valid", True)),
        "is_disqualified":    safe_bool(disq.get("is_disqualified", False)),

        # Individual disqualifier reasons (for debugging)
        "dq_pure_consulting": safe_bool(disq.get("pure_consulting_only", False)),
        "dq_non_technical":   safe_bool(disq.get("non_technical_no_ai_history", False)),
        "dq_too_junior":      safe_bool(disq.get("too_junior", False)),
        "dq_salary_too_high": safe_bool(disq.get("salary_too_high", False)),

        # ── PROFILE ─────────────────────────────────────────
        "current_title":      p.get("current_title", ""),
        "current_company":    p.get("current_company", ""),
        "current_industry":   p.get("current_industry", ""),
        "current_company_size": p.get("current_company_size", ""),
        "headline":           p.get("headline", ""),
        "summary_short":      p.get("summary", "")[:300],
        "location":           p.get("location", ""),
        "country":            p.get("country", ""),
        "yoe":                safe_float(p.get("years_of_experience", 0)),

        # ── LOCATION FLAGS ───────────────────────────────────
        "in_india":           in_india,
        "in_preferred_city":  in_city,

        # ── ALL 23 REDROB SIGNALS ────────────────────────────
        # Stored as individual columns for fast pandas filtering
        "profile_completeness":     safe_float(sig.get("profile_completeness_score", 0)),
        "last_active_date":         sig.get("last_active_date", ""),
        "days_inactive":            inactive_days,
        "open_to_work":             safe_bool(sig.get("open_to_work_flag", False)),
        "profile_views_30d":        safe_int(sig.get("profile_views_received_30d", 0)),
        "applications_30d":         safe_int(sig.get("applications_submitted_30d", 0)),
        "recruiter_response_rate":  safe_float(sig.get("recruiter_response_rate", 0)),
        "avg_response_hours":       safe_float(sig.get("avg_response_time_hours", 0)),
        "connection_count":         safe_int(sig.get("connection_count", 0)),
        "endorsements_received":    safe_int(sig.get("endorsements_received", 0)),
        "notice_period_days":       safe_int(sig.get("notice_period_days", 90)),
        "salary_min":               safe_float(salary.get("min", 0)),
        "salary_max":               safe_float(salary.get("max", 0)),
        "preferred_work_mode":      sig.get("preferred_work_mode", ""),
        "willing_to_relocate":      safe_bool(sig.get("willing_to_relocate", False)),
        "github_activity_score":    safe_float(sig.get("github_activity_score", -1)),
        "search_appearance_30d":    safe_int(sig.get("search_appearance_30d", 0)),
        "saved_by_recruiters_30d":  safe_int(sig.get("saved_by_recruiters_30d", 0)),
        "interview_completion_rate":safe_float(sig.get("interview_completion_rate", 0)),
        "offer_acceptance_rate":    safe_float(sig.get("offer_acceptance_rate", -1)),
        "verified_email":           safe_bool(sig.get("verified_email", False)),
        "verified_phone":           safe_bool(sig.get("verified_phone", False)),
        "linkedin_connected":       safe_bool(sig.get("linkedin_connected", False)),

        # ── DERIVED CAREER SIGNALS ──────────────────────────
        "product_ratio":       product_ratio,
        "avg_tenure_months":   avg_tenure,
        "n_jobs":              len(career),
        "ai_keyword_count":    len(ai_keywords),
        "ai_keywords_found":   json.dumps(ai_keywords[:8]),  # top 8 as JSON string

        # ── RECENT JOB INFO ─────────────────────────────────
        "recent_title":        recent_job["recent_title"],
        "recent_company":      recent_job["recent_company"],
        "recent_industry":     recent_job["recent_industry"],
        "recent_duration_months": recent_job["recent_duration"],
        "recent_is_product":   recent_job["recent_is_product"],

        # ── SKILL SIGNALS ───────────────────────────────────
        "top_skills_summary":  top_skills_str,
        "has_verified_skills": len(asmt) > 0,
        "n_verified_skills":   len(asmt),
        "verified_scores_json":json.dumps(asmt),  # full dict as JSON

        # ── SOCIAL PROOF ────────────────────────────────────
        # These are strong third-party validation signals
        "high_demand":         safe_int(sig.get("saved_by_recruiters_30d", 0)) >= 5,
        "well_networked":      safe_int(sig.get("connection_count", 0)) >= 300,
        "well_endorsed":       safe_int(sig.get("endorsements_received", 0)) >= 50,

        # ── PRE-BUILT BONUS FLAGS ───────────────────────────
        # rank.py applies these directly without recalculating
        "bonus_india":         in_india,
        "bonus_preferred_city":in_city,
        "bonus_short_notice":  safe_int(sig.get("notice_period_days", 90)) <= 30,
        "bonus_medium_notice": (31 <= safe_int(sig.get("notice_period_days", 90)) <= 60),
        "penalty_long_notice": safe_int(sig.get("notice_period_days", 90)) > 90,
        "penalty_stale":       inactive_days > 180,
        "penalty_semi_stale":  90 < inactive_days <= 180,
        "penalty_low_response":safe_float(sig.get("recruiter_response_rate", 0)) < 0.2,
        "penalty_open_to_work_false": not safe_bool(sig.get("open_to_work_flag", False)),

        # ── REASONING DATA ──────────────────────────────────
        # Pre-extracted facts for generating the 2-sentence reasoning
        # rank.py reads this JSON string and builds reasoning from it
        "reasoning_data":      json.dumps(reasoning_data),
    }


# ─────────────────────────────────────────────
# PROCESS AND SAVE
# ─────────────────────────────────────────────

def process_and_save(input_path, output_path):
    """
    Reads all candidates, builds the flat row for each,
    assembles into a DataFrame, and saves as Parquet.

    Parquet is a columnar format — loading only the columns
    you need is extremely fast, even for 100K rows.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs("artifacts", exist_ok=True)

    rows  = []
    total = 0

    print(f"\n  📥 Input : {input_path}")
    print(f"  📤 Output: {output_path}")
    print(f"\n  ⚙️  Building metadata rows...")

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            candidate = json.loads(line)
            row       = build_row(candidate, idx=total)
            rows.append(row)
            total    += 1

            if total % 10000 == 0:
                print(f"     Processed {total:,} candidates...")

    print(f"\n  📊 Building DataFrame from {total:,} rows...")
    df = pd.DataFrame(rows)

    # Set correct dtypes for Parquet efficiency
    float_cols = [
        "career_score", "behavioral_score", "yoe", "salary_min", "salary_max",
        "product_ratio", "avg_tenure_months", "recruiter_response_rate",
        "avg_response_hours", "github_activity_score", "interview_completion_rate",
        "offer_acceptance_rate", "profile_completeness",
    ]
    int_cols = [
        "faiss_idx", "notice_period_days", "days_inactive",
        "connection_count", "endorsements_received", "ai_keyword_count",
        "n_jobs", "n_verified_skills", "profile_views_30d",
        "applications_30d", "search_appearance_30d",
        "saved_by_recruiters_30d", "recent_duration_months",
    ]
    bool_cols = [
        "is_honeypot", "is_valid", "is_disqualified",
        "dq_pure_consulting", "dq_non_technical", "dq_too_junior",
        "dq_salary_too_high", "in_india", "in_preferred_city",
        "open_to_work", "willing_to_relocate", "verified_email",
        "verified_phone", "linkedin_connected", "has_verified_skills",
        "recent_is_product", "high_demand", "well_networked",
        "well_endorsed", "bonus_india", "bonus_preferred_city",
        "bonus_short_notice", "bonus_medium_notice",
        "penalty_long_notice", "penalty_stale", "penalty_semi_stale",
        "penalty_low_response", "penalty_open_to_work_false",
    ]

    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype("int32")

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype("bool")

    # Save as Parquet
    print(f"\n  💾 Saving to: {output_path}")
    df.to_parquet(output_path, index=False)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✅ Saved — {size_mb:.1f} MB, {total:,} rows, {len(df.columns)} columns")

    return df, total


# ─────────────────────────────────────────────
# VALIDATE AND REPORT
# ─────────────────────────────────────────────

def validate_and_report(df):
    """
    Validates the metadata file and prints a detailed
    summary so you can verify everything looks correct
    before running rank.py.
    """
    total = len(df)

    print("\n" + "="*65)
    print("         METADATA BUILDER — VALIDATION REPORT")
    print("="*65)

    # ── Counts ──
    print(f"\n  Total candidates         : {total:,}")
    print(f"  Valid structure          : {df['is_valid'].sum():,}")
    print(f"  Honeypots detected       : {df['is_honeypot'].sum():,} "
          f"({df['is_honeypot'].mean()*100:.1f}%)")
    print(f"  Disqualified             : {df['is_disqualified'].sum():,} "
          f"({df['is_disqualified'].mean()*100:.1f}%)")
    print(f"  Rankable candidates      : "
          f"{(~df['is_honeypot'] & ~df['is_disqualified'] & df['is_valid']).sum():,}")

    # ── Disqualifier breakdown ──
    print(f"\n  Disqualifier breakdown:")
    print(f"    Pure consulting only   : {df['dq_pure_consulting'].sum():,}")
    print(f"    Non-technical, no AI   : {df['dq_non_technical'].sum():,}")
    print(f"    Too junior (<3 yrs)    : {df['dq_too_junior'].sum():,}")
    print(f"    Salary too high (>80L) : {df['dq_salary_too_high'].sum():,}")

    # ── Score distributions ──
    print(f"\n  Career Score (0-100):")
    print(f"    Min     : {df['career_score'].min():.1f}")
    print(f"    Max     : {df['career_score'].max():.1f}")
    print(f"    Average : {df['career_score'].mean():.1f}")
    print(f"    Score >50 : {(df['career_score'] > 50).sum():,} candidates")
    print(f"    Score >70 : {(df['career_score'] > 70).sum():,} candidates")

    print(f"\n  Behavioral Score (0-100):")
    print(f"    Min     : {df['behavioral_score'].min():.1f}")
    print(f"    Max     : {df['behavioral_score'].max():.1f}")
    print(f"    Average : {df['behavioral_score'].mean():.1f}")

    # ── Availability ──
    print(f"\n  Availability signals:")
    print(f"    Open to work           : {df['open_to_work'].sum():,}")
    print(f"    Active in last 30 days : {(df['days_inactive'] <= 30).sum():,}")
    print(f"    Active in last 90 days : {(df['days_inactive'] <= 90).sum():,}")
    print(f"    Stale (>180 days)      : {(df['days_inactive'] > 180).sum():,}")
    print(f"    Notice ≤30 days        : {(df['notice_period_days'] <= 30).sum():,}")
    print(f"    Notice >90 days        : {(df['notice_period_days'] > 90).sum():,}")

    # ── Location ──
    print(f"\n  Location signals:")
    print(f"    Based in India         : {df['in_india'].sum():,}")
    print(f"    In preferred city      : {df['in_preferred_city'].sum():,}")
    print(f"    Willing to relocate    : {df['willing_to_relocate'].sum():,}")

    # ── AI signal coverage ──
    print(f"\n  AI keyword coverage:")
    print(f"    0 AI keywords in career  : {(df['ai_keyword_count'] == 0).sum():,}")
    print(f"    1-3 AI keywords          : {((df['ai_keyword_count'] >= 1) & (df['ai_keyword_count'] <= 3)).sum():,}")
    print(f"    4-8 AI keywords          : {((df['ai_keyword_count'] >= 4) & (df['ai_keyword_count'] <= 8)).sum():,}")
    print(f"    9+ AI keywords           : {(df['ai_keyword_count'] >= 9).sum():,}")

    # ── Columns check ──
    print(f"\n  Total columns in parquet : {len(df.columns)}")
    print(f"  All required cols present: ", end="")
    required = [
        "candidate_id", "faiss_idx", "career_score", "behavioral_score",
        "is_honeypot", "is_disqualified", "reasoning_data"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ MISSING: {missing}")
    else:
        print("✅ Yes")

    print("="*65)


def print_top_candidates(df, n=10):
    """
    Shows the top N candidates by career score.
    Helps you verify the metadata looks right.
    """
    rankable = df[
        ~df["is_honeypot"] &
        ~df["is_disqualified"] &
        df["is_valid"]
    ].sort_values("career_score", ascending=False).head(n)

    print(f"\n  {'─'*85}")
    print(f"  TOP {n} RANKABLE CANDIDATES (by career score)")
    print(f"  {'─'*85}")
    print(f"  {'ID':<15} {'Title':<28} {'C-Sc':>5} {'B-Sc':>5} "
          f"{'AI-Kw':>6} {'Notice':>6} {'City?':>5}")
    print(f"  {'─'*85}")

    for _, row in rankable.iterrows():
        print(
            f"  {row['candidate_id']:<15} "
            f"{str(row['current_title'])[:26]:<28} "
            f"{row['career_score']:>5.1f} "
            f"{row['behavioral_score']:>5.1f} "
            f"{row['ai_keyword_count']:>6} "
            f"{row['notice_period_days']:>6} "
            f"{'✓' if row['in_preferred_city'] else '':>5}"
        )

    print(f"  {'─'*85}")


def print_reasoning_sample(df):
    """
    Prints the pre-built reasoning data for the top candidate
    so you can verify it's correct before rank.py uses it.
    """
    best = df[~df["is_honeypot"] & ~df["is_disqualified"]].sort_values(
        "career_score", ascending=False
    ).iloc[0]

    print(f"\n  {'─'*65}")
    print(f"  REASONING DATA PREVIEW — {best['candidate_id']}")
    print(f"  {'─'*65}")

    rd = json.loads(best["reasoning_data"])
    for k, v in rd.items():
        if isinstance(v, str) and len(v) > 60:
            v = v[:60] + "..."
        print(f"  {k:<25}: {v}")

    print(f"  {'─'*65}\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build metadata Parquet for fast ranking"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "full"],
        default="test"
    )
    args = parser.parse_args()

    print("\n" + "="*65)
    print("      REDROB HACKATHON — 05 BUILD METADATA")
    print("="*65)

    if args.mode == "test":
        print("\n🧪 Running in TEST mode (50 candidates)")

        if not os.path.exists(INPUT_SAMPLE):
            print(f"❌ ERROR: {INPUT_SAMPLE} not found.")
            print("   Run 03_build_narratives.py --mode test first.")
            return

        df, total = process_and_save(INPUT_SAMPLE, OUTPUT_SAMPLE)
        validate_and_report(df)
        print_top_candidates(df, n=10)
        print_reasoning_sample(df)

        print("✅ TEST MODE COMPLETE")
        print(f"   Metadata saved to: {OUTPUT_SAMPLE}")
        print(f"   Load in rank.py with:")
        print(f"   df = pd.read_parquet('artifacts/sample_metadata.parquet')\n")

    else:
        print("\n🚀 Running in FULL mode (100,000 candidates)")

        if not os.path.exists(INPUT_FULL):
            print(f"❌ ERROR: {INPUT_FULL} not found.")
            print("   Run 03_build_narratives.py --mode full first.")
            return

        df, total = process_and_save(INPUT_FULL, OUTPUT_FULL)
        validate_and_report(df)
        print_top_candidates(df, n=10)
        print_reasoning_sample(df)

        # Also save a small sample version for the sandbox demo
        print("\n  Building sandbox sample metadata (first 50 rows)...")
        sample_df = df.head(50)
        sample_df.to_parquet(OUTPUT_SAMPLE, index=False)
        print(f"  ✅ Sandbox metadata saved to: {OUTPUT_SAMPLE}")

        print("\n✅ FULL MODE COMPLETE")
        print(f"\n   Files ready:")
        for fpath in [OUTPUT_FULL, OUTPUT_SAMPLE]:
            if os.path.exists(fpath):
                size = os.path.getsize(fpath) / (1024*1024)
                print(f"   ✓ {fpath} ({size:.1f} MB)")

        print("\n   You're ready to run rank.py!")
        print("   python rank.py --candidates data/candidates.jsonl")
        print("                  --artifacts artifacts/")
        print("                  --jd data/job_description.md")
        print("                  --out submission.csv\n")


if __name__ == "__main__":
    main()