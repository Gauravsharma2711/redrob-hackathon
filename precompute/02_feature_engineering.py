"""
02_feature_engineering.py
--------------------------
WHAT THIS FILE DOES:
    Reads the validated candidates from 01_parse_and_validate.py
    and computes three things for every candidate:

    1. career_score     (0-100) — How well does their actual work
                                  history match what the JD needs?

    2. behavioral_score (0-100) — Are they actually available
                                  and reachable right now?

    3. disqualifier_flags       — Should they be excluded completely?
                                  (consulting-only, non-technical, etc.)

    These scores are saved alongside the candidate data and used
    later in rank.py to produce the final composite score.

HOW TO RUN:

    Test mode (uses validated_sample.jsonl — 50 candidates):
        python precompute/02_feature_engineering.py --mode test

    Full mode (uses validated_candidates.jsonl — 100,000 candidates):
        python precompute/02_feature_engineering.py --mode full
"""

import json
import os
import argparse
from datetime import datetime

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────

INPUT_SAMPLE = "artifacts/validated_sample.jsonl"
INPUT_FULL   = "artifacts/validated_candidates.jsonl"

OUTPUT_SAMPLE = "artifacts/scored_sample.jsonl"
OUTPUT_FULL   = "artifacts/scored_candidates.jsonl"

TODAY = datetime(2026, 6, 10)

# ─────────────────────────────────────────────
# CAREER SCORE CONSTANTS
# ─────────────────────────────────────────────

# Keywords that signal real AI/ML/Search work in career descriptions
# We search for these inside job descriptions, NOT just the skills list
AI_KEYWORDS_IN_CAREER = [
    # Core retrieval and ranking (highest signal — exactly what JD wants)
    "embedding", "embeddings", "vector search", "vector database",
    "semantic search", "retrieval", "ranking", "reranking", "re-ranking",
    "recommendation", "recommendation system", "search engine",
    "information retrieval", "hybrid search", "dense retrieval",

    # LLM and NLP work
    "language model", "large language model", "llm", "nlp",
    "natural language processing", "fine-tun", "fine tuning",
    "rag", "retrieval augmented", "transformer", "bert", "gpt",
    "text classification", "named entity", "sentiment",

    # ML infrastructure (shows production experience)
    "machine learning", "deep learning", "neural network",
    "model serving", "model deployment", "inference", "mlops",
    "feature store", "training pipeline", "a/b test",

    # Specific tools the JD mentions
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "elasticsearch", "opensearch", "solr",
    "sentence-transformer", "huggingface", "pytorch", "tensorflow",
    "scikit-learn", "sklearn", "xgboost",

    # Evaluation (JD explicitly asks for this)
    "ndcg", "mrr", "map", "precision@", "recall@",
    "evaluation framework", "offline evaluation", "online evaluation",
    "a/b testing", "experimentation"
]

# Keywords in SKILLS LIST that signal AI/ML expertise
AI_SKILL_NAMES = [
    "nlp", "machine learning", "deep learning", "pytorch", "tensorflow",
    "transformers", "bert", "llm", "fine-tuning", "lora", "qlora",
    "rag", "vector search", "embeddings", "faiss", "milvus", "qdrant",
    "pinecone", "weaviate", "elasticsearch", "recommendation systems",
    "information retrieval", "ranking", "xgboost", "scikit-learn",
    "mlops", "kubeflow", "mlflow", "weights & biases", "huggingface",
    "sentence-transformers", "bm25", "sparse retrieval", "dense retrieval",
    "recsys", "collaborative filtering", "feature engineering",
    "model deployment", "model serving", "a/b testing", "ndcg"
]

# Companies/industries that count as IT services / consulting
# The JD explicitly says pure consulting background is a disqualifier
CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis", "hexaware",
    "l&t infotech", "ltimindtree", "persistent", "kpit", "mastek",
    "niit technologies", "sonata software"
}

CONSULTING_INDUSTRIES = {
    "it services", "consulting", "bpo", "outsourcing"
}

# Companies/industries that count as product companies
# (startup, e-commerce, fintech, software product)
PRODUCT_INDUSTRIES = {
    "software", "fintech", "e-commerce", "food delivery",
    "transportation", "saas", "ai/ml", "edtech", "healthtech",
    "media", "gaming", "telecommunications"
}

# Current titles that are clearly non-technical
# Used to flag candidates whose background doesn't match the JD at all
NON_TECHNICAL_TITLES = [
    "marketing manager", "operations manager", "accountant",
    "hr manager", "customer support", "sales manager",
    "civil engineer", "mechanical engineer", "graphic designer",
    "content writer", "business development", "finance manager",
    "supply chain", "procurement", "legal"
]

# India cities the JD prefers (Pune/Noida primarily, others acceptable)
PREFERRED_CITIES = [
    "pune", "noida", "delhi", "ncr", "gurgaon", "gurugram",
    "hyderabad", "mumbai", "bangalore", "bengaluru", "chennai"
]

# ─────────────────────────────────────────────
# HELPER: PARSE DATE
# ─────────────────────────────────────────────

def parse_date(date_string):
    """Converts '2024-03-15' into a datetime object."""
    try:
        return datetime.strptime(date_string, "%Y-%m-%d")
    except Exception:
        return None


# ─────────────────────────────────────────────
# SCORE 1: CAREER QUALITY SCORE
# ─────────────────────────────────────────────

def compute_career_score(candidate):
    """
    Scores how well a candidate's actual career matches the JD.
    Maximum score: 100 points.

    Broken into 5 sub-signals:
        1. AI/ML work in career descriptions    → up to 35 points
        2. AI/ML skills in skills list          → up to 15 points
        3. Product company experience           → up to 20 points
        4. Years of experience band             → up to 20 points
        5. Seniority of current title           → up to 10 points

    Location is handled separately as a bonus/penalty in rank.py
    so it doesn't inflate this score.
    """

    profile = candidate["profile"]
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])
    score   = 0.0

    # ── SUB-SIGNAL 1: AI/ML keywords in career descriptions ─────────
    # We look at the last 3 jobs only (recent experience matters more)
    # Each unique keyword found adds points, capped at 35
    #
    # Why career descriptions and not just skills?
    # Because the JD says: "A Tier 5 candidate may not use the words
    # 'RAG' or 'Pinecone' in their profile, but if their career history
    # shows they built a recommendation system at a product company,
    # they're a fit."

    recent_jobs = sorted(
        career,
        key=lambda j: j.get("start_date", "2000-01-01"),
        reverse=True
    )[:3]  # only look at the 3 most recent jobs

    combined_career_text = " ".join([
        j.get("description", "") for j in recent_jobs
    ]).lower()

    unique_ai_hits = set()
    for keyword in AI_KEYWORDS_IN_CAREER:
        if keyword.lower() in combined_career_text:
            unique_ai_hits.add(keyword)

    # Each unique AI keyword is worth 2.5 points, capped at 35
    career_ai_points = min(len(unique_ai_hits) * 2.5, 35)
    score += career_ai_points

    # ── SUB-SIGNAL 2: AI/ML skills in skills list ───────────────────
    # Check skill names against our AI skill list
    # Also weight by proficiency level and endorsements
    #
    # Proficiency weights:
    #   advanced     = 3 points
    #   intermediate = 2 points
    #   beginner     = 1 point

    PROFICIENCY_WEIGHT = {"advanced": 3, "intermediate": 2, "beginner": 1}

    skill_ai_points = 0.0
    for skill in skills:
        skill_name = skill.get("name", "").lower()
        if any(ai_skill in skill_name for ai_skill in AI_SKILL_NAMES):
            level  = skill.get("proficiency", "beginner")
            weight = PROFICIENCY_WEIGHT.get(level, 1)
            skill_ai_points += weight * 0.8  # each matching skill adds up

    skill_ai_points = min(skill_ai_points, 15)  # cap at 15
    score += skill_ai_points

    # ── SUB-SIGNAL 3: Product company experience ────────────────────
    # The JD explicitly rejects pure consulting backgrounds.
    # We calculate what fraction of their career was at product companies.
    #
    # product_ratio = product_months / (product_months + consulting_months)
    # A person who spent all their time at product companies scores 20.
    # A person who spent all their time at consulting firms scores 0.

    product_months    = 0
    consulting_months = 0

    for job in career:
        company_name  = job.get("company", "").lower()
        industry      = job.get("industry", "").lower()
        months        = job.get("duration_months", 0)

        # Check if this job is at a consulting firm
        is_consulting = (
            any(firm in company_name for firm in CONSULTING_COMPANIES)
            or any(ind in industry for ind in CONSULTING_INDUSTRIES)
        )

        # Check if this job is at a product company
        is_product = (
            any(ind in industry for ind in PRODUCT_INDUSTRIES)
            and not is_consulting
        )

        if is_consulting:
            consulting_months += months
        if is_product:
            product_months += months

    total_categorised = product_months + consulting_months
    if total_categorised > 0:
        product_ratio = product_months / total_categorised
    else:
        product_ratio = 0.5  # unknown — give benefit of the doubt

    score += product_ratio * 20  # up to 20 points

    # ── SUB-SIGNAL 4: Years of experience band ──────────────────────
    # JD says 5-9 years is the target range.
    # We also check for judgment quality, not just years.

    yoe = profile.get("years_of_experience", 0)

    if 5 <= yoe <= 9:
        score += 20        # perfect band
    elif 4 <= yoe < 5:
        score += 14        # slightly junior but close
    elif 9 < yoe <= 12:
        score += 12        # slightly senior, still fine
    elif 3 <= yoe < 4:
        score += 7         # junior, possible but weak
    elif yoe > 12:
        score += 6         # over-experienced, less ideal
    else:
        score += 0         # under 3 years — too junior

    # ── SUB-SIGNAL 5: Current title seniority ───────────────────────
    # A senior/lead/principal title suggests the person has
    # already reached the level the JD is hiring for.

    current_title = profile.get("current_title", "").lower()

    SENIOR_TITLES = [
        "senior", "lead", "principal", "staff", "architect",
        "head of", "founding", "director", "vp of"
    ]

    if any(title in current_title for title in SENIOR_TITLES):
        score += 10
    elif any(word in current_title for word in ["engineer", "scientist", "developer", "researcher"]):
        score += 5   # technical title but not senior yet
    else:
        score += 0   # non-technical title

    return round(min(score, 100), 2)


# ─────────────────────────────────────────────
# SCORE 2: BEHAVIORAL AVAILABILITY SCORE
# ─────────────────────────────────────────────

def compute_behavioral_score(candidate):
    """
    Scores whether the candidate is actually reachable right now.
    Maximum score: 100 points.

    The JD says:
    "A perfect-on-paper candidate who hasn't logged in for 6 months
    and has a 5% response rate is, for hiring purposes, not actually
    available. Down-weight them appropriately."

    Broken into 6 sub-signals:
        1. Days since last login      → up to 25 points
        2. Open to work flag          → 15 points
        3. Recruiter response rate    → up to 20 points
        4. Notice period              → up to 15 points
        5. GitHub activity            → up to 15 points
        6. Interview completion rate  → up to 10 points
    """

    signals = candidate["redrob_signals"]
    score   = 0.0

    # ── SUB-SIGNAL 1: Days since last login ─────────────────────────
    # This is the most important availability signal.
    # Someone who hasn't logged in for 6 months is effectively gone.

    last_active = parse_date(signals.get("last_active_date", "2020-01-01"))
    if last_active:
        days_inactive = (TODAY - last_active).days
    else:
        days_inactive = 999  # unknown — treat as very inactive

    if days_inactive <= 14:
        score += 25    # active in last 2 weeks — excellent
    elif days_inactive <= 30:
        score += 22    # active in last month — great
    elif days_inactive <= 60:
        score += 17    # active in last 2 months — good
    elif days_inactive <= 90:
        score += 12    # active in last 3 months — okay
    elif days_inactive <= 180:
        score += 6     # 3-6 months inactive — weak signal
    else:
        score += 0     # 6+ months inactive — essentially unavailable

    # ── SUB-SIGNAL 2: Open to work flag ─────────────────────────────
    # The candidate has explicitly marked themselves as available.
    # This is a direct opt-in signal.

    if signals.get("open_to_work_flag", False):
        score += 15

    # ── SUB-SIGNAL 3: Recruiter response rate ───────────────────────
    # How often do they actually reply to recruiters?
    # A 10% response rate means 9 out of 10 messages go unanswered.

    response_rate = signals.get("recruiter_response_rate", 0)
    score += response_rate * 20   # 0.0-1.0 multiplied by 20 = 0-20 points

    # ── SUB-SIGNAL 4: Notice period ─────────────────────────────────
    # JD says: "We'd love sub-30-day notice.
    #           We can buy out up to 30 days.
    #           30+ day candidates are still in scope but the bar gets higher."

    notice = signals.get("notice_period_days", 90)

    if notice <= 15:
        score += 15    # immediate joiner — ideal
    elif notice <= 30:
        score += 13    # within buyout window — great
    elif notice <= 60:
        score += 8     # 2 months — manageable
    elif notice <= 90:
        score += 4     # 3 months — less ideal
    else:
        score += 0     # 90+ days — the bar gets higher

    # ── SUB-SIGNAL 5: GitHub activity ───────────────────────────────
    # For an AI engineer role, active GitHub is a proxy for real work.
    # Score of -1 means no GitHub linked — not penalised, just no bonus.

    github = signals.get("github_activity_score", -1)

    if github >= 70:
        score += 15    # very active — strong signal
    elif github >= 50:
        score += 12
    elif github >= 30:
        score += 8
    elif github >= 10:
        score += 4
    elif github >= 0:
        score += 1     # has GitHub but low activity
    else:
        score += 0     # no GitHub linked (-1)

    # ── SUB-SIGNAL 6: Interview completion rate ─────────────────────
    # Shows up to interviews they agree to.
    # Low rate = likely to ghost = waste of recruiter time.

    icr = signals.get("interview_completion_rate", 0.5)
    score += icr * 10   # 0.0-1.0 multiplied by 10 = 0-10 points

    return round(min(score, 100), 2)


# ─────────────────────────────────────────────
# SCORE 3: DISQUALIFIER FLAGS
# ─────────────────────────────────────────────

def compute_disqualifiers(candidate):
    """
    Checks for hard disqualifying conditions.
    These are NOT score deductions — they are boolean flags.
    A candidate with any True flag will be excluded in rank.py.

    Returns a dictionary of True/False flags.
    """

    profile = candidate["profile"]
    career  = candidate.get("career_history", [])
    signals = candidate["redrob_signals"]
    flags   = {}

    # ── FLAG 1: Pure consulting background ──────────────────────────
    # The JD says: people who have ONLY worked at consulting firms
    # are not considered. "If you're currently at one of these
    # companies but have prior product-company experience, that's fine."
    #
    # We flag only if ALL jobs are at consulting firms.

    if len(career) > 0:
        all_consulting = all(
            any(firm in job.get("company", "").lower()
                for firm in CONSULTING_COMPANIES)
            or any(ind in job.get("industry", "").lower()
                   for ind in CONSULTING_INDUSTRIES)
            for job in career
        )
        flags["pure_consulting_only"] = all_consulting
    else:
        flags["pure_consulting_only"] = False

    # ── FLAG 2: Non-technical current role with no AI history ────────
    # A Marketing Manager who lists AI skills is a keyword stuffer.
    # But we only flag them if there are also no AI signals in their
    # career history — giving benefit of the doubt for career changers.

    current_title       = profile.get("current_title", "").lower()
    is_non_tech_title   = any(
        nt in current_title for nt in NON_TECHNICAL_TITLES
    )

    combined_career_text = " ".join([
        j.get("description", "") for j in career
    ]).lower()

    has_any_ai_in_career = any(
        kw in combined_career_text for kw in AI_KEYWORDS_IN_CAREER
    )

    flags["non_technical_no_ai_history"] = (
        is_non_tech_title and not has_any_ai_in_career
    )

    # ── FLAG 3: Too junior ───────────────────────────────────────────
    # Under 3 years total experience.
    # Even with great skills, this role needs production judgment.

    yoe = profile.get("years_of_experience", 0)
    flags["too_junior"] = yoe < 3

    # ── FLAG 4: Salary expectation is unrealistic ────────────────────
    # If minimum expected salary exceeds 80 LPA, they are
    # priced out of a typical Series A AI engineering role.

    salary  = signals.get("expected_salary_range_inr_lpa", {})
    sal_min = salary.get("min", 0)
    flags["salary_too_high"] = sal_min > 80

    # ── COMBINED: Is this candidate disqualified? ────────────────────
    flags["is_disqualified"] = any([
        flags["pure_consulting_only"],
        flags["non_technical_no_ai_history"],
        flags["too_junior"],
        flags["salary_too_high"]
    ])

    return flags


# ─────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ─────────────────────────────────────────────

def process_and_save(input_path, output_path):
    """
    Reads validated candidates, computes all three scores,
    and saves the enriched data to a new JSONL file.
    """

    os.makedirs("artifacts", exist_ok=True)

    # Counters for the summary report
    total          = 0
    disqualified   = 0
    honeypots      = 0

    career_scores_list    = []
    behavioral_scores_list = []

    print(f"\n⚙️  Computing scores...")
    print(f"📥 Input : {input_path}")
    print(f"📤 Output: {output_path}\n")

    with open(input_path, "r", encoding="utf-8") as in_file, \
         open(output_path, "w", encoding="utf-8") as out_file:

        for line in in_file:
            line = line.strip()
            if not line:
                continue

            candidate = json.loads(line)
            total += 1

            # ── Skip structure-invalid candidates ──
            validation = candidate.get("_validation", {})
            if not validation.get("is_valid", True):
                # Still write them but with zero scores
                candidate["_scores"] = {
                    "career_score": 0,
                    "behavioral_score": 0,
                    "disqualifiers": {"is_disqualified": True}
                }
                out_file.write(json.dumps(candidate) + "\n")
                continue

            # ── Compute the three scores ──
            career_score     = compute_career_score(candidate)
            behavioral_score = compute_behavioral_score(candidate)
            disqualifiers    = compute_disqualifiers(candidate)

            # ── Track stats ──
            career_scores_list.append(career_score)
            behavioral_scores_list.append(behavioral_score)

            if disqualifiers["is_disqualified"]:
                disqualified += 1

            if validation.get("is_honeypot", False):
                honeypots += 1

            # ── Attach scores to the candidate ──
            candidate["_scores"] = {
                "career_score":     career_score,
                "behavioral_score": behavioral_score,
                "disqualifiers":    disqualifiers
            }

            out_file.write(json.dumps(candidate) + "\n")

            # ── Progress update for large files ──
            if total % 10000 == 0:
                print(f"   Processed {total} candidates...")

    return total, disqualified, honeypots, career_scores_list, behavioral_scores_list


# ─────────────────────────────────────────────
# PRINT SUMMARY REPORT
# ─────────────────────────────────────────────

def print_report(total, disqualified, honeypots, career_scores, behavioral_scores):
    """Prints a clear summary of computed scores."""

    print("\n" + "="*55)
    print("       FEATURE ENGINEERING — SUMMARY REPORT")
    print("="*55)
    print(f"  Total candidates scored   : {total}")
    print(f"  Disqualified              : {disqualified} "
          f"({disqualified/total*100:.1f}%)")
    print(f"  Honeypots (from Phase 1)  : {honeypots}")
    print()

    if career_scores:
        print(f"  Career Score distribution:")
        print(f"    Min   : {min(career_scores):.1f}")
        print(f"    Max   : {max(career_scores):.1f}")
        print(f"    Avg   : {sum(career_scores)/len(career_scores):.1f}")

    if behavioral_scores:
        print(f"\n  Behavioral Score distribution:")
        print(f"    Min   : {min(behavioral_scores):.1f}")
        print(f"    Max   : {max(behavioral_scores):.1f}")
        print(f"    Avg   : {sum(behavioral_scores)/len(behavioral_scores):.1f}")

    print("="*55)


def print_top_candidates(output_path, n=10):
    """
    Reads the output file and prints the top N candidates
    by career score so you can verify the scoring makes sense.
    """
    scored = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                sc = c.get("_scores", {})
                if sc.get("career_score", 0) > 0:
                    scored.append(c)

    # Sort by career score descending
    scored.sort(
        key=lambda x: x["_scores"]["career_score"],
        reverse=True
    )

    print(f"\n{'─'*90}")
    print(f"  TOP {n} CANDIDATES BY CAREER SCORE")
    print(f"{'─'*90}")
    print(f"  {'ID':<15} {'Title':<32} {'Career':>7} {'Behav':>7} {'Disq?':>6}")
    print(f"{'─'*90}")

    for c in scored[:n]:
        sc   = c["_scores"]
        p    = c["profile"]
        disq = "YES" if sc["disqualifiers"]["is_disqualified"] else "no"

        print(
            f"  {c['candidate_id']:<15} "
            f"{p['current_title'][:30]:<32} "
            f"{sc['career_score']:>7.1f} "
            f"{sc['behavioral_score']:>7.1f} "
            f"{disq:>6}"
        )

    print(f"{'─'*90}\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute feature scores for all candidates"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "full"],
        default="test",
        help="'test' uses sample data, 'full' uses 100K dataset"
    )
    args = parser.parse_args()

    print("\n" + "="*55)
    print("     REDROB HACKATHON — 02 FEATURE ENGINEERING")
    print("="*55)

    if args.mode == "test":
        print("\n🧪 Running in TEST mode")

        if not os.path.exists(INPUT_SAMPLE):
            print(f"\n❌ ERROR: {INPUT_SAMPLE} not found.")
            print("   Run 01_parse_and_validate.py --mode test first.")
            return

        total, disq, honeys, c_scores, b_scores = process_and_save(
            INPUT_SAMPLE, OUTPUT_SAMPLE
        )
        print_report(total, disq, honeys, c_scores, b_scores)
        print_top_candidates(OUTPUT_SAMPLE, n=10)

    else:
        print("\n🚀 Running in FULL mode")

        if not os.path.exists(INPUT_FULL):
            print(f"\n❌ ERROR: {INPUT_FULL} not found.")
            print("   Run 01_parse_and_validate.py --mode full first.")
            return

        total, disq, honeys, c_scores, b_scores = process_and_save(
            INPUT_FULL, OUTPUT_FULL
        )
        print_report(total, disq, honeys, c_scores, b_scores)
        print_top_candidates(OUTPUT_FULL, n=10)


if __name__ == "__main__":
    main()