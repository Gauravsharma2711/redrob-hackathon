"""
03_build_narratives.py
-----------------------
WHAT THIS FILE DOES:
    Reads every scored candidate and builds a single rich text
    "narrative" per candidate.

    This narrative is what gets converted into a 768-dimensional
    embedding vector in the next step (04_embed_and_index.py).

    The quality of this narrative directly determines how well
    the semantic search finds the right candidates for a given JD.

WHY NARRATIVES MATTER:
    Embedding models understand language the way humans do.
    A candidate who "built a learning-to-rank system for product
    discovery at Swiggy" is semantically related to a JD asking for
    "ranking systems at product companies" — even if the words don't
    match exactly.

    A flat dump of skills and titles misses this. A well-structured
    narrative captures it.

NARRATIVE DESIGN PRINCIPLES:
    1. Lead with what matters most — headline + summary first
    2. Weight recent jobs more — only the last 3 jobs, in order
    3. Include verified signals — assessment scores are objective
    4. Be honest about gaps — if there's no AI work, say so
    5. Use natural language — not JSON, not bullet points
    6. Stay within 512 tokens — the embedding model's limit

HOW TO RUN:

    Test mode (50 candidates):
        python precompute/03_build_narratives.py --mode test

    Full mode (100,000 candidates):
        python precompute/03_build_narratives.py --mode full
"""

import json
import os
import argparse
from datetime import datetime

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────

INPUT_SAMPLE  = "artifacts/scored_sample.jsonl"
INPUT_FULL    = "artifacts/scored_candidates.jsonl"

OUTPUT_SAMPLE = "artifacts/narratives_sample.jsonl"
OUTPUT_FULL   = "artifacts/narratives_candidates.jsonl"

TODAY = datetime(2026, 6, 10)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Proficiency levels mapped to descriptive words
# Used to make the narrative read naturally
PROFICIENCY_WORDS = {
    "expert":       "expert-level",
    "advanced":     "advanced",
    "intermediate": "working knowledge of",
    "beginner":     "foundational"
}

# Education tier labels
# tier_1 = IITs, IIMs, top global universities
# tier_2 = NITs, good state universities
# tier_3 = decent private colleges
# tier_4 = local/lesser-known colleges
EDUCATION_TIER_LABELS = {
    "tier_1": "a top-tier institution",
    "tier_2": "a well-regarded institution",
    "tier_3": "a private university",
    "tier_4": "a local engineering college"
}

# Industries that are positive signals for this role
PRODUCT_INDUSTRIES = {
    "software", "fintech", "e-commerce", "food delivery",
    "transportation", "saas", "ai/ml", "edtech", "healthtech",
    "media", "gaming"
}

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def parse_date(date_string):
    try:
        return datetime.strptime(date_string, "%Y-%m-%d")
    except Exception:
        return None


def days_ago(date_string):
    """Returns how many days ago a date was from TODAY."""
    d = parse_date(date_string)
    if d:
        return (TODAY - d).days
    return 999


def get_top_skills(skills, max_skills=8):
    """
    Returns the most credible skills sorted by:
    1. Proficiency level (expert > advanced > intermediate > beginner)
    2. Number of endorsements (more = more credible)

    We cap at max_skills to keep the narrative focused.
    """
    PROFICIENCY_ORDER = {
        "expert": 4, "advanced": 3,
        "intermediate": 2, "beginner": 1
    }

    scored_skills = []
    for skill in skills:
        prof_score = PROFICIENCY_ORDER.get(
            skill.get("proficiency", "beginner"), 1
        )
        endorsements = skill.get("endorsements", 0)
        # Combined weight: proficiency matters more than endorsements
        weight = (prof_score * 10) + endorsements
        scored_skills.append((weight, skill))

    # Sort by weight descending
    scored_skills.sort(key=lambda x: x[0], reverse=True)

    return [s for _, s in scored_skills[:max_skills]]


def get_recent_jobs(career_history, max_jobs=3):
    """
    Returns the most recent N jobs sorted by start date.
    Recent experience is more relevant than old experience.
    """
    sorted_jobs = sorted(
        career_history,
        key=lambda j: j.get("start_date", "2000-01-01"),
        reverse=True
    )
    return sorted_jobs[:max_jobs]


def format_duration(months):
    """Converts months into human readable format."""
    if months < 12:
        return f"{months} months"
    years = months // 12
    remaining = months % 12
    if remaining == 0:
        return f"{years} year{'s' if years > 1 else ''}"
    return f"{years}y {remaining}m"


def get_company_type_label(job):
    """Returns a human-readable label for what kind of company this was."""
    industry = job.get("industry", "").lower()
    size     = job.get("company_size", "")

    if industry in PRODUCT_INDUSTRIES:
        if size in ["11-50", "51-200"]:
            return "an early-stage startup"
        elif size in ["201-500", "501-1000"]:
            return "a growth-stage company"
        elif size in ["1001-5000"]:
            return "a mid-sized product company"
        else:
            return "a product company"
    elif "it services" in industry or "consulting" in industry:
        return "an IT services firm"
    else:
        return f"a {industry} company" if industry else "a company"


# ─────────────────────────────────────────────
# CORE: BUILD THE NARRATIVE
# ─────────────────────────────────────────────

def build_narrative(candidate):
    """
    Builds a rich, natural-language narrative for one candidate.

    The narrative has 6 sections in order of importance:

        Section 1 — WHO THEY ARE
            Name, title, years of experience, location
            This is the opening statement.

        Section 2 — THEIR OWN WORDS
            Their summary/bio verbatim (truncated)
            This captures how they describe themselves.

        Section 3 — WHAT THEY'VE ACTUALLY DONE
            Recent 3 jobs with context about company type
            This is the most important section for semantic matching.

        Section 4 — WHAT THEY'RE GOOD AT
            Top skills with proficiency and verified scores
            Separates self-reported from verified skills.

        Section 5 — HOW THEY LEARNED
            Education background and tier
            Relevant for signal about rigor and background.

        Section 6 — AVAILABILITY CONTEXT
            Brief signal about their current availability
            Gives the embedding model context about reachability.
    """

    profile  = candidate["profile"]
    career   = candidate.get("career_history", [])
    skills   = candidate.get("skills", [])
    signals  = candidate["redrob_signals"]
    edu      = candidate.get("education", [])
    certs    = candidate.get("certifications", [])
    scores   = candidate.get("_scores", {})
    valid    = candidate.get("_validation", {})

    parts = []

    # ─────────────────────────────────────────
    # SECTION 1 — WHO THEY ARE
    # ─────────────────────────────────────────

    name     = profile.get("anonymized_name", "Candidate")
    title    = profile.get("current_title", "Professional")
    company  = profile.get("current_company", "")
    yoe      = profile.get("years_of_experience", 0)
    location = profile.get("location", "")
    country  = profile.get("country", "")
    industry = profile.get("current_industry", "")

    # Build the opening line
    location_str = f"{location}, {country}" if location else country
    company_str  = f" at {company}" if company else ""

    opening = (
        f"{name} is a {title}{company_str} with {yoe:.0f} years of "
        f"professional experience, currently based in {location_str}."
    )

    # Add industry context if it's a product company
    if industry.lower() in PRODUCT_INDUSTRIES:
        opening += f" They work in the {industry} industry."

    parts.append(opening)

    # ─────────────────────────────────────────
    # SECTION 2 — THEIR OWN WORDS (summary)
    # ─────────────────────────────────────────

    summary = profile.get("summary", "").strip()
    if summary:
        # Truncate at 400 characters to stay within token budget
        # but keep complete sentences
        if len(summary) > 400:
            truncated = summary[:400]
            # Find last full stop to not cut mid-sentence
            last_period = truncated.rfind(".")
            if last_period > 200:
                summary = truncated[:last_period + 1]
            else:
                summary = truncated + "..."

        parts.append(f"In their own words: {summary}")

    # ─────────────────────────────────────────
    # SECTION 3 — WHAT THEY'VE ACTUALLY DONE
    # ─────────────────────────────────────────

    recent_jobs = get_recent_jobs(career, max_jobs=3)

    if recent_jobs:
        parts.append("Career background:")

        for i, job in enumerate(recent_jobs):
            job_title    = job.get("title", "Unknown role")
            job_company  = job.get("company", "Unknown company")
            job_duration = format_duration(job.get("duration_months", 0))
            company_type = get_company_type_label(job)
            description  = job.get("description", "").strip()

            # Weight label: most recent = current, others = previous
            if i == 0 and job.get("is_current", False):
                recency_label = "Currently"
            elif i == 0:
                recency_label = "Most recently"
            elif i == 1:
                recency_label = "Previously"
            else:
                recency_label = "Earlier"

            # Build job sentence
            job_sentence = (
                f"{recency_label}, {job_title} at {job_company} "
                f"({company_type}, {job_duration})."
            )

            # Add description — truncate to 250 chars per job
            if description:
                if len(description) > 250:
                    truncated_desc = description[:250]
                    last_period    = truncated_desc.rfind(".")
                    if last_period > 100:
                        description = truncated_desc[:last_period + 1]
                    else:
                        description = truncated_desc + "..."

                job_sentence += f" {description}"

            parts.append(job_sentence)

    # ─────────────────────────────────────────
    # SECTION 4 — SKILLS AND VERIFIED SCORES
    # ─────────────────────────────────────────

    top_skills     = get_top_skills(skills, max_skills=8)
    skill_sections = []

    # Separate verified (has assessment score) from unverified
    assessment_scores = signals.get("skill_assessment_scores", {})
    verified_skills   = []
    unverified_skills = []

    for skill in top_skills:
        skill_name = skill.get("name", "")
        if skill_name in assessment_scores:
            score = assessment_scores[skill_name]
            verified_skills.append(
                f"{skill_name} (verified score: {score:.0f}/100)"
            )
        else:
            prof  = skill.get("proficiency", "beginner")
            label = PROFICIENCY_WORDS.get(prof, "knowledge of")
            end   = skill.get("endorsements", 0)
            end_str = f", {end} endorsements" if end > 0 else ""
            unverified_skills.append(f"{label} {skill_name}{end_str}")

    if verified_skills:
        skill_sections.append(
            "Verified skills (platform-assessed): "
            + "; ".join(verified_skills) + "."
        )

    if unverified_skills:
        skill_sections.append(
            "Additional skills: "
            + "; ".join(unverified_skills) + "."
        )

    if skill_sections:
        parts.extend(skill_sections)

    # Add certifications if any
    if certs:
        cert_names = [
            f"{c.get('name','')} ({c.get('issuer','')}, {c.get('year','')})"
            for c in certs[:3]  # max 3 certs
        ]
        parts.append(
            "Certifications: " + ", ".join(cert_names) + "."
        )

    # ─────────────────────────────────────────
    # SECTION 5 — EDUCATION
    # ─────────────────────────────────────────

    if edu:
        # Get the highest degree (last in list is usually highest)
        # Sort by end year descending to get most recent
        sorted_edu = sorted(
            edu,
            key=lambda e: e.get("end_year", 0),
            reverse=True
        )
        highest    = sorted_edu[0]
        degree     = highest.get("degree", "")
        field      = highest.get("field_of_study", "")
        tier       = highest.get("tier", "tier_4")
        tier_label = EDUCATION_TIER_LABELS.get(tier, "a university")

        if degree and field:
            parts.append(
                f"Education: {degree} in {field} from {tier_label}."
            )
        elif degree:
            parts.append(
                f"Education: {degree} from {tier_label}."
            )

    # ─────────────────────────────────────────
    # SECTION 6 — AVAILABILITY CONTEXT
    # ─────────────────────────────────────────

    # We include availability as context so the embedding model
    # can capture the semantic difference between an active
    # job-seeker and a passive one.

    availability_parts = []

    # Open to work
    if signals.get("open_to_work_flag", False):
        availability_parts.append("actively open to new opportunities")

    # Last active
    inactive_days = days_ago(signals.get("last_active_date", "2020-01-01"))
    if inactive_days <= 30:
        availability_parts.append("recently active on platform")
    elif inactive_days <= 90:
        availability_parts.append("moderately active on platform")
    else:
        availability_parts.append(
            f"last active {inactive_days} days ago"
        )

    # Notice period
    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        availability_parts.append(f"available within {notice} days")
    elif notice <= 60:
        availability_parts.append(f"{notice}-day notice period")
    else:
        availability_parts.append(f"longer notice period ({notice} days)")

    # Preferred work mode
    mode = signals.get("preferred_work_mode", "")
    if mode:
        availability_parts.append(f"prefers {mode} work")

    # Relocation
    if signals.get("willing_to_relocate", False):
        availability_parts.append("willing to relocate")

    if availability_parts:
        parts.append(
            "Availability: " + ", ".join(availability_parts) + "."
        )

    # ─────────────────────────────────────────
    # ASSEMBLE THE FULL NARRATIVE
    # ─────────────────────────────────────────

    narrative = " ".join(parts)

    return narrative


# ─────────────────────────────────────────────
# QUALITY CHECK
# ─────────────────────────────────────────────

def estimate_tokens(text):
    """
    Rough token estimate: ~1.3 tokens per word on average.
    The embedding model (bge-base-en-v1.5) has a 512-token limit.
    We target under 450 tokens to leave headroom.
    """
    word_count  = len(text.split())
    token_est   = int(word_count * 1.3)
    return token_est


def check_narrative_quality(narrative, candidate_id):
    """
    Runs basic quality checks on the generated narrative.
    Returns a list of warnings if anything looks wrong.
    """
    warnings = []

    # Too short — narrative has no useful content
    if len(narrative) < 100:
        warnings.append(f"{candidate_id}: narrative too short ({len(narrative)} chars)")

    # Too long — will be truncated by embedding model
    tokens = estimate_tokens(narrative)
    if tokens > 450:
        warnings.append(
            f"{candidate_id}: narrative may exceed token limit "
            f"(~{tokens} tokens estimated)"
        )

    # Check it doesn't have raw JSON artifacts
    if "{" in narrative and "}" in narrative:
        warnings.append(f"{candidate_id}: narrative may contain raw JSON")

    return warnings


# ─────────────────────────────────────────────
# PROCESS AND SAVE
# ─────────────────────────────────────────────

def process_and_save(input_path, output_path):
    """
    Reads scored candidates, builds narrative for each,
    and saves the result to a new JSONL file.

    Each output line is the original candidate dict
    with one new field added:
        candidate["_narrative"] = "the full text narrative"
    """

    os.makedirs("artifacts", exist_ok=True)

    total          = 0
    all_warnings   = []
    token_counts   = []

    print(f"\n⚙️  Building narratives...")
    print(f"📥 Input : {input_path}")
    print(f"📤 Output: {output_path}\n")

    with open(input_path,  "r", encoding="utf-8") as in_file, \
         open(output_path, "w", encoding="utf-8") as out_file:

        for line in in_file:
            line = line.strip()
            if not line:
                continue

            candidate = json.loads(line)
            total    += 1

            # Build the narrative
            narrative = build_narrative(candidate)

            # Quality check
            warnings = check_narrative_quality(
                narrative, candidate["candidate_id"]
            )
            all_warnings.extend(warnings)

            # Track token counts for reporting
            token_counts.append(estimate_tokens(narrative))

            # Attach narrative to candidate
            candidate["_narrative"] = narrative

            # Write to output
            out_file.write(json.dumps(candidate) + "\n")

            # Progress for large files
            if total % 10000 == 0:
                print(f"   Built {total} narratives...")

    return total, all_warnings, token_counts


# ─────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────

def print_report(total, warnings, token_counts):
    print("\n" + "="*60)
    print("         NARRATIVE BUILDER — SUMMARY REPORT")
    print("="*60)
    print(f"  Total narratives built : {total}")
    print(f"  Warnings               : {len(warnings)}")

    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        print(f"\n  Token estimates:")
        print(f"    Average : {avg_tokens:.0f} tokens")
        print(f"    Min     : {min(token_counts)} tokens")
        print(f"    Max     : {max(token_counts)} tokens")
        over_limit = sum(1 for t in token_counts if t > 450)
        if over_limit:
            print(f"    ⚠️  Over limit (>450): {over_limit} narratives")
        else:
            print(f"    ✅ All narratives within token limit")

    if warnings:
        print(f"\n  Warnings:")
        for w in warnings[:10]:   # show first 10 only
            print(f"    ⚠️  {w}")
        if len(warnings) > 10:
            print(f"    ... and {len(warnings)-10} more")

    print("="*60)


def print_sample_narratives(output_path, n=2):
    """
    Prints example narratives so you can read and verify
    they sound natural and informative.
    """
    candidates = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))

    # Sort by career score to show best and worst
    candidates.sort(
        key=lambda x: x.get("_scores", {}).get("career_score", 0),
        reverse=True
    )

    print(f"\n{'─'*70}")
    print("  SAMPLE NARRATIVES — verify these sound natural")
    print(f"{'─'*70}\n")

    # Show best candidate
    best = candidates[0]
    print(f"🥇 BEST CANDIDATE — {best['candidate_id']}")
    print(f"   Career Score: {best['_scores']['career_score']}")
    print(f"   Narrative:\n")
    # Print with word wrapping at 70 chars
    narrative = best["_narrative"]
    words = narrative.split()
    line_words = []
    for word in words:
        line_words.append(word)
        if len(" ".join(line_words)) > 70:
            print("   " + " ".join(line_words[:-1]))
            line_words = [word]
    if line_words:
        print("   " + " ".join(line_words))

    print()

    # Show a weak candidate for contrast
    weak = candidates[-1]
    print(f"📉 WEAK CANDIDATE — {weak['candidate_id']}")
    print(f"   Career Score: {weak['_scores']['career_score']}")
    print(f"   Narrative:\n")
    narrative = weak["_narrative"]
    words = narrative.split()
    line_words = []
    for word in words:
        line_words.append(word)
        if len(" ".join(line_words)) > 70:
            print("   " + " ".join(line_words[:-1]))
            line_words = [word]
    if line_words:
        print("   " + " ".join(line_words))

    print(f"\n{'─'*70}\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build rich text narratives for candidate embedding"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "full"],
        default="test",
        help="'test' uses sample data, 'full' uses 100K dataset"
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("      REDROB HACKATHON — 03 BUILD NARRATIVES")
    print("="*60)

    if args.mode == "test":
        print("\n🧪 Running in TEST mode")

        if not os.path.exists(INPUT_SAMPLE):
            print(f"\n❌ ERROR: {INPUT_SAMPLE} not found.")
            print("   Run 02_feature_engineering.py --mode test first.")
            return

        total, warnings, tokens = process_and_save(
            INPUT_SAMPLE, OUTPUT_SAMPLE
        )
        print_report(total, warnings, tokens)
        print_sample_narratives(OUTPUT_SAMPLE, n=2)

    else:
        print("\n🚀 Running in FULL mode")

        if not os.path.exists(INPUT_FULL):
            print(f"\n❌ ERROR: {INPUT_FULL} not found.")
            print("   Run 02_feature_engineering.py --mode full first.")
            return

        total, warnings, tokens = process_and_save(
            INPUT_FULL, OUTPUT_FULL
        )
        print_report(total, warnings, tokens)
        print_sample_narratives(OUTPUT_FULL, n=2)


if __name__ == "__main__":
    main()
