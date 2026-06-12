"""
01_parse_and_validate.py
------------------------
WHAT THIS FILE DOES:
    - Reads candidate data (either sample or full dataset)
    - Checks every candidate for honeypot signals (fake/impossible profiles)
    - Validates that required fields exist
    - Saves a clean output file with honeypot flags added

HOW TO RUN:

    Test mode (uses sample_candidates.json — 50 candidates):
        python precompute/01_parse_and_validate.py --mode test

    Full mode (uses candidates.jsonl.gz — 100,000 candidates):
        python precompute/01_parse_and_validate.py --mode full
"""

import json
import gzip
import os
import argparse
from datetime import datetime

# ─────────────────────────────────────────────
# SETTINGS — paths to your files
# ─────────────────────────────────────────────

# Input files
SAMPLE_FILE   = "data/sample_candidates.json"
FULL_FILE     = "data/candidates.jsonl"

# Output files (saved into artifacts/)
OUTPUT_SAMPLE = "artifacts/validated_sample.jsonl"
OUTPUT_FULL   = "artifacts/validated_candidates.jsonl"

# Today's date — used for honeypot timeline checks
TODAY = datetime(2026, 6, 10)

# ─────────────────────────────────────────────
# STEP 1 — READ THE DATA
# ─────────────────────────────────────────────

def load_sample(filepath):
    """
    Reads sample_candidates.json
    This file is a JSON array — one big list of candidates.
    Returns a list of candidate dictionaries.
    """
    print(f"\n📂 Loading sample file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"✅ Loaded {len(candidates)} candidates from sample file")
    return candidates


def load_full_dataset(filepath):
    """
    Reads candidates.jsonl.gz line by line WITHOUT loading
    everything into memory at once.
    Each line is one candidate JSON object.
    Yields one candidate at a time.
    """
    print(f"\n📂 Loading full dataset: {filepath}")
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:                          # skip empty lines
                candidate = json.loads(line)  # convert text to dictionary
                count += 1
                if count % 10000 == 0:
                    print(f"   ...read {count} candidates so far")
                yield candidate
    print(f"✅ Finished reading {count} candidates")


# ─────────────────────────────────────────────
# STEP 2 — VALIDATE REQUIRED FIELDS
# ─────────────────────────────────────────────

# These are the fields every candidate MUST have
REQUIRED_FIELDS = [
    "candidate_id",
    "profile",
    "career_history",
    "skills",
    "redrob_signals"
]

REQUIRED_SIGNAL_FIELDS = [
    "last_active_date",
    "signup_date",
    "expected_salary_range_inr_lpa",
    "notice_period_days",
    "recruiter_response_rate",
    "github_activity_score",
    "open_to_work_flag",
    "interview_completion_rate"
]

def validate_structure(candidate):
    """
    Checks that a candidate has all required fields.
    Returns (True, None) if valid.
    Returns (False, reason) if something is missing.
    """
    # Check top-level fields
    for field in REQUIRED_FIELDS:
        if field not in candidate:
            return False, f"Missing required field: {field}"

    # Check signal fields
    signals = candidate.get("redrob_signals", {})
    for field in REQUIRED_SIGNAL_FIELDS:
        if field not in signals:
            return False, f"Missing signal field: {field}"

    # Check salary range exists and has min/max
    salary = signals.get("expected_salary_range_inr_lpa", {})
    if "min" not in salary or "max" not in salary:
        return False, "Missing salary min or max"

    return True, None


# ─────────────────────────────────────────────
# STEP 3 — HONEYPOT DETECTION
# ─────────────────────────────────────────────

def parse_date(date_string):
    """
    Converts a date string like "2024-03-15" into a datetime object
    so we can do date comparisons.
    """
    return datetime.strptime(date_string, "%Y-%m-%d")


def detect_honeypot(candidate):
    """
    Checks a candidate for impossible or suspicious data patterns.
    The dataset contains ~80 honeypots. Ranking too many of them
    in your top 100 gets you disqualified.

    Returns a dictionary:
    {
        "is_honeypot": True or False,
        "reasons": ["list of reasons why it's flagged"]
    }
    """
    reasons = []
    signals  = candidate["redrob_signals"]
    career   = candidate.get("career_history", [])
    skills   = candidate.get("skills", [])

    # ── CHECK 1: Salary min is greater than salary max ──────────────
    # Example: min=50 LPA but max=30 LPA — impossible
    salary = signals["expected_salary_range_inr_lpa"]
    sal_min = salary.get("min", 0)
    sal_max = salary.get("max", 0)
    if sal_min > sal_max :
        reasons.append(
            f"Impossible salary: min ({sal_min}) > max ({sal_max})"
        )

    # ── CHECK 2: Signup date is AFTER last active date ───────────────
    # You can't be active on the platform before you signed up
    try:
        signup      = parse_date(signals["signup_date"])
        last_active = parse_date(signals["last_active_date"])
        if signup > last_active:
            reasons.append(
                f"Signup date ({signals['signup_date']}) is after "
                f"last active date ({signals['last_active_date']})"
            )
    except Exception:
        reasons.append("Invalid date format in signup_date or last_active_date")

    # ── CHECK 3: Job started in the future ───────────────────────────
    # A current job that started after today is impossible
    for job in career:
        try:
            start = parse_date(job["start_date"])
            if start > TODAY:
                reasons.append(
                    f"Job at {job.get('company','unknown')} starts "
                    f"in the future: {job['start_date']}"
                )
        except Exception:
            pass

    # ── CHECK 4: Job duration is impossible ─────────────────────────
    # If a job shows 150 months but only started 12 months ago,
    # the numbers don't add up
    for job in career:
        try:
            start          = parse_date(job["start_date"])
            claimed_months = job.get("duration_months", 0)
            actual_months  = (TODAY - start).days / 30

            # Allow 3 months buffer for rounding
            if claimed_months > actual_months + 3:
                reasons.append(
                    f"Impossible job duration at {job.get('company','unknown')}: "
                    f"claims {claimed_months} months but only "
                    f"{int(actual_months)} months have passed since start date"
                )
        except Exception:
            pass

    # ── CHECK 5: Too many advanced skills with zero endorsements ──────
    # Real advanced skills get endorsed by colleagues.
    # 8+ advanced skills with 0 total endorsements is suspicious.
    advanced_skills = [
        s for s in skills
        if s.get("proficiency") == "advanced"
    ]
    if len(advanced_skills) >= 8:
        total_endorsements = sum(
            s.get("endorsements", 0) for s in advanced_skills
        )
        if total_endorsements == 0:
            reasons.append(
                f"Suspicious: {len(advanced_skills)} advanced skills "
                f"but 0 total endorsements"
            )

    return {
        "is_honeypot": len(reasons) > 0,
        "honeypot_reasons": reasons
    }


# ─────────────────────────────────────────────
# STEP 4 — PROCESS AND SAVE
# ─────────────────────────────────────────────

def process_and_save(candidates_iterable, output_path, total_expected=None):
    """
    Loops through every candidate, runs validation and honeypot detection,
    adds the results as new fields, and saves to a JSONL output file.

    JSONL format = one JSON object per line.
    This is memory-efficient for large files.
    """

    # Make sure the artifacts folder exists
    os.makedirs("artifacts", exist_ok=True)

    # Counters for the summary report
    total          = 0
    valid          = 0
    invalid        = 0
    honeypots      = 0

    print(f"\n⚙️  Processing candidates...")
    print(f"📝 Output will be saved to: {output_path}\n")

    with open(output_path, "w", encoding="utf-8") as out_file:

        for candidate in candidates_iterable:
            total += 1

            # ── Validate structure ──
            is_valid, invalid_reason = validate_structure(candidate)

            if not is_valid:
                invalid += 1
                # Still save it, but mark as invalid so we can skip later
                candidate["_validation"] = {
                    "is_valid": False,
                    "invalid_reason": invalid_reason,
                    "is_honeypot": False,
                    "honeypot_reasons": []
                }
                out_file.write(json.dumps(candidate) + "\n")
                continue

            # ── Detect honeypot ──
            honeypot_result = detect_honeypot(candidate)

            if honeypot_result["is_honeypot"]:
                honeypots += 1

            # ── Add validation results to candidate ──
            candidate["_validation"] = {
                "is_valid": True,
                "invalid_reason": None,
                "is_honeypot": honeypot_result["is_honeypot"],
                "honeypot_reasons": honeypot_result["honeypot_reasons"]
            }

            valid += 1

            # ── Write this candidate to the output file ──
            out_file.write(json.dumps(candidate) + "\n")

            # ── Progress update for full dataset ──
            if total_expected and total % 10000 == 0:
                print(f"   Processed {total}/{total_expected}...")

    return total, valid, invalid, honeypots


# ─────────────────────────────────────────────
# STEP 5 — PRINT SUMMARY REPORT
# ─────────────────────────────────────────────

def print_report(total, valid, invalid, honeypots, output_path):
    """
    Prints a clear summary of what was found.
    """
    print("\n" + "="*55)
    print("         PARSE & VALIDATE — SUMMARY REPORT")
    print("="*55)
    print(f"  Total candidates processed : {total}")
    print(f"  Valid candidates           : {valid}")
    print(f"  Invalid (missing fields)   : {invalid}")
    print(f"  Honeypots detected         : {honeypots}")
    print(f"  Honeypot rate              : "
          f"{(honeypots/total*100):.2f}% of total")
    print("-"*55)
    print(f"  Output saved to: {output_path}")
    print("="*55)

    # Warning if honeypot rate looks high
    if honeypots / total > 0.05:
        print("\n⚠️  WARNING: Honeypot rate is above 5%.")
        print("   If this continues in the full dataset, review your")
        print("   detection logic — you want to catch them but not")
        print("   over-flag real candidates.\n")
    else:
        print("\n✅ Honeypot rate looks healthy.\n")


# ─────────────────────────────────────────────
# MAIN — runs when you execute this file
# ─────────────────────────────────────────────

def main():
    # Read the --mode argument from the command line
    parser = argparse.ArgumentParser(
        description="Parse and validate candidate data"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "full"],
        default="test",
        help="'test' uses sample_candidates.json, 'full' uses candidates.jsonl.gz"
    )
    args = parser.parse_args()

    print("\n" + "="*55)
    print("       REDROB HACKATHON — 01 PARSE & VALIDATE")
    print("="*55)

    if args.mode == "test":
        # ── TEST MODE ──
        print("\n🧪 Running in TEST mode (sample_candidates.json)")
        candidates_list = load_sample(SAMPLE_FILE)
        total, valid, invalid, honeypots = process_and_save(
            candidates_list,
            OUTPUT_SAMPLE,
            total_expected=len(candidates_list)
        )
        print_report(total, valid, invalid, honeypots, OUTPUT_SAMPLE)

    else:
        # ── FULL MODE ──
        print("\n🚀 Running in FULL mode (candidates.jsonl.gz)")

        # Check the file exists before starting
        if not os.path.exists(FULL_FILE):
            print(f"\n❌ ERROR: {FULL_FILE} not found.")
            print("   Place candidates.jsonl.gz in your data/ folder first.")
            return

        candidates_stream = load_full_dataset(FULL_FILE)
        total, valid, invalid, honeypots = process_and_save(
            candidates_stream,
            OUTPUT_FULL,
            total_expected=100000
        )
        print_report(total, valid, invalid, honeypots, OUTPUT_FULL)


if __name__ == "__main__":
    main()