"""
Create a Prolific study linked to a Qualtrics survey.

Workflow:
  1. Create the Qualtrics survey (via qualtrics.py)
  2. Create a draft Prolific study pointing to the survey URL
  3. Retrieve the study's unique completion code
  4. Update the Qualtrics survey redirect URL with the real completion code

Usage:
  python prolific_publish.py <jsonl_file> [--total_available_places N] [--estimated_completion_time M] [--prolific-reward P]

Requires PROLIFIC_API_TOKEN in qualtrics_config.json (or env var PROLIFIC_API_TOKEN).
"""
import argparse
import json
import re
import secrets
import sys
import requests

from qualtrics import create_survey_and_add_questions, activate_survey


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path="qualtrics_config.json"):
    with open(path) as f:
        return json.load(f)


def qualtrics_survey_url(config, survey_id):
    """Build the participant-facing Qualtrics URL with Prolific URL parameters."""
    # BASE_URL is e.g. https://samgu.eu.qualtrics.com/API/v3
    base = re.sub(r"/API/v\d+$", "", config["BASE_URL"])
    return (
        f"{base}/jfe/form/{survey_id}"
        "?PROLIFIC_PID={{%PROLIFIC_PID%}}"
        "&STUDY_ID={{%STUDY_ID%}}"
        "&SESSION_ID={{%SESSION_ID%}}"
    )


def prolific_headers(config):
    import os
    token = os.environ.get("PROLIFIC_API_TOKEN") or config.get("PROLIFIC_API_TOKEN", "")
    if not token:
        sys.exit("Error: set PROLIFIC_API_TOKEN in qualtrics_config.json or as an env var.")
    return {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Prolific API calls
# ---------------------------------------------------------------------------

PROLIFIC_API = "https://api.prolific.com/api/v1"


def publish_prolific_study(study_id, headers):
    """Transition a Prolific study from UNPUBLISHED to ACTIVE."""
    resp = requests.post(
        f"{PROLIFIC_API}/studies/{study_id}/transition/",
        headers=headers,
        json={"action": "PUBLISH"},
    )
    if not resp.ok:
        print(f"Warning: failed to publish Prolific study ({resp.status_code}): {resp.text}")
    else:
        print(f"       Prolific study published (ACTIVE).")
    return resp


def create_prolific_study(study_name, survey_url, config, headers,
                          total_available_places=None,
                          estimated_completion_time=None,
                          prolific_reward=None):
    """Create a draft study on Prolific and return the full study JSON."""
    # Hardcoded filter IDs and choice values from the Prolific /api/v1/filters/ endpoint.
    #   first-language        → '18' = English
    #   country-of-birth      → '0'  = United Kingdom
    #   current-country-of-residence → '0' = United Kingdom
    filters_payload = [
        {"filter_id": "first-language",               "selected_values": ["18"]},  # English
        {"filter_id": "country-of-birth",             "selected_values": ["0"]},   # United Kingdom
        {"filter_id": "current-country-of-residence", "selected_values": ["0"]},   # United Kingdom
    ]

    payload = {
        "name": config.get("PROLIFIC_STUDY_NAME", "Definition Similarity Annotation"),
        "internal_name": study_name,
        "description": config.get("PROLIFIC_STUDY_DESCRIPTION",
                                  "Please complete this Qualtrics survey rating definition similarity."),
        "external_study_url": survey_url,
        "prolific_id_option": "url_parameters",          # auto-capture IDs via URL
        "completion_option": "url",                        # redirect back to Prolific
        "total_available_places": total_available_places,
        "estimated_completion_time": estimated_completion_time,
        "reward": prolific_reward,      # reward in pence/cents
        "filters": filters_payload,
        "completion_codes": [
            {"code": secrets.token_hex(12).upper(), "code_type": "COMPLETED",
             "actions": [{"action": "MANUALLY_REVIEW"}]}
        ],
    }

    resp = requests.post(f"{PROLIFIC_API}/studies/", headers=headers, json=payload)
    if not resp.ok:
        print(f"Prolific API error {resp.status_code}: {resp.text}")
    resp.raise_for_status()
    return resp.json()


def get_completion_code(study):
    """Extract the completion code from a Prolific study object."""
    # Check the completion_codes list (current API)
    for item in study.get("completion_codes", []):
        if item.get("code_type") == "COMPLETED":
            return item["code"]
    # Fallback: legacy top-level field
    if study.get("completion_code"):
        return study["completion_code"]
    return None


def update_qualtrics_redirect(survey_id, completion_code, config):
    """Patch the Qualtrics survey end-of-survey redirect URL with the real code."""
    url = f"{config['BASE_URL']}/survey-definitions/{survey_id}/options"
    headers = {
        "x-api-token": config["API_TOKEN"],
        "content-type": "application/json",
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    options = resp.json()["result"]

    redirect_url = f"https://app.prolific.com/submissions/complete?cc={completion_code}"
    options["SurveyTermination"] = "Redirect"
    options["EOSRedirectURL"] = redirect_url

    resp = requests.put(url, headers=headers, json=options)
    resp.raise_for_status()
    return redirect_url


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Create Qualtrics survey + Prolific study.")
    parser.add_argument("jsonl_file", help="Path to the .jsonl file with definition pairs.")
    parser.add_argument("--total_available_places", type=int, required=True,
                        help="Number of Prolific participants.")
    parser.add_argument("--estimated_completion_time", type=int, required=True,
                        help="Estimated minutes to complete.")
    parser.add_argument("--language", default="EN", help="Survey language (default: EN).")
    parser.add_argument("--publish-qualtrics", dest="publish_qualtrics",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Activate (publish) the Qualtrics survey once uploaded (default: True).")
    parser.add_argument("--publish-prolific", dest="publish_prolific",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Publish the Prolific study (go live, not draft) once done (default: False).")
    parser.add_argument("--prolific-reward", type=int, required=True,
                        help="Reward per participant in pence/cents.")
    args = parser.parse_args()

    config = load_config()

    # --- Qualtrics headers ---
    q_headers = {
        "x-api-token": config["API_TOKEN"],
        "content-type": "application/json",
    }

    # --- 1. Create Qualtrics survey ---
    dataset_name = f'definition_similarity_{args.jsonl_file.split("/")[-1].replace(".jsonl", "")}'
    print(f"[1/4] Creating Qualtrics survey: {dataset_name}")
    survey_id = create_survey_and_add_questions(
        args.jsonl_file, dataset_name, args.language, config, q_headers
    )
    print(f"       Survey ID: {survey_id}")

    # --- 2. Create Prolific study ---
    survey_url = qualtrics_survey_url(config, survey_id)
    p_headers = prolific_headers(config)

    print(f"[2/4] Creating Prolific study...")
    study = create_prolific_study(
        study_name=dataset_name,
        survey_url=survey_url,
        config=config,
        headers=p_headers,
        total_available_places=args.total_available_places,
        estimated_completion_time=args.estimated_completion_time,
        prolific_reward=args.prolific_reward
    )
    study_id = study["id"]
    print(f"       Prolific study ID: {study_id}")

    # --- 3. Get completion code ---
    print(f"[3/4] Retrieving completion code...")
    completion_code = get_completion_code(study)
    if not completion_code:
        print("       Warning: could not extract completion code from study response.")
        print("       You may need to set it manually in Qualtrics.")
        print(f"       Full study response: {json.dumps(study, indent=2)}")
    else:
        print(f"       Completion code: {completion_code}")

    # --- 4. Patch Qualtrics redirect URL ---
    if completion_code:
        print(f"[4/4] Updating Qualtrics redirect URL...")
        redirect_url = update_qualtrics_redirect(survey_id, completion_code, config)
        print(f"       Redirect URL: {redirect_url}")
    else:
        print("[4/4] Skipped (no completion code).")

    # --- Activate Qualtrics survey if requested (after redirect URL is set) ---
    if args.publish_qualtrics:
        print(f"[5/5] Activating Qualtrics survey...")
        activate_survey(survey_id, config, q_headers)

    # --- Publish Prolific study if requested ---
    if args.publish_prolific:
        print(f"[6/6] Publishing Prolific study (going live)...")
        publish_prolific_study(study_id, p_headers)

    # --- Summary ---
    prolific_status = "ACTIVE (published)" if args.publish_prolific else "UNPUBLISHED (draft)"
    qualtrics_status = "Active" if args.publish_qualtrics else "Inactive (not published)"
    print("\n=== Done ===")
    print(f"  Qualtrics survey ID : {survey_id} [{qualtrics_status}]")
    print(f"  Prolific study ID   : {study_id} [{prolific_status}]")
    print(f"  Completion code     : {completion_code or 'N/A'}")
    print(f"  Survey URL (Prolific→Qualtrics): {survey_url}")


if __name__ == "__main__":
    main()
