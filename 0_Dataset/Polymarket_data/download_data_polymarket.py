"""
Download the historical price data (daily) for a Polymarket event from its URL.

Usage (in the Terminal):
    python polymarket.py "https://polymarket.com/event/event-name"
    python polymarket.py "https://polymarket.com/it/event/february-inflation-us-annual"

Flow:
    1. Extract the event slug from the URL
    2. Query the Gamma API to get the event and its markets (outcomes)
    3. For each market/outcome, download the daily price history from the CLOB API
    4. Save one CSV per outcome in the output folder
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

FIDELITY = 1440  # minuti = 1 punto al giorno


def slug_from_url(url: str) -> str:
    """Extracts the event slug from a URL like polymarket.com/event/<slug>[?tid=...]"""
    path = urlparse(url).path.strip("/")
    parts = path.split("/")
    if "event" in parts:
        idx = parts.index("event")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    # fallback: takes the last path segment
    return parts[-1]


def get_event(slug: str) -> dict:
    resp = requests.get(f"{GAMMA_API}/events/slug/{slug}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_price_history(token_id: str) -> list[dict]:
    params = {"market": token_id, "interval": "max", "fidelity": FIDELITY}
    resp = requests.get(f"{CLOB_API}/prices-history", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("history", [])


def save_csv(path: Path, history: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_unix", "datetime_utc", "price"])
        for point in history:
            t = point["t"]
            p = point["p"]
            dt = datetime.fromtimestamp(t, tz=timezone.utc).isoformat()
            writer.writerow([t, dt, p])


def main():
    parser = argparse.ArgumentParser(description="Download the daily price history of a Polymarket event")
    parser.add_argument("url", help="Polymarket event URL, e.g. https://polymarket.com/event/event-name")
    parser.add_argument("--outdir", default="polymarket_data",
                         help="Folder where the CSV files will be saved (default: polymarket_data)")
    args = parser.parse_args()

    slug = slug_from_url(args.url)
    print(f"Extracted slug: {slug}")

    event = get_event(slug)
    title = event.get("title", slug)
    markets = event.get("markets", [])

    if not markets:
        print("No markets found for this event.", file=sys.stderr)
        sys.exit(1)

    print(f"Event: {title}")
    print(f"Found {len(markets)} markets/outcomes in the event.\n")

    outdir = Path(args.outdir) / slug
    outdir.mkdir(parents=True, exist_ok=True)

    for market in markets:
        question = market.get("question", "unknown")
        clob_token_ids_raw = market.get("clobTokenIds")
        outcomes_raw = market.get("outcomes")

        if not clob_token_ids_raw:
            print(f"  [SKIP] '{question}': no clobTokenIds available")
            continue

        token_ids = json.loads(clob_token_ids_raw)
        outcomes = json.loads(outcomes_raw) if outcomes_raw else [f"outcome_{i}" for i in range(len(token_ids))]

        print(f"Market: {question}")
        for token_id, outcome_name in zip(token_ids, outcomes):
            print(f"  -> downloading history for outcome '{outcome_name}' (token {token_id[:12]}...)")
            try:
                history = get_price_history(token_id)
            except requests.HTTPError as e:
                print(f"     ERROR: {e}", file=sys.stderr)
                continue

            if not history:
                print("     No data returned (market without trades or wrong token).")
                continue

            safe_question = "".join(c if c.isalnum() or c in " _-" else "_" for c in question)[:60]
            safe_outcome = "".join(c if c.isalnum() or c in " _-" else "_" for c in outcome_name)[:30]
            filename = f"{safe_question}_{safe_outcome}.csv".replace(" ", "_")
            filepath = outdir / filename

            save_csv(filepath, history)
            print(f"     Saved {len(history)} points in: {filepath}")

    print(f"\nDone. Data saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()