"""
Scarica lo storico prezzi (giornaliero) di un evento Polymarket a partire dal suo URL.

Uso (nel Terminale):
    python polymarket.py "https://polymarket.com/event/nome-evento"
    python polymarket.py "https://polymarket.com/it/event/february-inflation-us-annual"

Flusso:
    1. Estrae lo slug dall'URL dell'evento
    2. Interroga la Gamma API per ottenere l'evento e i suoi mercati (outcome)
    3. Per ogni mercato/outcome, scarica lo storico prezzi giornaliero dalla CLOB API
    4. Salva un CSV per ogni outcome nella cartella di output
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
    """Estrae lo slug dell'evento da un URL tipo polymarket.com/event/<slug>[?tid=...]"""
    path = urlparse(url).path.strip("/")
    parts = path.split("/")
    if "event" in parts:
        idx = parts.index("event")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    # fallback: prende l'ultimo pezzo del path
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
    parser = argparse.ArgumentParser(description="Scarica lo storico prezzi giornaliero di un evento Polymarket")
    parser.add_argument("url", help="URL dell'evento Polymarket, es. https://polymarket.com/event/nome-evento")
    parser.add_argument("--outdir", default="polymarket_data",
                         help="Cartella dove salvare i CSV (default: polymarket_data)")
    args = parser.parse_args()

    slug = slug_from_url(args.url)
    print(f"Slug estratto: {slug}")

    event = get_event(slug)
    title = event.get("title", slug)
    markets = event.get("markets", [])

    if not markets:
        print("Nessun mercato trovato per questo evento.", file=sys.stderr)
        sys.exit(1)

    print(f"Evento: {title}")
    print(f"Trovati {len(markets)} mercati/outcome nell'evento.\n")

    outdir = Path(args.outdir) / slug
    outdir.mkdir(parents=True, exist_ok=True)

    for market in markets:
        question = market.get("question", "sconosciuto")
        clob_token_ids_raw = market.get("clobTokenIds")
        outcomes_raw = market.get("outcomes")

        if not clob_token_ids_raw:
            print(f"  [SKIP] '{question}': nessun clobTokenIds disponibile")
            continue

        token_ids = json.loads(clob_token_ids_raw)
        outcomes = json.loads(outcomes_raw) if outcomes_raw else [f"outcome_{i}" for i in range(len(token_ids))]

        print(f"Mercato: {question}")
        for token_id, outcome_name in zip(token_ids, outcomes):
            print(f"  -> scarico storico per outcome '{outcome_name}' (token {token_id[:12]}...)")
            try:
                history = get_price_history(token_id)
            except requests.HTTPError as e:
                print(f"     ERRORE: {e}", file=sys.stderr)
                continue

            if not history:
                print("     Nessun dato restituito (mercato senza scambi o token errato).")
                continue

            safe_question = "".join(c if c.isalnum() or c in " _-" else "_" for c in question)[:60]
            safe_outcome = "".join(c if c.isalnum() or c in " _-" else "_" for c in outcome_name)[:30]
            filename = f"{safe_question}_{safe_outcome}.csv".replace(" ", "_")
            filepath = outdir / filename

            save_csv(filepath, history)
            print(f"     Salvati {len(history)} punti in: {filepath}")

    print(f"\nFatto. Dati salvati in: {outdir.resolve()}")


if __name__ == "__main__":
    main()