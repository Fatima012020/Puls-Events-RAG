import requests
import pandas as pd
from pathlib import Path
import re


API_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/"
    "datasets/evenements-publics-openagenda/records"
)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DIR / "events_raw.csv"

DEPARTMENT = "Loiret"

# Nombre maximum récupéré par requête
PAGE_SIZE = 100


def fetch_events():
    """
    Récupère les événements du Loiret
    dont la date de fin est inférieure à un an d'ancienneté
    ou qui sont à venir.
    """

    all_events = []
    offset = 0

    one_year_ago = (
        pd.Timestamp.now(tz="UTC")
        - pd.DateOffset(years=1)
    ).strftime("%Y-%m-%dT%H:%M")

    print(f"Récupération des événements du {DEPARTMENT}...")
    print(f"Date minimale retenue : {one_year_ago}")

    while True:

        params = {
            "where": (
                f'location_department="{DEPARTMENT}" '
                f"AND lastdate_end >= date'{one_year_ago}'"
            ),
            "limit": PAGE_SIZE,
            "offset": offset,
        }

        response = requests.get(
            API_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()
        events = data.get("results", [])

        if not events:
            break

        all_events.extend(events)

        print(f"{len(all_events)} événements récupérés...")

        if len(events) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

        if offset >= 10000:
            print(
                "Limite de pagination atteinte."
            )
            break

    return all_events

def main():

    events = fetch_events()

    if not events:
        print("Aucun événement trouvé.")
        return

    df = pd.DataFrame(events)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n--- Collecte terminée ---")
    print(f"Nombre total : {len(df)}")
    print(f"Fichier créé : {OUTPUT_FILE}")

    if "location_department" in df.columns:
        print("\nDépartements présents :")
        print(df["location_department"].value_counts())


if __name__ == "__main__":
    main()