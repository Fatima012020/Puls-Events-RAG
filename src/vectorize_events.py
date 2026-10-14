import os
import time
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from mistralai.client import Mistral


# =========================
# PARAMÈTRES
# =========================

INPUT_FILE = Path("data/processed/events_processed.csv")

OUTPUT_DIR = Path("data/vectorized")
OUTPUT_FILE = OUTPUT_DIR / "events_embeddings.json"

BATCH_SIZE = 50

MODEL_NAME = "mistral-embed"


# =========================
# CHARGEMENT API
# =========================

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError(
        "La variable MISTRAL_API_KEY est absente du fichier .env"
    )

client = Mistral(api_key=api_key)


# =========================
# CHARGEMENT DES DONNÉES
# =========================

def load_events():
    """
    Charge les événements préparés pour le RAG.
    """

    df = pd.read_csv(INPUT_FILE)

    if "rag_text" not in df.columns:
        raise ValueError(
            "La colonne 'rag_text' est absente du fichier traité."
        )

    df = df.dropna(subset=["rag_text"]).copy()

    df["rag_text"] = (
        df["rag_text"]
        .astype(str)
        .str.strip()
    )

    df = df[df["rag_text"] != ""]

    print(f"Nombre d'événements à vectoriser : {len(df)}")

    return df


# =========================
# GÉNÉRATION DES EMBEDDINGS
# =========================

def create_embeddings(texts):
    """
    Envoie un lot de textes à Mistral
    et retourne les vecteurs associés.
    """

    response = client.embeddings.create(
        model=MODEL_NAME,
        inputs=texts
    )

    embeddings = [
        item.embedding
        for item in response.data
    ]

    return embeddings


# =========================
# VECTORISATION PAR LOTS
# =========================

def vectorize_events(df):
    """
    Vectorise tous les événements par lots.
    """

    results = []

    total = len(df)

    for start in range(0, total, BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch = df.iloc[start:end]

        texts = batch["rag_text"].tolist()

        print(
            f"Vectorisation : "
            f"{start + 1} à {end} / {total}"
        )

        try:

            embeddings = create_embeddings(texts)

        except Exception as error:

            print(
                f"Erreur pendant le lot "
                f"{start + 1}-{end} : {error}"
            )

            raise

        for (_, row), embedding in zip(
            batch.iterrows(),
            embeddings
        ):

            results.append(
                {
                    "uid": str(row.get("uid", "")),
                    "title": str(
                        row.get("title_fr", "")
                    ),
                    "city": str(
                        row.get("location_city", "")
                    ),
                    "department": str(
                        row.get(
                            "location_department",
                            ""
                        )
                    ),
                    "start_date": str(
                        row.get(
                            "firstdate_begin",
                            ""
                        )
                    ),
                    "end_date": str(
                        row.get(
                            "lastdate_end",
                            ""
                        )
                    ),
                    "url": str(
                        row.get(
                            "canonicalurl",
                            ""
                        )
                    ),
                    "text": row["rag_text"],
                    "embedding": embedding
                }
            )

        # Petite pause pour éviter d'enchaîner
        # trop rapidement les appels API
        time.sleep(0.5)

    return results


# =========================
# SAUVEGARDE
# =========================

def save_embeddings(results):
    """
    Sauvegarde les événements et leurs vecteurs.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False
        )

    print("\n--- Vectorisation terminée ---")
    print(
        f"Nombre de vecteurs créés : "
        f"{len(results)}"
    )
    print(
        f"Fichier créé : "
        f"{OUTPUT_FILE}"
    )


# =========================
# PROGRAMME PRINCIPAL
# =========================

def main():

    df = load_events()

    results = vectorize_events(df)

    save_embeddings(results)

    if results:

        first_embedding = results[0]["embedding"]

        print(
            f"Dimension d'un vecteur : "
            f"{len(first_embedding)}"
        )


if __name__ == "__main__":
    main()