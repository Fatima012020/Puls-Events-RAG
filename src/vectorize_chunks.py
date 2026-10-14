"""
Vectorisation des chunks d'événements avec l'API Mistral.

Ce module charge les chunks préparés, génère leurs embeddings
avec le modèle ``mistral-embed`` par lots, associe chaque vecteur
aux métadonnées de son événement et sauvegarde le résultat en JSON.
"""

import os
import json
import time
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from mistralai.client import Mistral


INPUT_FILE = Path(
    "data/chunks/events_chunks.csv"
)

OUTPUT_DIR = Path(
    "data/vectorized"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "chunks_embeddings.json"
)

MODEL_NAME = "mistral-embed"

BATCH_SIZE = 50


load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError(
        "MISTRAL_API_KEY absente du fichier .env"
    )


client = Mistral(
    api_key=api_key
)


def load_chunks():
    """
    Charge et nettoie les chunks à vectoriser.

    Les chunks sans texte ou contenant uniquement des espaces
    sont supprimés.

    Returns:
        pandas.DataFrame: Chunks valides prêts pour la vectorisation.
    """

    df = pd.read_csv(INPUT_FILE)

    df = df.dropna(
        subset=["chunk_text"]
    ).copy()

    df["chunk_text"] = (
        df["chunk_text"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["chunk_text"] != ""
    ]

    print(
        f"Nombre de chunks à vectoriser : {len(df)}"
    )

    return df


def embed_texts(texts):
    """
    Génère les embeddings Mistral d'une liste de textes.

    Args:
        texts (list[str]): Textes à vectoriser.

    Returns:
        list[list[float]]: Vecteurs d'embedding retournés par Mistral.
    """

    response = client.embeddings.create(
        model=MODEL_NAME,
        inputs=texts
    )

    return [
        item.embedding
        for item in response.data
    ]


def vectorize(df):
    """
    Vectorise l'ensemble des chunks par lots.

    Les chunks sont envoyés à l'API Mistral par groupes définis
    par ``BATCH_SIZE``. Chaque embedding est ensuite associé
    aux métadonnées du chunk correspondant.

    Args:
        df (pandas.DataFrame): Chunks à vectoriser.

    Returns:
        list[dict]: Métadonnées des chunks accompagnées de leurs embeddings.
    """

    results = []

    total = len(df)

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch = df.iloc[
            start:end
        ]

        texts = (
            batch["chunk_text"]
            .tolist()
        )

        print(
            f"Vectorisation : "
            f"{start + 1} à {end} / {total}"
        )

        embeddings = embed_texts(
            texts
        )

        for (_, row), embedding in zip(
            batch.iterrows(),
            embeddings
        ):

            results.append({
                "uid": str(
                    row.get("uid", "")
                ),
                "chunk_id": str(
                    row.get("chunk_id", "")
                ),
                "chunk_number": int(
                    row.get(
                        "chunk_number",
                        0
                    )
                ),
                "title": str(
                    row.get("title", "")
                ),
                "city": str(
                    row.get("city", "")
                ),
                "department": str(
                    row.get(
                        "department",
                        ""
                    )
                ),
                "start_date": str(
                    row.get(
                        "start_date",
                        ""
                    )
                ),
                "end_date": str(
                    row.get(
                        "end_date",
                        ""
                    )
                ),
                "url": str(
                    row.get("url", "")
                ),
                "text": str(
                    row["chunk_text"]
                ),
                "embedding": embedding
            })

        time.sleep(0.5)

    return results


def save_results(results):
    """
    Sauvegarde les embeddings et leurs métadonnées au format JSON.

    Args:
        results (list[dict]): Chunks vectorisés avec leurs métadonnées.
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

    print(
        "\n--- Vectorisation des chunks terminée ---"
    )

    print(
        f"Nombre de vecteurs : {len(results)}"
    )

    print(
        f"Dimension : "
        f"{len(results[0]['embedding'])}"
    )

    print(
        f"Fichier créé : {OUTPUT_FILE}"
    )


def main():
    """
    Exécute le pipeline complet de vectorisation des chunks.
    """

    df = load_chunks()

    results = vectorize(df)

    save_results(results)


if __name__ == "__main__":
    main()