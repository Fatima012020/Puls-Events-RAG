import os
import json
from pathlib import Path

import faiss
import numpy as np

from dotenv import load_dotenv
from mistralai.client import Mistral


# =========================
# PARAMÈTRES
# =========================

INDEX_FILE = Path("faiss_index/events.index")
METADATA_FILE = Path("faiss_index/metadata.json")

MODEL_NAME = "mistral-embed"

TOP_K = 5


# =========================
# MISTRAL
# =========================

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError(
        "MISTRAL_API_KEY absente du fichier .env"
    )

client = Mistral(api_key=api_key)


# =========================
# CHARGEMENT DE FAISS
# =========================

def load_faiss():

    print("Chargement de l'index FAISS...")

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    print(
        f"Vecteurs chargés : {index.ntotal}"
    )

    print(
        f"Métadonnées chargées : {len(metadata)}"
    )

    if index.ntotal != len(metadata):
        raise ValueError(
            "Incohérence entre FAISS et les métadonnées."
        )

    return index, metadata


# =========================
# EMBEDDING DE LA QUESTION
# =========================

def embed_query(query):

    response = client.embeddings.create(
        model=MODEL_NAME,
        inputs=[query]
    )

    vector = np.array(
        [response.data[0].embedding],
        dtype="float32"
    )

    # Même normalisation que lors
    # de la construction de l'index
    faiss.normalize_L2(vector)

    return vector


# =========================
# RECHERCHE
# =========================

def search(query, index, metadata):

    query_vector = embed_query(query)

    scores, indices = index.search(
        query_vector,
        TOP_K
    )

    results = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx == -1:
            continue

        item = metadata[idx]

        results.append({
            "rank": rank,
            "score": float(score),
            "metadata": item
        })

    return results


# =========================
# AFFICHAGE
# =========================

def display_results(query, results):

    print("\n" + "=" * 70)

    print(f"QUESTION : {query}")

    print("=" * 70)

    for result in results:

        item = result["metadata"]

        print(
            f"\nRésultat {result['rank']}"
        )

        print(
            f"Score de similarité : "
            f"{result['score']:.4f}"
        )

        print(
            f"Titre : {item.get('title', '')}"
        )

        print(
            f"Ville : {item.get('city', '')}"
        )

        print(
            f"Département : "
            f"{item.get('department', '')}"
        )

        print(
            f"Date de début : "
            f"{item.get('start_date', '')}"
        )

        print(
            f"Date de fin : "
            f"{item.get('end_date', '')}"
        )

        print(
            "Texte : "
            + item.get("text", "")[:400]
            + "..."
        )

        print("-" * 70)


# =========================
# PROGRAMME PRINCIPAL
# =========================

def main():

    index, metadata = load_faiss()

    query = input(
        "\nPosez une question sur les événements : "
    )

    results = search(
        query,
        index,
        metadata
    )

    display_results(
        query,
        results
    )


if __name__ == "__main__":
    main()