"""
Construction et sauvegarde de l'index vectoriel FAISS de Puls-Events.

Ce module charge les embeddings des chunks, les convertit en matrice
NumPy, applique une normalisation L2, construit un index exact
``IndexFlatIP`` et sauvegarde l'index ainsi que les métadonnées associées.
"""

import json
from pathlib import Path

import faiss
import numpy as np


# =========================
# PARAMÈTRES
# =========================

INPUT_FILE = Path(
    "data/vectorized/chunks_embeddings.json"
)

OUTPUT_DIR = Path("faiss_index")

INDEX_FILE = OUTPUT_DIR / "events.index"
METADATA_FILE = OUTPUT_DIR / "metadata.json"


# =========================
# CHARGEMENT
# =========================

def load_data():
    """
    Charge les embeddings et leurs métadonnées depuis le fichier JSON.

    Returns:
        list[dict]: Chunks vectorisés et métadonnées associées.

    Raises:
        ValueError: Si le fichier d'embeddings est vide.
    """

    print("Chargement des embeddings...")

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    if not data:
        raise ValueError(
            "Le fichier d'embeddings est vide."
        )

    print(f"Nombre de chunks chargés : {len(data)}")

    return data


# =========================
# PRÉPARATION DES VECTEURS
# =========================

def prepare_vectors(data):
    """
    Convertit les embeddings en matrice NumPy et les normalise.

    La normalisation L2 permet d'utiliser le produit scalaire
    de ``IndexFlatIP`` comme mesure de similarité cosinus.

    Args:
        data (list[dict]): Chunks contenant leurs embeddings.

    Returns:
        numpy.ndarray: Matrice float32 normalisée compatible avec FAISS.
    """

    vectors = np.array(
        [
            item["embedding"]
            for item in data
        ],
        dtype="float32"
    )

    print(f"Forme de la matrice : {vectors.shape}")

    # Normalisation pour utiliser
    # le produit scalaire comme similarité cosinus
    faiss.normalize_L2(vectors)

    return vectors


# =========================
# CRÉATION DE L'INDEX
# =========================

def build_index(vectors):
    """
    Construit un index FAISS exact avec ``IndexFlatIP``.

    Args:
        vectors (numpy.ndarray): Matrice des embeddings normalisés.

    Returns:
        faiss.IndexFlatIP: Index contenant les vecteurs.
    """

    dimension = vectors.shape[1]

    print(f"Dimension des vecteurs : {dimension}")

    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    print(
        f"Nombre de vecteurs dans FAISS : "
        f"{index.ntotal}"
    )

    return index


# =========================
# MÉTADONNÉES
# =========================

def prepare_metadata(data):
    """
    Extrait les métadonnées associées à chaque vecteur FAISS.

    L'ordre des métadonnées est conservé afin que la position
    d'un résultat FAISS corresponde au bon événement.

    Args:
        data (list[dict]): Chunks vectorisés et métadonnées associées.

    Returns:
        list[dict]: Métadonnées alignées avec les vecteurs de l'index.
    """

    metadata = []

    for item in data:

        metadata.append({
            "uid": item.get("uid", ""),
            "chunk_id": item.get("chunk_id", ""),
            "chunk_number": item.get(
                "chunk_number",
                0
            ),
            "title": item.get("title", ""),
            "city": item.get("city", ""),
            "department": item.get(
                "department",
                ""
            ),
            "start_date": item.get(
                "start_date",
                ""
            ),
            "end_date": item.get(
                "end_date",
                ""
            ),
            "url": item.get("url", ""),
            "text": item.get("text", "")
        })

    return metadata


# =========================
# SAUVEGARDE
# =========================

def save_index(index, metadata):
    """
    Sauvegarde l'index FAISS et les métadonnées sur disque.

    Args:
        index: Index FAISS à sauvegarder.
        metadata (list[dict]): Métadonnées associées aux vecteurs.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\n--- Base FAISS créée ---")

    print(f"Index : {INDEX_FILE}")
    print(f"Métadonnées : {METADATA_FILE}")


# =========================
# CONTRÔLES
# =========================

def validate(index, metadata):
    """
    Vérifie la cohérence entre l'index FAISS et les métadonnées.

    Args:
        index: Index FAISS à contrôler.
        metadata (list[dict]): Métadonnées associées aux vecteurs.

    Raises:
        ValueError: Si l'index est vide ou si le nombre de vecteurs
            ne correspond pas au nombre de métadonnées.
    """

    print("\n--- Vérifications ---")

    print(
        f"Vecteurs FAISS : {index.ntotal}"
    )

    print(
        f"Métadonnées : {len(metadata)}"
    )

    if index.ntotal != len(metadata):
        raise ValueError(
            "Le nombre de vecteurs et de "
            "métadonnées ne correspond pas."
        )

    if index.ntotal == 0:
        raise ValueError(
            "L'index FAISS est vide."
        )

    print("Correspondance vecteurs/métadonnées : OK")


# =========================
# PROGRAMME PRINCIPAL
# =========================

def main():
    """
    Exécute la construction, la validation et la sauvegarde de l'index FAISS.
    """

    data = load_data()

    vectors = prepare_vectors(data)

    index = build_index(vectors)

    metadata = prepare_metadata(data)

    validate(index, metadata)

    save_index(index, metadata)

    print("\nConstruction FAISS réussie !")


if __name__ == "__main__":
    main()