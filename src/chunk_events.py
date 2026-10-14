"""
Découpage des événements en chunks pour le système RAG Puls-Events.

Ce module charge les textes pré-traités, les découpe avec
RecursiveCharacterTextSplitter et conserve les métadonnées nécessaires
à la recherche vectorielle.
"""

import pandas as pd
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


INPUT_FILE = Path("data/processed/events_processed.csv")

OUTPUT_DIR = Path("data/chunks")
OUTPUT_FILE = OUTPUT_DIR / "events_chunks.csv"


CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_events():
    """
    Charge les événements pré-traités contenant le champ ``rag_text``.

    Les lignes dont ``rag_text`` est absent sont supprimées.

    Returns:
        pandas.DataFrame: Événements prêts pour le découpage.

    Raises:
        ValueError: Si la colonne ``rag_text`` est absente.
    """
    df = pd.read_csv(INPUT_FILE)

    if "rag_text" not in df.columns:
        raise ValueError(
            "La colonne rag_text est absente."
        )

    df = df.dropna(subset=["rag_text"]).copy()

    print(f"Événements chargés : {len(df)}")

    return df


def create_chunks(df):
    """
    Découpe le texte de chaque événement en chunks avec chevauchement.

    Le découpage utilise RecursiveCharacterTextSplitter avec une taille
    maximale de 800 caractères et un chevauchement de 100 caractères.
    Chaque chunk conserve les principales métadonnées de son événement.

    Args:
        df (pandas.DataFrame): Événements contenant la colonne ``rag_text``.

    Returns:
        pandas.DataFrame: Chunks accompagnés de leurs métadonnées.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for _, row in df.iterrows():

        texts = splitter.split_text(
            str(row["rag_text"])
        )

        for chunk_number, chunk_text in enumerate(texts):

            chunks.append({
                "uid": row.get("uid", ""),
                "chunk_id": (
                    f"{row.get('uid', '')}_{chunk_number}"
                ),
                "chunk_number": chunk_number,
                "title": row.get("title_fr", ""),
                "city": row.get("location_city", ""),
                "department": row.get(
                    "location_department",
                    ""
                ),
                "start_date": row.get(
                    "firstdate_begin",
                    ""
                ),
                "end_date": row.get(
                    "lastdate_end",
                    ""
                ),
                "url": row.get(
                    "canonicalurl",
                    ""
                ),
                "chunk_text": chunk_text
            })

    chunks_df = pd.DataFrame(chunks)

    return chunks_df


def save_chunks(df):
    """
    Sauvegarde les chunks et leurs métadonnées dans un fichier CSV.

    Args:
        df (pandas.DataFrame): Chunks à sauvegarder.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n--- Chunking terminé ---")
    print(f"Nombre de chunks : {len(df)}")
    print(f"Fichier créé : {OUTPUT_FILE}")


def main():
    """
    Exécute le pipeline de chargement, découpage et sauvegarde des chunks.
    """

    events_df = load_events()

    chunks_df = create_chunks(events_df)

    save_chunks(chunks_df)

    print("\nExemple de chunk :")

    print(
        chunks_df.iloc[0]["chunk_text"]
    )


if __name__ == "__main__":
    main()