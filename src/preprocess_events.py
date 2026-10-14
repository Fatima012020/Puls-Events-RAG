"""
Pré-processing des événements OpenAgenda pour le système RAG Puls-Events.

Ce module charge les données brutes, filtre les événements du Loiret,
applique le périmètre temporel du projet, nettoie les champs textuels,
supprime les doublons et construit le champ ``rag_text`` destiné
à la vectorisation.
"""

import re
import pandas as pd
from pathlib import Path


# =========================
# PARAMÈTRES
# =========================

RAW_FILE = Path("data/raw/events_raw.csv")

PROCESSED_DIR = Path("data/processed")
PROCESSED_FILE = PROCESSED_DIR / "events_processed.csv"

DEPARTMENT = "Loiret"


# =========================
# CHARGEMENT DES DONNÉES
# =========================

def load_data():
    """
    Charge les événements bruts depuis le fichier CSV OpenAgenda.

    Returns:
        pandas.DataFrame: Données brutes des événements.
    """

    df = pd.read_csv(RAW_FILE)

    print(f"Événements bruts : {len(df)}")

    return df


# =========================
# FILTRE GÉOGRAPHIQUE
# =========================

def filter_location(df):
    """
    Conserve uniquement les événements situés dans le Loiret.

    Args:
        df (pandas.DataFrame): Données des événements à filtrer.

    Returns:
        pandas.DataFrame: Événements appartenant au département du Loiret.
    """

    df = df.copy()

    df["location_department"] = (
        df["location_department"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["location_department"].str.lower()
        == DEPARTMENT.lower()
    ].copy()

    print(f"Événements dans le Loiret : {len(df)}")

    return df


# =========================
# FILTRE TEMPOREL
# =========================

def filter_dates(df):
    """
    Conserve :
    - les événements terminés depuis moins d'un an
    - les événements actuels
    - les événements à venir
    """"""
    Conserve les événements terminés depuis moins d'un an,
    les événements en cours et les événements à venir.

    La date de fin est utilisée comme référence lorsqu'elle existe ;
    sinon, la date de début est utilisée.

    Args:
        df (pandas.DataFrame): Données des événements.

    Returns:
        pandas.DataFrame: Événements respectant le périmètre temporel.
    """

    df = df.copy()

    df["firstdate_begin"] = pd.to_datetime(
        df["firstdate_begin"],
        errors="coerce",
        utc=True
    )

    df["lastdate_end"] = pd.to_datetime(
        df["lastdate_end"],
        errors="coerce",
        utc=True
    )

    today = pd.Timestamp.now(tz="UTC")

    one_year_ago = today - pd.DateOffset(years=1)

    # On utilise la date de fin comme référence.
    # Si elle est manquante, on prend la date de début.
    reference_date = df["lastdate_end"].fillna(
        df["firstdate_begin"]
    )

    df = df[
        reference_date.notna()
        & (reference_date >= one_year_ago)
    ].copy()

    print(f"Événements récents / à venir : {len(df)}")

    return df


# =========================
# NETTOYAGE HTML
# =========================

def clean_html(text):
    """
    Nettoie un champ textuel en supprimant les balises HTML
    et les espaces superflus.

    Args:
        text: Valeur textuelle à nettoyer.

    Returns:
        str: Texte nettoyé ou chaîne vide si la valeur est absente.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Suppression des balises HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # Remplacement des espaces multiples
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================
# NETTOYAGE DES DONNÉES
# =========================

def clean_data(df):
    """
    Sélectionne les colonnes utiles et nettoie les événements.

    La fonction supprime les lignes sans titre, élimine les doublons
    et applique le nettoyage HTML aux principales colonnes textuelles.

    Args:
        df (pandas.DataFrame): Données à nettoyer.

    Returns:
        pandas.DataFrame: Données nettoyées.
    """

    df = df.copy()

    columns = [
        "uid",
        "title_fr",
        "description_fr",
        "longdescription_fr",
        "keywords_fr",
        "firstdate_begin",
        "lastdate_end",
        "location_name",
        "location_address",
        "location_postalcode",
        "location_city",
        "location_department",
        "location_region",
        "category",
        "canonicalurl",
    ]

    # Conserver uniquement les colonnes existantes
    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    df = df[available_columns].copy()

    # Le titre est indispensable
    if "title_fr" in df.columns:
        df = df.dropna(subset=["title_fr"])

    # Suppression des doublons
    if "uid" in df.columns:
        df = df.drop_duplicates(subset=["uid"])
    else:
        df = df.drop_duplicates()

    text_columns = [
        "title_fr",
        "description_fr",
        "longdescription_fr",
        "keywords_fr",
        "location_name",
        "location_address",
        "location_city",
        "location_department",
        "location_region",
        "category",
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].apply(clean_html)

    return df


# =========================
# CRÉATION DU TEXTE POUR LE RAG
# =========================

def create_rag_text(df):
    """
    Construit le texte documentaire utilisé par le système RAG.

    Les informations utiles d'un événement sont regroupées dans
    un champ structuré ``rag_text`` qui sera ensuite découpé
    puis vectorisé.

    Args:
        df (pandas.DataFrame): Événements nettoyés.

    Returns:
        pandas.DataFrame: DataFrame enrichi de la colonne ``rag_text``.
    """

    df = df.copy()

    def value(row, column):
        """
        Récupère une valeur de manière sécurisée.

        Args:
            row (pandas.Series): Ligne représentant un événement.
            column (str): Nom de la colonne à récupérer.

        Returns:
            str: Valeur convertie en texte ou chaîne vide si elle est absente.
        """

        if column not in row.index:
            return ""

        val = row[column]

        if pd.isna(val):
            return ""

        return str(val)

    def build_text(row):
        """
        Assemble les métadonnées d'un événement sous forme de texte structuré.

        Args:
            row (pandas.Series): Ligne représentant un événement.

        Returns:
            str: Texte complet destiné au champ ``rag_text``.
        """

        return (
            f"Titre : {value(row, 'title_fr')}\n"
            f"Description : {value(row, 'description_fr')}\n"
            f"Description détaillée : "
            f"{value(row, 'longdescription_fr')}\n"
            f"Mots-clés : {value(row, 'keywords_fr')}\n"
            f"Lieu : {value(row, 'location_name')}\n"
            f"Adresse : {value(row, 'location_address')}\n"
            f"Code postal : "
            f"{value(row, 'location_postalcode')}\n"
            f"Ville : {value(row, 'location_city')}\n"
            f"Département : "
            f"{value(row, 'location_department')}\n"
            f"Région : {value(row, 'location_region')}\n"
            f"Date de début : "
            f"{value(row, 'firstdate_begin')}\n"
            f"Date de fin : "
            f"{value(row, 'lastdate_end')}\n"
            f"Catégorie : {value(row, 'category')}"
        )

    df["rag_text"] = df.apply(
        build_text,
        axis=1
    )

    return df


# =========================
# SAUVEGARDE
# =========================

def save_data(df):
    """
    Sauvegarde les événements pré-traités dans un fichier CSV.

    Args:
        df (pandas.DataFrame): Données finales à sauvegarder.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n--- Résultat du pré-processing ---")
    print(f"Nombre final d'événements : {len(df)}")
    print(f"Fichier créé : {PROCESSED_FILE}")


# =========================
# PROGRAMME PRINCIPAL
# =========================

def main():
    """
    Exécute l'ensemble du pipeline de pré-processing des événements.
    """

    df = load_data()

    df = filter_location(df)

    df = filter_dates(df)

    df = clean_data(df)

    df = create_rag_text(df)

    save_data(df)

    if not df.empty:

        print("\nExemple :")

        print(df.iloc[0]["rag_text"])


if __name__ == "__main__":
    main()