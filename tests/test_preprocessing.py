"""
Tests de qualité des données prétraitées du projet Puls-Events.

Ces tests contrôlent le contenu du fichier events_processed.csv.
"""

import unittest
from pathlib import Path

import pandas as pd


PROCESSED_FILE = Path("data/processed/events_processed.csv")
EXPECTED_DEPARTMENT = "Loiret"

# Date provisoire du POC, à confirmer avec le script de collecte.
POC_REFERENCE_DATE = pd.Timestamp("2026-09-01", tz="UTC")


class TestPreprocessing(unittest.TestCase):
    """Contrôle la qualité des données après prétraitement."""

    @classmethod
    def setUpClass(cls):
        """Charge une seule fois les données utilisées par les tests."""
        cls.df = pd.read_csv(PROCESSED_FILE)

    def test_dataset_not_empty(self):
        """Vérifie que le dataset contient des événements."""
        self.assertFalse(
            self.df.empty,
            "Le dataset traité est vide."
        )

    def test_department_is_loiret(self):
        """Vérifie que tous les événements appartiennent au Loiret."""
        departments = (
            self.df["location_department"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )

        self.assertTrue(
            (departments == EXPECTED_DEPARTMENT.lower()).all(),
            "Des événements hors du Loiret sont présents."
        )

    def test_events_are_recent_or_upcoming(self):
        """
        Vérifie la fraîcheur des événements par rapport
        à la date de référence du POC.
        """
        first_dates = pd.to_datetime(
            self.df["firstdate_begin"],
            errors="coerce",
            utc=True
        )

        end_dates = pd.to_datetime(
            self.df["lastdate_end"],
            errors="coerce",
            utc=True
        )

        reference_dates = end_dates.fillna(first_dates)

        one_year_ago = (
            POC_REFERENCE_DATE - pd.DateOffset(years=1)
        )

        self.assertTrue(
            (reference_dates >= one_year_ago).all(),
            "Certains événements sont trop anciens "
            "pour la période de référence du POC."
        )

    def test_titles_not_empty(self):
        """Vérifie que tous les événements possèdent un titre."""
        titles = (
            self.df["title_fr"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        self.assertTrue(
            (titles != "").all(),
            "Certains événements n'ont pas de titre."
        )

    def test_uid_unique(self):
        """Vérifie l'absence de doublons sur les UID OpenAgenda."""
        self.assertTrue(
            self.df["uid"].is_unique,
            "Des UID dupliqués sont présents."
        )

    def test_rag_text_not_empty(self):
        """Vérifie que chaque événement possède un texte RAG."""
        rag_text = (
            self.df["rag_text"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        self.assertTrue(
            (rag_text != "").all(),
            "Certains événements n'ont pas de rag_text."
        )


if __name__ == "__main__":
    unittest.main()