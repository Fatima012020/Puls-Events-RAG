import unittest
import pandas as pd
from pathlib import Path


CHUNKS_FILE = Path("data/chunks/events_chunks.csv")


class TestChunking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Charge le fichier des chunks une seule fois pour tous les tests."""
        cls.df = pd.read_csv(CHUNKS_FILE)

    def test_chunks_file_not_empty(self):
        """Vérifie que le fichier contient au moins un chunk."""
        self.assertFalse(
            self.df.empty,
            "Le fichier des chunks est vide."
        )

    def test_chunk_text_not_empty(self):
        """Vérifie que tous les chunks contiennent du texte."""
        chunk_text = (
            self.df["chunk_text"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        self.assertTrue(
            (chunk_text != "").all(),
            "Certains chunks ne contiennent aucun texte."
        )

    def test_chunk_id_unique(self):
        """Vérifie que chaque chunk possède un identifiant unique."""
        self.assertTrue(
            self.df["chunk_id"].is_unique,
            "Des chunk_id dupliqués sont présents."
        )

    def test_uid_not_empty(self):
        """Vérifie que chaque chunk est associé à un événement."""
        uid = (
            self.df["uid"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        self.assertTrue(
            (uid != "").all(),
            "Certains chunks ne sont associés à aucun UID."
        )

    def test_chunk_number_not_negative(self):
        """Vérifie que les numéros de chunks sont valides."""
        self.assertTrue(
            (self.df["chunk_number"] >= 0).all(),
            "Certains numéros de chunks sont négatifs."
        )

    def test_required_columns_exist(self):
        """Vérifie la présence des colonnes nécessaires au pipeline RAG."""
        required_columns = [
            "uid",
            "chunk_id",
            "chunk_number",
            "title",
            "city",
            "department",
            "start_date",
            "end_date",
            "url",
            "chunk_text",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.df.columns
        ]

        self.assertEqual(
            missing_columns,
            [],
            f"Colonnes manquantes : {missing_columns}"
        )


if __name__ == "__main__":
    unittest.main()