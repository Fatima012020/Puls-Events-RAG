import json
import unittest
from pathlib import Path

import faiss


INDEX_FILE = Path("faiss_index/events.index")
METADATA_FILE = Path("faiss_index/metadata.json")


class TestFaissIndex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Charge l'index FAISS et les métadonnées une seule fois."""
        cls.index = faiss.read_index(str(INDEX_FILE))

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            cls.metadata = json.load(file)

    def test_index_not_empty(self):
        """Vérifie que l'index FAISS contient des vecteurs."""
        self.assertGreater(
            self.index.ntotal,
            0,
            "L'index FAISS est vide."
        )

    def test_metadata_not_empty(self):
        """Vérifie que les métadonnées sont présentes."""
        self.assertGreater(
            len(self.metadata),
            0,
            "Le fichier de métadonnées est vide."
        )

    def test_vectors_metadata_correspondence(self):
        """Vérifie qu'un ensemble de métadonnées existe pour chaque vecteur."""
        self.assertEqual(
            self.index.ntotal,
            len(self.metadata),
            "Le nombre de vecteurs FAISS ne correspond pas "
            "au nombre de métadonnées."
        )

    def test_index_dimension(self):
        """Vérifie que l'index utilise des vecteurs de dimension 1024."""
        self.assertEqual(
            self.index.d,
            1024,
            "La dimension de l'index FAISS n'est pas égale à 1024."
        )

    def test_required_metadata_fields_exist(self):
        """Vérifie les champs indispensables dans les métadonnées."""
        required_fields = [
            "uid",
            "chunk_id",
            "chunk_number",
            "title",
            "city",
            "department",
            "start_date",
            "end_date",
            "url",
            "text",
        ]

        for item in self.metadata:
            for field in required_fields:
                self.assertIn(
                    field,
                    item,
                    f"Champ '{field}' absent des métadonnées."
                )


if __name__ == "__main__":
    unittest.main()