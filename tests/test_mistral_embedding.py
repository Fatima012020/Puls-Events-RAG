"""
Tests d'intégration de l'API Mistral Embeddings.

Ces tests effectuent de véritables appels à l'API Mistral.
Ils sont exécutés uniquement lorsque la variable
RUN_MISTRAL_INTEGRATION_TESTS vaut "1".
"""

import os
import unittest

from dotenv import load_dotenv
from mistralai.client import Mistral


@unittest.skipUnless(
    os.getenv("RUN_MISTRAL_INTEGRATION_TESTS") == "1",
    "Test d'intégration Mistral désactivé par défaut."
)
class TestMistralEmbedding(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialise le client Mistral pour les tests d'intégration."""
        load_dotenv()

        api_key = os.getenv("MISTRAL_API_KEY")

        if not api_key:
            raise unittest.SkipTest(
                "MISTRAL_API_KEY absente : test d'intégration ignoré."
            )

        cls.client = Mistral(api_key=api_key)

    def test_embedding_generation(self):
        """Vérifie qu'un embedding est généré pour chaque texte."""
        texts = [
            "Festival culturel à Orléans",
            "Exposition artistique dans le Loiret"
        ]

        response = self.client.embeddings.create(
            model="mistral-embed",
            inputs=texts
        )

        self.assertEqual(len(response.data), len(texts))

    def test_embedding_dimension(self):
        """Vérifie que les embeddings ont 1 024 dimensions."""
        response = self.client.embeddings.create(
            model="mistral-embed",
            inputs=["Festival culturel à Orléans"]
        )

        vector = response.data[0].embedding

        self.assertEqual(len(vector), 1024)


if __name__ == "__main__":
    unittest.main()