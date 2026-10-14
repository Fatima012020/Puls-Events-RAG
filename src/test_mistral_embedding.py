import os

from dotenv import load_dotenv
from mistralai.client import Mistral


# Charger les variables du fichier .env
load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError(
        "MISTRAL_API_KEY est absente du fichier .env"
    )


# Création du client Mistral
client = Mistral(api_key=api_key)


# Petit échantillon de test
texts = [
    "Festival culturel à Orléans",
    "Exposition artistique dans le Loiret"
]


print("Test de connexion à Mistral...")


# Génération des embeddings
response = client.embeddings.create(
    model="mistral-embed",
    inputs=texts
)


print("Connexion Mistral : OK")
print(f"Nombre de vecteurs : {len(response.data)}")


for i, item in enumerate(response.data):
    print(
        f"Vecteur {i + 1} : "
        f"{len(item.embedding)} dimensions"
    )


print("\nTest des embeddings Mistral réussi !")