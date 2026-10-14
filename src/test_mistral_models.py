import os
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("MISTRAL_API_KEY introuvable")

client = Mistral(api_key=api_key)

print("Test de connexion à l'API Mistral...")

models = client.models.list()

print("Connexion API : OK")
print("Nombre de modèles accessibles :", len(models.data))

for model in models.data:
    if "mistral-small" in model.id:
        print("Modèle trouvé :", model.id)