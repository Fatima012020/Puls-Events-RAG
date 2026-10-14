import os

from dotenv import load_dotenv
from mistralai.client import Mistral


load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("MISTRAL_API_KEY absente du fichier .env")


client = Mistral(api_key=api_key)


print("Test du modèle de génération Mistral...")


response = client.chat.complete(
    model="mistral-small-2603",
    messages=[
        {
            "role": "user",
            "content": "Réponds uniquement par : Test réussi"
        }
    ],
    max_tokens=20
)


print(
    response.choices[0].message.content
)

print("\nAPI Chat Mistral : OK")