import os
import json
import time
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")

if not API_KEY:
    raise RuntimeError("MISTRAL_API_KEY introuvable.")

BASE_URL = "https://api.mistral.ai/v1"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

MODELS = [
    "ministral-3b-2512",
]


def print_response(model, response):
    print("\n" + "=" * 75)
    print(f"MODÈLE : {model}")
    print("=" * 75)

    print(f"HTTP status : {response.status_code}")

    print("\nEn-têtes de rate limit :")

    rate_headers = {}

    for key, value in response.headers.items():
        key_lower = key.lower()

        if (
            "ratelimit" in key_lower
            or "rate-limit" in key_lower
            or "retry-after" in key_lower
        ):
            rate_headers[key] = value
            print(f"  {key}: {value}")

    if not rate_headers:
        print("  Aucun en-tête de rate limit retourné.")

    print("\nRequest ID :")

    request_id_found = False

    for key, value in response.headers.items():
        if (
            "request-id" in key.lower()
            or "request_id" in key.lower()
        ):
            print(f"  {key}: {value}")
            request_id_found = True

    if not request_id_found:
        print("  Aucun Request ID trouvé.")

    print("\nCorps de la réponse :")

    try:
        body = response.json()

        # Évite d'imprimer une réponse de génération inutilement longue
        if response.status_code == 200:
            print("  Requête réussie.")

            if "choices" in body and body["choices"]:
                content = body["choices"][0].get(
                    "message", {}
                ).get("content")

                print(f"  Réponse : {content}")

        else:
            # Pour une erreur, on affiche le corps complet
            print(json.dumps(
                body,
                indent=2,
                ensure_ascii=False
            ))

    except ValueError:
        print(response.text)

    return rate_headers


def test_chat(model):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Réponds uniquement par : Test réussi"
            }
        ],
        "max_tokens": 20,
        "temperature": 0,
    }

    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=30,
        )

        headers = print_response(model, response)

        return {
            "model": model,
            "status": response.status_code,
            "headers": headers,
        }

    except requests.RequestException as exc:
        print(f"\nErreur réseau avec {model}:")
        print(repr(exc))

        return {
            "model": model,
            "status": "NETWORK ERROR",
            "headers": {},
        }


if __name__ == "__main__":

    print("Comparaison des rate limits Mistral")
    print("La clé API ne sera jamais affichée.")

    results = []

    for index, model in enumerate(MODELS):

        result = test_chat(model)
        results.append(result)

        # Évite que le deuxième appel soit confondu
        # avec une limite de requêtes/seconde.
        if index < len(MODELS) - 1:
            print("\nAttente de 5 secondes avant le modèle suivant...")
            time.sleep(5)

    print("\n\n" + "=" * 75)
    print("RÉSUMÉ COMPARATIF")
    print("=" * 75)

    for result in results:

        print(f"\n{result['model']}")
        print(f"  HTTP : {result['status']}")

        for key, value in result["headers"].items():
            print(f"  {key}: {value}")

    print("\nDiagnostic terminé.")