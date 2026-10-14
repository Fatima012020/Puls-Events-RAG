import os
import json
from pathlib import Path
import time

import faiss
import numpy as np

from dotenv import load_dotenv
from mistralai.client import Mistral

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from datetime import datetime, timezone


# =========================
# PARAMÈTRES
# =========================

INDEX_FILE = Path("faiss_index/events.index")
METADATA_FILE = Path("faiss_index/metadata.json")

EMBEDDING_MODEL = "mistral-embed"

# Modèle génératif Mistral
LLM_MODEL = "ministral-3b-2512"

TOP_K = 5


# =========================
# CONFIGURATION API
# =========================

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError(
        "MISTRAL_API_KEY absente du fichier .env"
    )


embedding_client = Mistral(
    api_key=api_key
)


llm = ChatMistralAI(
    model=LLM_MODEL,
    temperature=0.2,
    api_key=api_key
)


# =========================
# CHARGEMENT FAISS
# =========================

def load_vector_database():

    print("Chargement de la base FAISS...")

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    if index.ntotal != len(metadata):
        raise ValueError(
            "Incohérence entre FAISS et les métadonnées."
        )

    print(
        f"Base chargée : {index.ntotal} vecteurs"
    )

    return index, metadata


# =========================
# EMBEDDING DE LA QUESTION
# =========================

def embed_query(query):

    response = embedding_client.embeddings.create(
        model=EMBEDDING_MODEL,
        inputs=[query]
    )

    vector = np.array(
        [response.data[0].embedding],
        dtype="float32"
    )

    faiss.normalize_L2(vector)

    return vector


# =========================
# RECHERCHE FAISS
# =========================

def retrieve_documents(
    query,
    index,
    metadata,
    top_k=TOP_K
):

    query_vector = embed_query(query)

    scores, indices = index.search(
        query_vector,
        top_k
    )

    documents = []

    used_uids = set()

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        item = metadata[idx]

        uid = str(
            item.get("uid", "")
        )

        # Évite plusieurs chunks
        # du même événement
        if uid and uid in used_uids:
            continue

        if uid:
            used_uids.add(uid)

        documents.append({
            "score": float(score),
            "title": item.get(
                "title",
                ""
            ),
            "city": item.get(
                "city",
                ""
            ),
            "department": item.get(
                "department",
                ""
            ),
            "start_date": item.get(
                "start_date",
                ""
            ),
            "end_date": item.get(
                "end_date",
                ""
            ),
            "url": item.get(
                "url",
                ""
            ),
            "text": item.get(
                "text",
                ""
            )
        })

    return documents


# =========================
# CONSTRUCTION DU CONTEXTE
# =========================

def build_context(documents):

    context_parts = []

    for i, doc in enumerate(
        documents,
        start=1
    ):

        city = doc["city"]

        if city in [
            "",
            "nan",
            "None"
        ]:
            city = "Non renseignée"

        block = (
            f"Événement {i}\n"
            f"Titre : {doc['title']}\n"
            f"Ville : {city}\n"
            f"Département : {doc['department']}\n"
            f"Date de début : {doc['start_date']}\n"
            f"Date de fin : {doc['end_date']}\n"
            f"Description : {doc['text']}\n"
            f"URL : {doc['url']}\n"
        )

        context_parts.append(block)

    return "\n".join(
        context_parts
    )


# =========================
# PROMPT LANGCHAIN
# =========================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Tu es l'assistant culturel de Puls-Events.

Tu dois répondre uniquement à partir
des événements présents dans le contexte fourni.

La date actuelle est : {current_date}.

RÈGLES OBLIGATOIRES :
- Réponds uniquement à partir des informations présentes dans le contexte.
- N'invente jamais un événement.
- N'invente jamais une date, un lieu, un horaire, un âge ou une description.
- Ne complète jamais une information manquante par supposition.
- N'utilise pas "probablement", "sans doute", "devrait"
  ou toute autre formulation spéculative.
- Si une information n'est pas présente dans le contexte,
  indique qu'elle n'est pas renseignée.
- Si le contexte ne permet pas de répondre,
  dis clairement que tu ne disposes pas d'informations suffisantes.
- Si la question concerne des événements prochains ou à venir,
  ne présente jamais comme futur un événement déjà terminé
  avant la date actuelle.
- Propose en priorité les événements réellement pertinents
  pour la question.
- Réponds en français.
- Sois clair et concis.
- Ne prétends jamais avoir vérifié l'ensemble des événements de la base.
- Formule tes réponses à partir des événements fournis dans le contexte.
- Pour une recherche ou une recommandation actuelle, propose uniquement
  les événements encore en cours ou à venir présents dans le contexte.
"""
        ),
        (
            "human",
            """
Question :
{question}

Événements disponibles :
{context}

Réponds à la question en t'appuyant
uniquement sur ces informations.
"""
        )
    ]
)


# =========================
# DÉTECTION DES QUESTIONS FUTURES
# =========================

def asks_for_future_events(question):

    keywords = [
        "prochain",
        "prochaine",
        "prochains",
        "prochaines",
        "prochainement",
        "à venir",
        "a venir",
        "futur",
        "future",
        "futurs",
        "futures",
        "prévu",
        "prévue",
        "prévus",
        "prévues",
    ]

    question_lower = question.lower()

    return any(
        keyword in question_lower
        for keyword in keywords
    )

# =========================
# DÉTECTION DES QUESTIONS PASSEES
# =========================

def asks_for_past_events(question):
    """
    Détecte si l'utilisateur recherche explicitement
    des événements passés.
    """
    keywords = [
        "passé",
        "passée",
        "passés",
        "passées",
        "a eu lieu",
        "ont eu lieu",
        "avait lieu",
        "historique",
        "l'année dernière",
        "année dernière",
    ]

    question_lower = question.lower()

    return any(
        keyword in question_lower
        for keyword in keywords
    )

# =========================
# PARSING DES DATES
# =========================

def parse_event_date(value):

    if value is None:
        return None

    value = str(value).strip()

    if not value or value.lower() == "nan":
        return None

    try:
        value = value.replace(
            "Z",
            "+00:00"
        )

        return datetime.fromisoformat(
            value
        )

    except ValueError:
        return None


# =========================
# FILTRE ÉVÉNEMENTS FUTURS
# =========================

def filter_future_events(documents):

    now = datetime.now(
        timezone.utc
    )

    future_documents = []

    for doc in documents:

        date_value = (
            doc.get("end_date")
            or doc.get("start_date")
        )

        event_date = parse_event_date(
            date_value
        )

        if event_date is None:
            continue

        if event_date.tzinfo is None:
            event_date = event_date.replace(
                tzinfo=timezone.utc
            )

        if event_date >= now:
            future_documents.append(
                doc
            )

    return future_documents


# =========================
# GÉNÉRATION RAG
# =========================

def generate_answer(
    question,
    documents
):

    context = build_context(
        documents
    )

    chain = prompt | llm

    max_attempts = 5

    for attempt in range(
        max_attempts
    ):

        try:

            response = chain.invoke(
                {
                    "question": question,
                    "context": context,
                    "current_date": datetime.now().strftime(
                        "%d/%m/%Y"
                    )
                }
            )

            return response.content

        except Exception as error:

            if "429" not in str(error):
                raise

            if attempt == max_attempts - 1:
                raise

            wait_time = 2 ** attempt

            print(
                f"\nLimite API Mistral atteinte. "
                f"Nouvelle tentative dans "
                f"{wait_time} secondes..."
            )

            time.sleep(
                wait_time
            )


# =========================
# CHATBOT
# =========================

def main():

    index, metadata = (
        load_vector_database()
    )

    print(
        "\nChatbot Puls-Events prêt."
    )

    print(
        "Tapez 'quit' pour quitter.\n"
    )

    while True:

        question = input(
            "Vous : "
        ).strip()

        if question.lower() in [
            "quit",
            "exit",
            "q"
        ]:

            print(
                "Fin du chatbot."
            )

            break

        if not question:
            continue

        # Si la question concerne
        # des événements futurs,
        # on récupère davantage de candidats
        # avant de filtrer les dates.
                # Cas 1 : l'utilisateur demande explicitement
        # des événements passés.
        if asks_for_past_events(question):

            documents = retrieve_documents(
                question,
                index,
                metadata,
                top_k=5
            )

        # Cas 2 : recherche normale ou demande
        # explicite d'événements futurs.
        else:

            # On récupère davantage de candidats
            # avant d'appliquer le filtre temporel.
            documents = retrieve_documents(
                question,
                index,
                metadata,
                top_k=20
            )

            documents = filter_future_events(
                documents
            )

            documents = documents[:5]

        if not documents:

            print(
                "\nAssistant : "
                "Je n'ai trouvé aucun événement pertinent.\n"
            )

            continue

        answer = generate_answer(
            question,
            documents
        )

        print(
            "\nAssistant :\n"
        )

        print(
            answer
        )

        print(
            "\n" + "-" * 70 + "\n"
        )


if __name__ == "__main__":
    main()