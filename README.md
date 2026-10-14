# Puls-Events — Système RAG de recommandation d'événements

## 1. Présentation du projet

Ce projet a été réalisé dans le cadre du parcours **Data Engineer OpenClassrooms**.

L'objectif est de concevoir un **Proof of Concept (POC) de chatbot RAG** pour la plateforme **Puls-Events**, spécialisée dans les événements culturels.

Le système permet à un utilisateur de poser une question en langage naturel, par exemple :

- « Je cherche un spectacle pour enfants dans le Loiret »
- « Je cherche une exposition ou un musée dans le Loiret »
- « Quels événements sont prévus prochainement dans le Loiret ? »

Le chatbot recherche les événements les plus pertinents dans une base vectorielle FAISS, puis utilise un modèle Mistral pour générer une réponse contextualisée.

Les données proviennent d'**OpenAgenda**.

---

## 2. Objectifs

Les principaux objectifs du projet sont :

- collecter des événements publics OpenAgenda ;
- conserver les événements du département du Loiret ;
- nettoyer et préparer les données ;
- créer un texte exploitable pour un système RAG ;
- découper les textes en chunks ;
- vectoriser les chunks avec les embeddings Mistral ;
- construire un index vectoriel avec FAISS ;
- rechercher les événements les plus proches sémantiquement d'une question ;
- utiliser LangChain pour orchestrer la génération de réponse ;
- générer des recommandations avec un modèle Mistral ;
- limiter les hallucinations grâce à un prompt contraint ;
- gérer les événements passés et futurs selon la question de l'utilisateur ;
- tester la qualité des données et le fonctionnement du pipeline.

---

## 3. Architecture du système

Le pipeline général est le suivant :

```text
OpenAgenda
    |
    v
Collecte des événements
    |
    v
Nettoyage / Pré-processing
    |
    v
Création du champ rag_text
    |
    v
Découpage en chunks
    |
    v
Embeddings Mistral
    |
    v
Normalisation L2
    |
    v
Index FAISS
    |
    v
Question utilisateur
    |
    v
Embedding de la question
    |
    v
Recherche sémantique FAISS
    |
    v
Déduplication des événements
    |
    v
Filtrage temporel si nécessaire
    |
    v
Construction du contexte
    |
    v
LangChain
    |
    v
Modèle Mistral
    |
    v
Réponse du chatbot
```

---

## 4. Technologies utilisées

Le projet repose principalement sur :

- **Python 3.11**
- **Pandas**
- **NumPy**
- **FAISS CPU**
- **LangChain**
- **LangChain Text Splitters**
- **LangChain MistralAI**
- **Mistral AI API**
- **python-dotenv**
- **Requests**
- **Unittest**

Deux usages de Mistral sont présents dans le projet :

- `mistral-embed` pour la vectorisation ;
- `ministral-3b-2512` pour la génération des réponses du chatbot.

---

## 5. Structure du projet

```text
Concevez_et_deployez_un_systeme_RAG/
│
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── test_environment.py
│
├── data/
│   ├── raw/
│   │   └── events_raw.csv
│   ├── processed/
│   │   └── events_processed.csv
│   ├── chunks/
│   │   └── events_chunks.csv
│   └── vectorized/
│       ├── chunks_embeddings.json
│       └── events_embeddings.json
│
├── faiss_index/
│   ├── events.index
│   └── metadata.json
│
├── src/
│   ├── collect_openagenda.py
│   ├── preprocess_events.py
│   ├── chunk_events.py
│   ├── vectorize_chunks.py
│   ├── build_faiss_index.py
│   ├── search_faiss.py
│   ├── rag_chatbot.py
│   ├── diagnose_mistral_api.py
│   ├── test_mistral_chat.py
│   ├── test_mistral_embedding.py
│   ├── test_mistral_models.py
│   └── vectorize_events.py
│
└── tests/
    ├── test_mistral_embedding.py
    └── test_preprocessing.py
```

---

## 6. Description des principaux fichiers

### `src/collect_openagenda.py`

Collecte les événements depuis OpenAgenda et les enregistre dans :

```text
data/raw/events_raw.csv
```

### `src/preprocess_events.py`

Nettoie et prépare les données : nettoyage des champs textuels, filtrage géographique et temporel, suppression des doublons et création du champ `rag_text`.

Le fichier généré est :

```text
data/processed/events_processed.csv
```

### `src/chunk_events.py`

Découpe les textes avec `RecursiveCharacterTextSplitter`.

Configuration utilisée :

```text
chunk_size = 800
chunk_overlap = 100
```

Le fichier produit est :

```text
data/chunks/events_chunks.csv
```

### `src/vectorize_chunks.py`

Vectorise chaque chunk avec le modèle :

```text
mistral-embed
```

Les embeddings obtenus ont une dimension de **1024**.

Le résultat est enregistré dans :

```text
data/vectorized/chunks_embeddings.json
```

### `src/build_faiss_index.py`

Construit l'index vectoriel FAISS à partir des embeddings normalisés.

L'index utilisé est :

```text
faiss.IndexFlatIP
```

Les fichiers générés sont :

```text
faiss_index/events.index
faiss_index/metadata.json
```

### `src/search_faiss.py`

Permet de tester directement la recherche vectorielle et d'observer les événements retournés par FAISS.

### `src/rag_chatbot.py`

Contient le système RAG final. Le chatbot :

1. charge l'index FAISS ;
2. vectorise la question utilisateur ;
3. recherche les événements pertinents ;
4. évite les doublons liés aux différents chunks ;
5. filtre les événements selon leur date lorsque nécessaire ;
6. construit le contexte ;
7. utilise LangChain pour créer la chaîne de génération ;
8. interroge le modèle Mistral ;
9. génère une réponse contextualisée en français.

Le prompt impose notamment au modèle de ne pas inventer d'événement, de date, de lieu ou d'information absente du contexte.

### Scripts de diagnostic Mistral

Les fichiers suivants ont été utilisés pour vérifier le fonctionnement de l'API Mistral :

```text
src/test_mistral_chat.py
src/test_mistral_embedding.py
src/test_mistral_models.py
src/diagnose_mistral_api.py
```

---

## 7. Installation

### 7.1 Se placer dans le projet

```powershell
cd Concevez_et_deployez_un_systeme_RAG
```

### 7.2 Créer l'environnement virtuel

```powershell
python -m venv .venv
```

### 7.3 Activer l'environnement

```powershell
.venv\Scripts\Activate.ps1
```

### 7.4 Installer les dépendances

```powershell
pip install -r requirements.txt
```

---

## 8. Configuration de l'API Mistral

Créer un fichier `.env` à la racine du projet et ajouter :

```env
MISTRAL_API_KEY=VOTRE_CLE_API
```

La clé API ne doit jamais être publiée dans Git ou intégrée directement au code.

Le fichier `.env` doit être exclu via `.gitignore`.

---

## 9. Vérification de l'environnement

Lancer :

```powershell
python test_environment.py
```

Ce script vérifie notamment la disponibilité de LangChain, FAISS et Mistral AI.

---

## 10. Reproduction complète du pipeline

Tous les scripts doivent être lancés depuis la racine du projet.

### Étape 1 — Collecte

```powershell
python src/collect_openagenda.py
```

Résultat :

```text
data/raw/events_raw.csv
```

### Étape 2 — Pré-processing

```powershell
python src/preprocess_events.py
```

Résultat :

```text
data/processed/events_processed.csv
```

### Étape 3 — Chunking

```powershell
python src/chunk_events.py
```

Résultat :

```text
data/chunks/events_chunks.csv
```

### Étape 4 — Vectorisation

```powershell
python src/vectorize_chunks.py
```

Résultat :

```text
data/vectorized/chunks_embeddings.json
```

### Étape 5 — Construction de l'index FAISS

```powershell
python src/build_faiss_index.py
```

Résultats :

```text
faiss_index/events.index
faiss_index/metadata.json
```

### Étape 6 — Test de recherche vectorielle

```powershell
python src/search_faiss.py
```

### Étape 7 — Lancement du chatbot RAG

```powershell
python src/rag_chatbot.py
```

Exemples de questions :

```text
Je cherche un spectacle pour enfants dans le Loiret
```

```text
Je cherche une exposition ou un musée dans le Loiret
```

```text
Quels événements sont prévus prochainement dans le Loiret ?
```

Pour quitter :

```text
quit
```

---

## 11. Tests

Les tests sont regroupés dans le dossier :

```text
tests/
```

Pour lancer les tests de preprocessing :

```powershell
python -m unittest tests/test_preprocessing.py -v
```

Les tests vérifient notamment :

- que le dataset n'est pas vide ;
- que les événements appartiennent au Loiret ;
- que la logique temporelle est cohérente ;
- que `rag_text` n'est pas vide ;
- que les titres sont renseignés ;
- que les identifiants d'événements sont uniques.

Un test dédié aux embeddings Mistral est également présent.

---

## 12. Résultats obtenus

Le pipeline RAG Puls-Events a produit les résultats suivants, vérifiés sur les fichiers locaux :

- **4 554 événements prétraités**.
- **7 548 chunks** générés.
- **7 548 vecteurs** dans l'index FAISS.
- **1 024 dimensions** par vecteur.
- Recherche sémantique avec `faiss.IndexFlatIP` et normalisation L2.
- Génération de réponses contextualisées avec LangChain et Mistral AI.

### Tests automatisés

Dernière exécution locale :

- **17 tests réussis**.
- **2 tests d'intégration Mistral ignorés par défaut**.
- **0 échec**.

Commande d'exécution :

```powershell
python -m pytest tests/ -q
```

Les tests d'intégration Mistral peuvent être activés avec la variable d'environnement `RUN_MISTRAL_INTEGRATION_TESTS=1`.

### Reproductibilité

Les fichiers de données générés, les embeddings et l'index FAISS sont exclus du dépôt GitHub. Ils peuvent être reconstruits à partir des scripts du pipeline et des données publiques OpenAgenda, sous réserve de la disponibilité des services externes.

Les tests de qualité des données nécessitent actuellement le fichier local `data/processed/events_processed.csv`.

## 13. Choix techniques

### FAISS

FAISS a été choisi pour sa simplicité et ses performances dans le cadre d'un POC local.

L'index `IndexFlatIP` réalise une recherche exacte. Les vecteurs étant normalisés avec L2, le produit scalaire permet d'obtenir une mesure équivalente à la similarité cosinus.

Compte tenu du volume de 7 612 vecteurs, une recherche approximative plus complexe n'était pas nécessaire.

### Mistral Embeddings

Le modèle `mistral-embed` transforme les textes et les questions en représentations vectorielles sémantiques.

Cela permet de retrouver des événements même lorsque la question utilisateur n'utilise pas exactement les mêmes mots que les descriptions OpenAgenda.

### LangChain

LangChain est utilisé pour orchestrer la génération de réponse : construction du prompt, injection du contexte récupéré et appel au modèle génératif.

### Filtrage temporel

FAISS réalise une recherche sémantique et ne connaît pas naturellement la notion de date actuelle.

Un filtrage Python supplémentaire est donc appliqué pour les recherches actuelles et futures afin d'éviter qu'un événement passé soit présenté comme événement « prochain ».

---

## 14. Limites du POC

Le système actuel présente plusieurs limites :

- certaines métadonnées OpenAgenda peuvent être incomplètes ;
- la qualité des réponses dépend directement de la qualité des données sources ;
- le chatbot ne conserve pas d'historique conversationnel ;
- les résultats sont limités aux événements présents dans le corpus ;
- l'index FAISS est local ;
- certains événements OpenAgenda peuvent être enregistrés très longtemps à l'avance ;
- le POC ne garantit pas la disponibilité réelle d'un événement en dehors des informations fournies par OpenAgenda.

---

## 15. Améliorations possibles

Plusieurs évolutions pourraient être envisagées :

- ajouter une interface web ;
- ajouter un historique conversationnel ;
- améliorer le filtrage par date, ville ou catégorie ;
- permettre à l'utilisateur de préciser une période ;
- appliquer un reranking des résultats ;
- ajouter une évaluation automatique du retrieval ;
- automatiser la mise à jour des données OpenAgenda ;
- déployer le chatbot sous forme d'API ;
- utiliser une base vectorielle persistante et distribuée pour un volume plus important.

---

## 16. Conclusion

Ce POC démontre la mise en œuvre complète d'un système **Retrieval-Augmented Generation (RAG)**.

Il associe :

```text
OpenAgenda
+ Python
+ Mistral Embeddings
+ FAISS
+ LangChain
+ Mistral LLM
```

Le pipeline couvre les étapes suivantes :

```text
Collecte
→ Nettoyage
→ Chunking
→ Vectorisation
→ Indexation
→ Retrieval
→ Augmentation du contexte
→ Génération
→ Tests
```

Ce système permet ainsi de proposer des recommandations d'événements contextualisées dans le Loiret à partir de données publiques OpenAgenda.
