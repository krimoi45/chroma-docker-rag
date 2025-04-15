# RAG avec ChromaDB et Docker

## Prérequis
- Docker
- Docker Compose

## Installation et Démarrage

1. Clonez le dépôt
```bash
git clone https://github.com/krimoi45/chroma-docker-rag.git
cd chroma-docker-rag
```

2. Démarrez les services
```bash
docker-compose up --build
```

## Architecture

- ChromaDB : Base de données vectorielle
- Python App : Script de démonstration RAG
- Docker Compose : Orchestration des services

## Fonctionnalités

- Création de collections vectorielles
- Recherche de similarité sémantique
- Configuration dynamique avec variables d'environnement

## Technologies

- ChromaDB
- Sentence Transformers
- Docker
- Python

## Utilisation

Le script démontre :
- La création d'une collection de documents
- La génération d'embeddings
- La recherche de documents similaires par similarité sémantique

## Personnalisation

Modifiez `main.py` pour ajouter vos propres documents et requêtes de recherche.
