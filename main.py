import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Configuration du client ChromaDB avec les variables d'environnement Docker
chroma_host = os.getenv('CHROMA_SERVER_HOST', 'localhost')
chroma_port = os.getenv('CHROMA_SERVER_PORT', '8000')

print(f"Connexion à ChromaDB sur {chroma_host}:{chroma_port}")

# Configuration du client ChromaDB
chroma_client = chromadb.HttpClient(
    host=chroma_host, 
    port=chroma_port
)

# Modèle d'embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

def creer_collection(nom_collection='documents_test'):
    """Créer une collection dans ChromaDB"""
    try:
        # Supprimer la collection si elle existe déjà
        chroma_client.delete_collection(name=nom_collection)
    except:
        pass

    # Créer une nouvelle collection
    collection = chroma_client.create_collection(name=nom_collection)
    
    # Documents d'exemple
    documents = [
        "Le machine learning est un domaine passionnant de l'intelligence artificielle.",
        "Python est un langage de programmation très populaire pour le développement de logiciels.",
        "Les réseaux de neurones profonds permettent des avancées remarquables en traitement du langage naturel.",
        "La data science combine statistiques, programmation et analyse de données.",
        "L'apprentissage automatique trouve des applications dans de nombreux domaines comme la santé et la finance."
    ]

    # Générer des embeddings
    embeddings = model.encode(documents)

    # Ajouter les documents à la collection
    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        collection.add(
            embeddings=[embedding.tolist()],
            documents=[doc],
            ids=[f"doc_{i}"]
        )
    
    return collection

def recherche_similaire(collection, requete, top_k=2):
    """Rechercher des documents similaires"""
    # Embedding de la requête
    requete_embedding = model.encode([requete])[0].tolist()
    
    # Recherche des documents similaires
    resultats = collection.query(
        query_embeddings=[requete_embedding],
        n_results=top_k
    )
    
    return resultats['documents'][0]

def main():
    # Créer une collection
    collection = creer_collection()
    
    # Exemples de recherche
    requetes = [
        "Intelligence artificielle",
        "Programmation",
        "Analyse de données"
    ]
    
    print("\n--- Recherche de documents similaires ---")
    for requete in requetes:
        resultats = recherche_similaire(collection, requete)
        print(f"\nRequête : {requete}")
        print("Documents similaires :")
        for doc in resultats:
            print(f"- {doc}")

if __name__ == "__main__":
    main()