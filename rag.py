import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME, HUGGINGFACE_API_KEY, HUGGINGFACE_MODEL_NAME

#Initialisation globale du modèle d'embeddings
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

#Initialisation du client ChromaDB et récupération de la collection
client = chromadb.Client(persist_directory=CHROMA_PERSIST_DIR)
collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)

def retrieve_context(query, n_results=1):
    """
    Calcule l'embedding de la requête et recherche le document le plus pertinent.
    """
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    if results and results.get("documents") and results["documents"][0]:
        return results["documents"][0][0]
    else:
        return None

def generate_response(query, context):
    """
    Construit un prompt à partir du contexte récupéré et de la question,
    puis interroge l'API Hugging Face pour générer une réponse.
    """
    prompt = (
        f"Analyse financière automatisée:\n"
        f"Document contextuel : {context}\n"
        f"Question : {query}\n"
        f"Réponse :"
    )
    API_URL = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL_NAME}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    try:
        result = response.json()[0]['generated_text']
    except Exception as e:
        result = "Erreur lors de la génération de la réponse."
    return result
