import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import os
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME, HUGGINGFACE_MODEL_NAME

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if HUGGINGFACE_API_KEY:
    print("API key retrieved successfully!")
else:
    print("API key not found. Make sure it's set in your environment.")
    
#Initialisation globale du modèle d'embeddings
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

#Initialisation du client ChromaDB et récupération de la collection
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)

def retrieve_context(query, n_results=10):
    """
    Calcule l'embedding de la requête et recherche les n_results documents les plus pertinents.
    Retourne une chaîne concaténée des résultats.
    """
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    if results and results.get("documents") and results["documents"][0]:
        # results["documents"][0] est une liste de documents correspondant aux top-k résultats.
        # On les concatène pour obtenir un contexte complet.
        context = "\n".join(results["documents"][0])
        return context
    else:
        return None


'''def generate_response(query, context):
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
        print(e)
        result = "Erreur lors de la génération de la réponse."
    return result'''

def generate_response(query, context):
    prompt = (
        f"Analyse financière automatisée:\n"
        f"Document contextuel : {context}\n"
        f"Question : {query}\n"
        f"Réponse :"
    )
    API_URL = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL_NAME}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    
    # Afficher le texte brut de la réponse pour débogage
    print("Response status code:", response.status_code)
    print("Response text:", response.text)
    
    try:
        result = response.json()[0]['generated_text']
    except Exception as e:
        print("Erreur lors de la conversion en JSON:", e)
        result = "Erreur lors de la génération de la réponse."
    return result

from transformers import pipeline
from config import HUGGINGFACE_MODEL_NAME

def generate_response_local(query, context):
    prompt = (
        f"Analyse financière automatisée:\n"
        f"Document contextuel : {context}\n"
        f"Question : {query}\n"
        f"Réponse :"
    )
    
    # Initialiser le générateur en local
    generator = pipeline("text-generation", model=HUGGINGFACE_MODEL_NAME)
    
    # Générer la réponse (vous pouvez ajuster max_length et d'autres paramètres)
    outputs = generator(prompt, max_length=300, do_sample=False, truncation=True)
    print(outputs)
    # Récupérer le texte généré
    generated_text = outputs[0]['generated_text']
    return generated_text

