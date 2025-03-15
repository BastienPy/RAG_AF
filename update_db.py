import os
import glob
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME, FINANCIAL_REPORTS_DIR

def load_financial_reports(directory):
    """
    Charge tous les rapports financiers (fichiers PDF) depuis le dossier.
    Pour chaque PDF, on utilise PyPDFLoader de LangChain pour extraire le texte.
    """
    reports = []
    file_paths = glob.glob(os.path.join(directory, "*.pdf"))
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        docs = loader.load()  # Retourne une liste de Document
        # On combine le contenu de toutes les pages en une seule chaîne
        content = "\n".join([doc.page_content for doc in docs])
        reports.append({
            "id": os.path.basename(file_path),
            "content": content
        })
    return reports

def update_vector_store():
    #Initialisation du modèle d'embeddings
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    #Initialisation du client ChromaDB
    client = chromadb.Client(persist_directory=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)

    #Chargement des rapports financiers
    reports = load_financial_reports(FINANCIAL_REPORTS_DIR)
    if not reports:
        print("Aucun rapport PDF trouvé dans", FINANCIAL_REPORTS_DIR)
        return

    #Ajout ou mise à jour de chaque rapport dans la collection
    for report in reports:
        embedding = embedder.encode(report["content"]).tolist()
        #Pour simplifier, on ajoute le rapport directement (en production, penser à gérer les doublons ou mises à jour)
        collection.add(
            ids=[report["id"]],
            documents=[report["content"]],
            embeddings=[embedding]
        )
        print(f"Ajout/MAJ du rapport {report['id']} effectué.")

if __name__ == "__main__":
    update_vector_store()
