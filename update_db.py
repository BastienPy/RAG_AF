import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import chromadb
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

def chunk_text(text, max_words=300, overlap=50):
    """
    Découpe le texte en segments contenant au maximum max_words mots,
    avec un chevauchement de 'overlap' mots entre deux chunks pour conserver le contexte.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = words[start:start+max_words]
        chunks.append(" ".join(chunk))
        start += max_words - overlap  # avance en laissant un chevauchement
    return chunks

def update_vector_store():
    # Initialisation du modèle d'embeddings
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Initialisation du client ChromaDB avec persistance
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)

    # Chargement des rapports financiers
    reports = load_financial_reports(FINANCIAL_REPORTS_DIR)
    if not reports:
        print("Aucun rapport PDF trouvé dans", FINANCIAL_REPORTS_DIR)
        return

    # Récupération des IDs déjà présents dans la collection (s'il y en a)
    try:
        existing_data = collection.get()
        existing_ids = set(existing_data["ids"])
    except Exception as e:
        existing_ids = set()

    for report in reports:
        report_id = report["id"]
        report_content = report["content"]
        # Découpage du contenu en chunks
        chunks = chunk_text(report_content, max_words=200, overlap=50)

        for i, chunk in enumerate(chunks):
            # Créer un identifiant unique pour chaque chunk
            chunk_id = f"{report_id}_chunk{i}"
            
            if chunk_id in existing_ids:
                # Récupération du document existant pour ce chunk
                existing_doc_data = collection.get(ids=[chunk_id])
                existing_doc = existing_doc_data["documents"][0] if existing_doc_data["documents"] else ""
                if existing_doc == chunk:
                    print(f"Le chunk {chunk_id} existe déjà et est identique. On ne l'ajoute pas.")
                    continue
                else:
                    # Le contenu a changé, on met à jour le chunk
                    embedding = embedder.encode(chunk).tolist()
                    collection.update(
                        ids=[chunk_id],
                        documents=[chunk],
                        embeddings=[embedding]
                    )
                    print(f"Le chunk {chunk_id} a été mis à jour.")
            else:
                # Nouveau chunk, on l'ajoute
                embedding = embedder.encode(chunk).tolist()
                collection.add(
                    ids=[chunk_id],
                    documents=[chunk],
                    embeddings=[embedding]
                )
                print(f"Ajout du chunk {chunk_id} effectué.")

if __name__ == "__main__":
    update_vector_store()
