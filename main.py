import streamlit as st
from rag import retrieve_context, generate_response

st.title("Analyse Financière Automatisée - RAG")
st.write("Posez votre question sur l'analyse financière et obtenez une réponse basée sur des rapports financiers PDF.")

query = st.text_input("Votre question :")

if query:
    context = retrieve_context(query)
    if context:
        st.markdown("**Document récupéré :**")
        st.write(context)
        st.markdown("**Génération de la réponse...**")
        answer = generate_response(query, context)
        st.markdown("**Réponse :**")
        st.write(answer)
    else:
        st.error("Aucun document pertinent trouvé dans la base de données.")
