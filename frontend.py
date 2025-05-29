from rag_pipeline import self_correcting_query, retrieve_docs, llm_model, critic_model
import streamlit as st
import re

uploaded_file = st.file_uploader("Upload PDF",
                                 type="pdf",
                                 accept_multiple_files=False)


#Step2: Chatbot Skeleton (Question & Answer)

user_query = st.text_area("Enter your prompt: ", height=150 , placeholder= "Ask Anything!")

ask_question = st.button("Ask AI Lawyer")

if ask_question:

    if uploaded_file: 

        st.chat_message("user").write(user_query)

        # RAG Pipeline
        retrieved_docs=retrieve_docs(user_query)
        response=self_correcting_query(documents=retrieved_docs, model1=llm_model, query=user_query, model2=critic_model)
        #fixed_response = "Hi, this is a fixed response!"
        
        result = {}
        result['answer'] = response
        result['answer'] = response.split('Answer')[-1].strip()

        st.chat_message("AI Lawyer").write(result['answer'])
    
    else:
        st.error("Kindly upload a valid PDF file first!")