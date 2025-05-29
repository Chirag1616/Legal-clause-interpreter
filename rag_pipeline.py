# rag_pipeline.py (refactored to use conversational)
from langchain_community.vectorstores import FAISS
from vector_database import faiss_db
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import InferenceClient
import requests
import os

load_dotenv(find_dotenv())

hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

if not hf_token or not hf_token.startswith("hf_"):
    raise ValueError("Missing or incorrect Hugging Face API token")

# Custom wrapper for conversational task
class HFChatModel:
    def __init__(self, model_name, token):
        self.model = model_name
        self.token = token
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def invoke(self, inputs):
        # Allow direct use of formatted prompt via "inputs"
        prompt = inputs.get("inputs") or f"{inputs['context']}\n\nLegal Question: {inputs['question']}\nLegal Answer:"
        payload = {"inputs": prompt}

        response = requests.post(self.api_url, headers=self.headers, json=payload)

        resp_json = response.json()

        # • Most text-generation models return a *list* of dicts
        # • Some pipelines (or errors) may return a single dict
        if isinstance(resp_json, list) and resp_json:
            return resp_json[0].get("generated_text", "[No response returned]")
        elif isinstance(resp_json, dict):
            return resp_json.get("generated_text", "[No response returned]")
        else:
            return "[No response returned]"


llm_model = HFChatModel(model_name, hf_token)
critic_model = HFChatModel(model_name, hf_token)  # Using same model for critique

# Prompt template remains useful to format the question
gen_prompt_template = ChatPromptTemplate.from_template("""
You are a legal assistant AI. Using only the information provided in the context, respond to the user's legal question.
Give short simple answers, and if the question is not answerable based on the context, state that clearly.
Do not make assumptions or provide information not present in the context. If it is gibberish, say so.

If the context lacks sufficient information, state clearly: "The provided context does not contain enough information to answer this question."

Context:
{context}

Legal Question:
{question}

Legal Answer:
""")

'''eval_prompts = [
    ChatPromptTemplate.from_template("""
Evaluate the answer provided. Is it factually correct and faithful to the context?
Context: {context}
Answer: {answer}
Respond with an evaluation and correction if needed.
"""),
    ChatPromptTemplate.from_template("""
Act as a legal expert. Check if the answer violates any factual legal interpretation.
Context: {context}
Answer: {answer}
Provide your critique and corrections if needed.
"""),
    ChatPromptTemplate.from_template("""
You are a helpful peer reviewer. Does the answer follow the instructions and stay within the context?
Context: {context}
Answer: {answer}
Provide constructive feedback.
""")
]
'''
def retrieve_docs(query, k=4):
    results = faiss_db.similarity_search_with_score(query, k=k)
    filtered_results = [doc for doc, score in results if score >= 0.65]
    return filtered_results

def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

def generate_answer(query, documents, model):
    context = get_context(documents)
    prompt_inputs = {"question": query, "context": context}
    return model.invoke(prompt_inputs), context


final_eval_prompt = ChatPromptTemplate.from_template("""
You are a legal assistant. Improve the given answer using the context if necessary, and return only the final correct legal answer.

- Only respond to the legal question.
- Be concise (1–2 sentences).
- Do not repeat the question or include any explanation.
- Do not add background about the UDHR, its history, or related articles.
- Do not include links or sources.

Context:
{context}

Legal Question:
{question}

Initial Answer:
{answer}

Return only the final, corrected legal answer:
""")


def critique_and_correct_answer(initial_answer, context, question, model):
    # Combine everything into one string for the "question" field
    prompt_text = f"""You are a legal assistant. Improve the given answer using the context if necessary, and return only the final correct legal answer.
- Be concise , just 6-7 sentences.



Context:
{context}

Legal Question:
{question}

Initial Answer:
{initial_answer}

Return only the final, corrected legal answer:"""

    # Now pass a dict with both 'question' and 'context'
    return model.invoke({
        "question": prompt_text,
        "context": ""
    }).strip()

def self_correcting_query(query, documents, model1, model2):
    if not documents:
        return "No relevant documents found to answer this question."

    # Generate initial answer from first model
    initial_answer, context = generate_answer(query, documents, model1)

    # Get clean, final answer from second model
    final_answer = critique_and_correct_answer(initial_answer, context, query, model2)

    return final_answer
    