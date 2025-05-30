from langchain_community.vectorstores import FAISS
from vector_database import faiss_db
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
import openai
import os
import time

load_dotenv(find_dotenv())

# Use Groq API
openai.api_key = os.environ.get("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"
model_name = "deepseek-r1-distill-llama-70b"

if not openai.api_key:
    raise ValueError("Missing GROQ_API_KEY in environment")

response_cache = {}

def cached_invoke(model, prompt_dict):
    key = str(prompt_dict)
    if key in response_cache:
        return response_cache[key]
    response = model.invoke(prompt_dict)
    response_cache[key] = response
    return response

# --- Groq LLM Wrapper ---
class GroqChatModel:
    def __init__(self, model_name):
        self.model = model_name

    def invoke(self, inputs, retries=2):
        if "inputs" in inputs:
            prompt = inputs["inputs"]
        elif "question" in inputs and "context" in inputs:
            prompt = f"{inputs['context']}\n\nLegal Question: {inputs['question']}\nLegal Answer:"
        elif "question" in inputs:
            prompt = inputs["question"]
        else:
            return "[No input provided]"

        messages = [
            {"role": "system", "content": "You are a helpful legal assistant."},
            {"role": "user", "content": prompt}
        ]

        for attempt in range(retries + 1):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7
                )
                return response["choices"][0]["message"]["content"]
            except Exception as e:
                print(f" Error: {e} (attempt {attempt + 1})")
                time.sleep(1.5 * (attempt + 1))

        return "[Failed after retries]"


llm_model = GroqChatModel(model_name)
critic_model = GroqChatModel(model_name)

gen_prompt_template = ChatPromptTemplate.from_template("""
You are a legal assistant AI. Using only the information provided in the context, respond to the user's legal question.
Do not make assumptions or provide information not present in the context.
Be concise and short (6-7 sentences max).
If the context lacks sufficient information, say: "The context does not contain enough information."
Context:
{context}
Legal Question:
{question}
Legal Answer:
""")

def classify_query_type(question, model):
    print(f" Classifying query type for: {question}")
    prompt = f"""Classify the following legal question into one or more of: factual, analytical, creative, procedural, opinion-based, comparison.
Question: \"{question}\"
Respond with a comma-separated list."""
    result = model.invoke({"inputs": prompt})
    print(f" Query type classified as: {result}")
    return [cat.strip() for cat in result.lower().split(",")]

def select_evaluation_checks(query_type_list):
    print(f" Selecting evaluation checks for query types: {query_type_list}")
    eval_map = {
        "factual": ["hallucination", "relevance", "contradiction"],
        "analytical": ["reasoning", "contradiction", "completeness"],
        "creative": ["vagueness", "relevance"],
        "procedural": ["completeness", "clarity"],
        "opinion-based": ["bias", "justification"],
        "comparison": ["balance", "missing_dimensions"]
    }
    selected = set()
    for qtype in query_type_list:
        selected.update(eval_map.get(qtype, []))
    selected_checks = list(selected)
    print(f" Selected checks: {selected_checks}")
    return selected_checks

def evaluate_answer(check_type, context, question, answer, model):
    print(f" Evaluating answer with check: {check_type}")
    prompt = {
        "inputs": f"Check: {check_type}.\nContext: {context}\nQuestion: {question}\nAnswer: {answer}\nCritique or say 'None'."
    }

    for attempt in range(3):
        try:
            result = cached_invoke(model, prompt).strip()
            print(f" Critique result: {result}")
            return result
        except Exception as e:
            print(f" Critique error: {e} (Attempt {attempt + 1})")
            if "rate limit" in str(e).lower():
                wait_time = 20
            else:
                wait_time = 2 ** attempt
            print(f" Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)

    return "[Critique failed after retries]"


def get_context(documents):
    print(f" Combining {len(documents)} documents into context")
    return "\n\n".join([doc.page_content for doc in documents])

def retrieve_docs(query, k=4):
    print(f" Retrieving documents for query: {query}")
    results = faiss_db.similarity_search_with_score(query, k=k)
    filtered = [doc for doc, score in results if score >= 0.65]
    print(f" Retrieved {len(filtered)} relevant documents")
    return filtered

def retry_logic(model, prompt, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            response = model.invoke({"inputs": prompt})
            if response:
                return response
        except Exception as e:
            print(f"❌ Error: {e} (Attempt {attempt + 1})")
            attempt += 1
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    return "[Failed after retries]"

def generate_answer(query, documents, model):
    print(" Generating initial answer...")
    context = get_context(documents)
    answer = model.invoke({"question": query, "context": context})
    print(f"Generated Answer: {answer}")
    return answer, context

def critique_and_correct_answer(initial_answer, context, question, model):
    print(" Critiquing and correcting answer...")
    prompt = f"""Improve the following legal answer using the context. Be concise (6-7 sentences max).
Context:
{context}
Question:
{question}
Initial Answer:
{initial_answer}
Final Answer:"""
    result = model.invoke({"inputs": prompt}).strip()
    print(f" Final corrected answer: {result}")
    return result


def self_correcting_query(query, documents, model1, model2):
    print(f"\n Self-correcting query: {query}")
    if not documents:
        print(" No documents found.")
        return "No relevant documents found to answer this question."

    initial_answer, context = generate_answer(query, documents, model1)
    query_type = classify_query_type(query, model2)
    checks = select_evaluation_checks(query_type)

    errors = []
    for check in checks:
        critique = evaluate_answer(check, context, query, initial_answer, model2)
        if critique.lower() != "none":
            errors.append((check, critique))

    corrected = critique_and_correct_answer(initial_answer, context, query, model2) if errors else initial_answer

   
    return corrected
'''
test_cases = [
    {"question": "Can a 15-year-old work 10-hour shifts in a hotel in Maharashtra without formal approval?"},
    {"question": "Compare a Private Limited Company and LLP in terms of liability and compliance under Indian law."},
    {"question": "Explain the process to file a case against a Resident Welfare Association (RWA) for discriminatory treatment based on religion."},
    {"question": "Is it ethical and legal for a lawyer in India to publicly disclose workplace harassment faced by a client?"},
    {"question": "Draft a contract clause that allows an Indian software freelancer to terminate the contract immediately if client misuses personal data."}
]

def run_test_cases(log_to_file=True):
    log_lines = []

    for i, case in enumerate(test_cases, 1):
        print(f"\n === Legal Test Case {i} ===")
        question = case["question"]
        documents = retrieve_docs(question)

        if not documents:
            result = f" No relevant documents for: {question}\n"
            print(result)
            log_lines.append(result)
            continue

        context = get_context(documents)
        initial_answer, _ = generate_answer(question, documents, llm_model)

        query_type = classify_query_type(question, critic_model)
        checks = select_evaluation_checks(query_type)

        errors = []
        for check in checks:
            critique = evaluate_answer(check, context, question, initial_answer, critic_model)
            if critique.lower() != "none":
                errors.append((check, critique))
            time.sleep(2.5)

        if errors:
            corrected_answer = critique_and_correct_answer(initial_answer, context, question, critic_model)
        else:
            corrected_answer = initial_answer

        # Collect detailed log
        log = [
            f"\n===  Legal Test Case {i} ===",
            f"Legal Question: {question}",
            f"Context: {' '.join(doc.page_content[:200] for doc in documents)}...",
            f" Initial Answer: {initial_answer}",
            "Detected Issues:" if errors else " No issues found.",
        ]
        for err_type, critique in errors:
            log.append(f"   [{err_type.upper()}] - {critique}")
        log.append(f" Final Answer: {corrected_answer}")
        log.append("=" * 60)
        log_text = "\n".join(log)
        print(log_text)
        log_lines.append(log_text)

    if log_to_file:
        with open("legal_self_correction_logs.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(log_lines))
            
        print("\n Logs saved to legal_self_correction_logs.txt")


if __name__ == "__main__":
    
    run_test_cases()
    
   test_model = GroqChatModel(model_name)
    prompt = {"inputs": "What is the capital of France?"}
    response = test_model.invoke(prompt)
    print(" Test response:", response)
'''
