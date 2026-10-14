from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM, LLM_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = "You are a Tamil Nadu Agri Assistant. Answer query strictly using context. Max 3 bullets. Under 60 words."

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        input=[text.replace("\n", " ")],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM
    )
    return response.data[0].embedding

def generate_answer(query: str, retrieved_chunks: list[str]) -> str:
    context_block = "\n".join([f"- {chunk}" for chunk in retrieved_chunks])
    user_content = f"Context:\n{context_block}\n\nQuery: {query}"

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        temperature=0.1,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()
