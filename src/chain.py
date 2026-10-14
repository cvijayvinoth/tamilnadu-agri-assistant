"""LCEL chains: history-aware query rewriting + streaming grounded answers.

Pattern follows the Week2-Day5 chat.py streaming approach and the Week2-Day4
citation prompt, composed with LCEL and wrapped so the conversation survives
every Streamlit rerun until the page is refreshed.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY

SYSTEM_PROMPT = """You are the Tamil Nadu Agriculture Schemes Assistant, helping \
farmers, extension officers and officials understand government farm schemes.

You answer ONLY from the numbered context passages supplied below. Each passage \
is labelled with its file name and page number.

Rules:
1. Cite every factual claim with its passage number in square brackets, e.g. [1] \
or [2][4]. Never invent a citation number that is not in the context.
2. Lead with a direct answer in one or two sentences, then give the supporting \
detail as short bullets. Keep subsidy amounts, percentages, eligibility \
conditions and deadlines exactly as written in the source - never round or guess.
3. If passages from DIFFERENT files conflict, do not silently pick one. Add a \
"⚠️ Conflicting sources" section stating what each file says, with citations.
4. If the context does not contain the answer, say plainly that you could not \
find it in the indexed documents and suggest what to ingest or whom to contact. \
Do NOT fall back on outside knowledge.
5. If the user writes in Tamil, answer in Tamil; otherwise answer in English. \
Keep the language simple enough for a farmer to act on.
6. Scheme rules change. When an answer depends on a year or budget cycle, say \
which year the source refers to and advise confirming with the local \
Agricultural Extension Centre (உழவர் உதவியகம்)."""

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context passages:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
])

# Turns "what about the subsidy for that?" into a self-contained query, because
# a bare follow-up embeds terribly against the vector store.
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation so far and a follow-up question, rewrite the "
     "follow-up as a standalone search query that makes sense without the "
     "conversation. Resolve pronouns like 'it'/'that scheme' into the actual "
     "scheme name. Return ONLY the rewritten query, nothing else. If it is "
     "already standalone, return it unchanged."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def get_llm(streaming: bool = True, temperature: float = LLM_TEMPERATURE):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
        max_tokens=LLM_MAX_TOKENS,
        streaming=streaming,
    )


def build_answer_chain(streaming: bool = True):
    """context + question + chat_history -> streamed answer tokens."""
    return ANSWER_PROMPT | get_llm(streaming=streaming) | StrOutputParser()


def rewrite_query(question: str, chat_history: list) -> str:
    """Make a follow-up standalone. Falls back to the raw question on any error."""
    if not chat_history:
        return question
    try:
        chain = REWRITE_PROMPT | get_llm(streaming=False, temperature=0) | StrOutputParser()
        rewritten = chain.invoke({
            "question": question,
            "chat_history": chat_history[-6:],   # last 3 turns is plenty
        }).strip()
        return rewritten or question
    except Exception:
        return question


def stream_answer(question: str, context: str, chat_history: list):
    """Yield answer tokens as they arrive, for st.write_stream."""
    chain = build_answer_chain(streaming=True)
    for token in chain.stream({
        "question": question,
        "context": context,
        "chat_history": chat_history[-8:],
    }):
        yield token


def suggest_followups(question: str, answer: str) -> list[str]:
    """Three short follow-up chips. Best-effort: returns [] if anything fails."""
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Suggest exactly 3 short follow-up questions (max 10 words each) a "
             "Tamil Nadu farmer might ask next. One per line, no numbering, no "
             "bullets, no extra text."),
            ("human", "Question: {q}\n\nAnswer: {a}"),
        ])
        chain = prompt | get_llm(streaming=False, temperature=0.4) | StrOutputParser()
        raw = chain.invoke({"q": question, "a": answer[:1200]})
        lines = [
            ln.strip().lstrip("-•*0123456789. ").strip()
            for ln in raw.splitlines() if ln.strip()
        ]
        return [ln for ln in lines if len(ln) > 5][:3]
    except Exception:
        return []
