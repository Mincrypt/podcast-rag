import os
from typing import List, Dict, Any

TEMPLATE = """You are a helpful assistant answering questions about podcast episodes.
Use the provided context to answer the user's question accurately.
If the information is not in the context, say so clearly.

Context: {context}

Question: {question}

Answer:"""


def format_context(hits: List[Dict[str, Any]]) -> str:
    lines = []
    for h in hits:
        ep = h["metadata"].get("episode_id", "episode")
        st = h["metadata"].get("start", 0.0)
        en = h["metadata"].get("end", 0.0)
        lines.append(f"[{ep} {st:.1f}-{en:.1f}] {h['document']}")
    return "\n".join(lines[:10])

def generate_answer(question: str, hits: List[Dict[str, Any]]) -> str:
    prompt = TEMPLATE.format(question=question, context=format_context(hits))

    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    try:
        if groq_key:
            from groq import Groq
            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        print("Groq failed:", e)

    try:
        if openai_key:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI failed:", e)

    bullets = []
    for h in hits[:5]:
        ep = h["metadata"].get("episode_id", "episode")
        st = h["metadata"].get("start", 0.0)
        bullets.append(f"- {ep} @ {st:.1f}s: {h['document'][:200]}...")
    if not bullets:
        return "I couldn't find relevant context in the indexed episodes."
    return "Based on the most relevant segments:\n" + "\n".join(bullets)
