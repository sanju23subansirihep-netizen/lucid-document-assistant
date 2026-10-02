import os
import sys
import chromadb
from openai import OpenAI

# ---------- CONFIG ----------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
KEY_FILE   = os.path.join(BASE_DIR, "openai_key.txt")
DB_DIR     = os.path.join(BASE_DIR, "chroma_db")
COLLECTION = "gkhep_docs"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"     # cheap and fast; good enough for document Q&A
TOP_K       = 5                  # how many chunks to send to the model

# ---------- SETUP ----------
api_key = open(KEY_FILE, "r", encoding="utf-8").read().strip()
client  = OpenAI(api_key=api_key)

chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_collection(name=COLLECTION)

# ---------- SEARCH ----------
def find_relevant_chunks(question, k=TOP_K):
    # 1. Convert question to embedding
    q_emb = client.embeddings.create(
        model=EMBED_MODEL,
        input=[question],
    ).data[0].embedding

    # 2. Query the vector store
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
    )
    docs  = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return list(zip(docs, metas, dists))

# ---------- ANSWER ----------
def ask(question):
    hits = find_relevant_chunks(question)
    context = "\n\n---\n\n".join(
        f"[From: {m['source']}, chunk {m['chunk']}]\n{d}"
        for d, m, _ in hits
    )

    system_prompt = (
        "You are an assistant for the Ghunsa Khola Hydroelectric Project (GKHEP). "
        "Answer ONLY using the context below. If the answer is not in the context, "
        "say 'I could not find this in the provided documents.' "
        "Always cite the source file name(s) at the end of your answer."
    )

    user_prompt = f"""CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content, hits

# ---------- INTERACTIVE LOOP ----------
print("\n" + "="*60)
print("  GKHEP Document Q&A — Stage 3")
print("  Type your question and press Enter.")
print("  Type 'exit' to quit.")
print("="*60 + "\n")

while True:
    try:
        q = input("❓ Your question: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        break
    if not q:
        continue
    if q.lower() in ("exit", "quit", "q"):
        print("Bye.")
        break

    try:
        answer, hits = ask(q)
        print("\n💡 ANSWER:")
        print("-" * 60)
        print(answer)
        print("-" * 60)
        print("\n📎 Sources used:")
        for d, m, dist in hits:
            print(f"   • {m['source']}  (chunk {m['chunk']}, score {dist:.3f})")
        print()
    except Exception as e:
        print(f"\n❌ Error: {e}\n")