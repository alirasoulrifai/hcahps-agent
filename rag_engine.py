import os
import json
import requests
import numpy as np
import pandas as pd
import faiss
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer

from config import (
    TEXT_PATH, SUMM_PATH, FACTS_INDEX_PATH, SUMM_INDEX_PATH,
    FACTS_IDS_PATH, SUMM_IDS_PATH, GROQ_API_KEY, GROQ_URL, DEFAULT_LLM,
    SYS, AGENT_SYS, CFR_RULES
)

# ===================== CACHED RESOURCES =====================

@st.cache_resource
def get_embedder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device
    )
    return model

@st.cache_resource
def load_data_and_indexes():
    if not os.path.exists(TEXT_PATH):
        raise FileNotFoundError(f"Missing {TEXT_PATH}")
    if not os.path.exists(SUMM_PATH):
        raise FileNotFoundError(f"Missing {SUMM_PATH}")

    text_df = pd.read_csv(TEXT_PATH).fillna("")
    summ_df = pd.read_csv(SUMM_PATH).fillna("")

    # Try to convert numeric-looking columns
    text_df = text_df.apply(pd.to_numeric, errors="ignore")
    summ_df = summ_df.apply(pd.to_numeric, errors="ignore")

    for p in [FACTS_INDEX_PATH, SUMM_INDEX_PATH, FACTS_IDS_PATH, SUMM_IDS_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing index file: {p}")

    facts_index = faiss.read_index(FACTS_INDEX_PATH)
    summ_index  = faiss.read_index(SUMM_INDEX_PATH)
    facts_ids   = np.load(FACTS_IDS_PATH)
    summ_ids    = np.load(SUMM_IDS_PATH)

    return text_df, summ_df, facts_index, summ_index, facts_ids, summ_ids

# ===================== RETRIEVAL LOGIC =====================

def route(q: str) -> str:
    ql = q.lower()
    numeric_keywords = [
        "percent", "percentage", "%", "score", "rate", "how many",
        "clean", "cleanliness", "quiet", "quietness",
        "communication", "responsiveness",
        "discharge", "transition", "overall", "satisfaction",
        "always", "usually", "sometimes", "never"
    ]
    interpretive_keywords = [
        "why", "how can", "improve", "recommend", "suggest",
        "insight", "root cause"
    ]
    
    numeric_score = sum(k in ql for k in numeric_keywords)
    interpretive_score = sum(k in ql for k in interpretive_keywords)

    if numeric_score > 0:
        return "facts"
    if interpretive_score > 0:
        return "summaries"
    return "both"

def search(query: str, which: str = "facts", k: int = 5):
    embedder = get_embedder()
    text_df, summ_df, facts_index, summ_index, facts_ids, summ_ids = load_data_and_indexes()

    vec = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    if which == "facts":
        scores, idxs = facts_index.search(vec, k)
        rows = facts_ids[idxs[0]]
        pool = text_df
        col  = "text"
    else:
        scores, idxs = summ_index.search(vec, k)
        rows = summ_ids[idxs[0]]
        pool = summ_df
        col  = "summary_text"

    results = []
    for s, r in zip(scores[0], rows):
        # r might be an index in the original dataframe
        if r < len(pool):
            row = pool.iloc[int(r)]
            results.append({
                "score": float(s),
                "row": int(r),
                "Facility Name": row.get("Facility Name", ""),
                "State": row.get("State", ""),
                "measure_id": row.get("measure_id", ""),
                "measure_title": row.get("measure_title", ""),
                "snippet": str(row[col])[:300].replace("\n", " "),
                "source": which
            })
    return results

def search_both(query: str, k_each: int = 5):
    res_f = search(query, which="facts", k=k_each)
    res_s = search(query, which="summaries", k=k_each)
    merged = res_f + res_s
    merged_sorted = sorted(merged, key=lambda x: -x["score"])
    return merged_sorted

def decompose_query(question: str, model: str = DEFAULT_LLM) -> list[str]:
    """
    Break a complex question into simple search queries.
    """
    prompt = f"""Break this question into 2-3 simple search queries for an HCAHPS database.
    Question: "{question}"
    Output ONLY the queries, one per line. No numbering or bullets.
    """
    try:
        # Use valid non-streaming call for this internal logic
        raw = ask_ollama_str(prompt, model=model, temperature=0.0)
        queries = [line.strip() for line in raw.split("\n") if line.strip()]
        return queries[:3] # Limit to top 3
    except Exception:
        return [question]

def retrieve_for_question(q: str, k_each: int = 6, smart: bool = False, model: str = DEFAULT_LLM):
    if smart:
        queries = decompose_query(q, model=model)
    else:
        queries = [q]

    all_hits = []
    seen_rows = set()
    
    # We'll use a slightly smaller K if we have multiple queries to avoid context overflow
    k_adjusted = k_each if len(queries) == 1 else max(2, k_each // 2)

    for query in queries:
        r = route(query)
        if r == "both":
            hits = search_both(query, k_each=k_adjusted)
        else:
            hits = search(query, which=r, k=k_adjusted)
        
        for h in hits:
            # Deduplicate by (source, row_index)
            key = (h["source"], h["row"])
            if key not in seen_rows:
                seen_rows.add(key)
                all_hits.append(h)

    # Re-rank might be good here, but for now just sort by existing scores
    # (Note: scores are from different queries, so not perfectly comparable, but okay for rough sort)
    all_hits.sort(key=lambda x: x["score"], reverse=True)
    
    return "smart" if smart else r, all_hits, queries

# ===================== LLM HELPERS =====================

from typing import Iterator

def ask_groq(prompt: str, model: str = DEFAULT_LLM, temperature: float = 0.1) -> Iterator[str]:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it in Streamlit Cloud secrets.")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": True
    }

    with requests.post(GROQ_URL, headers=headers, json=payload, stream=True, timeout=180) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                if line_str.startswith("data: "):
                    data = line_str[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError):
                        pass

def ask_groq_str(prompt: str, model: str = DEFAULT_LLM, temperature: float = 0.1) -> str:
    """Non-streaming version for internal logic (decomposition, etc.)"""
    return "".join(list(ask_groq(prompt, model=model, temperature=temperature)))

# Keep old names as aliases for compatibility
ask_ollama = ask_groq
ask_ollama_str = ask_groq_str

def ellipsize(s, n=450):
    s = " ".join(str(s).split())
    return (s[:n] + " …") if len(s) > n else s

def make_citation(row: dict) -> str:
    st_ = row.get("State", "")
    mid = row.get("measure_id", "")
    fac = row.get("Facility Name", "")
    return f"[{st_} | {mid}] {fac}"

def build_context_from_hits(hits, top_k=5):
    lines = []
    for h in hits[:top_k]:
        snippet = h.get("snippet") or ""
        lines.append(f"- {ellipsize(snippet)}  ⟨{make_citation(h)}⟩")
    return "\n".join(lines)

def make_prompt(question: str, context_block: str, history: list = None) -> str:
    # History handling can be added here
    hist_str = ""
    if history:
        # Take last few turns
        recent = history[-4:] 
        hist_str = "\nConversation History:\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent]) + "\n"
        
    return f"""{SYS}

{hist_str}
Question: {question}

Context:
{context_block}
"""

def rag_answer(question: str, k_each: int = 6, model: str = DEFAULT_LLM, temperature: float = 0.1, history: list = None, smart: bool = False) -> dict:
    
    # Unpack the new 3-element tuple
    route_used, hits, queries_used = retrieve_for_question(question, k_each=k_each, smart=smart, model=model)
    
    ctx = build_context_from_hits(hits, top_k=min(10, len(hits))) # Increased context window slightly
    prompt = make_prompt(question, ctx, history)
    
    # Return generator directly
    stream = ask_ollama(prompt, model=model, temperature=temperature)
    
    return {
        "mode": "rag",
        "route": route_used,
        "prompt": prompt,
        "context_preview": ctx,
        "stream": stream, 
        "hits": hits,
        "queries": queries_used
    }

def agent_decide(user_msg: str, model: str = DEFAULT_LLM) -> dict:
    decision_prompt = f"{AGENT_SYS}\n\nUser: {user_msg}\nYour decision:"
    # Decision step needs full response, not stream, so we consume the generator here
    stream = ask_ollama(decision_prompt, model=model, temperature=0.0)
    raw = "".join(list(stream))
    try:
        obj = json.loads(raw)
        return obj
    except json.JSONDecodeError:
        return {"tool": "rag", "args": {"question": user_msg}}

def agent_answer(user_msg: str, k_each: int = 6, model: str = DEFAULT_LLM, rag_temperature: float = 0.1, history: list = None, smart: bool = False) -> dict:
    decision = agent_decide(user_msg, model=model)
    tool = decision.get("tool", "rag")

    if tool == "rag":
        q = decision.get("args", {}).get("question", user_msg)
        return rag_answer(q, k_each=k_each, model=model, temperature=rag_temperature, history=history, smart=smart)
    elif tool == "dashboard":
        # Pass the args back to the UI handler
        return {
            "mode": "dashboard_control",
            "args": decision.get("args", {})
        }
    elif tool == "code_interpreter":
        return {
            "mode": "code_interpreter",
            "args": decision.get("args", {})
        }
    elif tool == "chitchat":
        prompt = SYS + f"\n\nUser question (no tools needed):\n{user_msg}"
        stream = ask_ollama(prompt, model=model, temperature=0.5)
        return {
            "mode": "chitchat",
            "route": None,
            "prompt": prompt,
            "context_preview": "",
            "stream": stream,
            "hits": []
        }
    
    # fallback
    return rag_answer(user_msg, k_each=k_each, model=model, temperature=rag_temperature, history=history)

def chat_answer(user_msg: str, model: str = DEFAULT_LLM, temperature: float = 0.5) -> dict:
    prompt = SYS + f"\n\nUser question (no data retrieval):\n{user_msg}"
    stream = ask_ollama(prompt, model=model, temperature=temperature)
    return {
        "mode": "chitchat",
        "route": None,
        "prompt": prompt,
        "context_preview": "",
        "stream": stream,
        "hits": []
    }

def detect_applicable_cfr_sections(question: str, hits: list[dict]):
    """
    Detect which CFR sections apply based on:
    - user's question
    - retrieved measure titles/snippets
    """
    text = (question or "").upper()

    for h in hits:
        text += " " + str(h.get("measure_title", "")).upper()
        text += " " + str(h.get("snippet", "")).upper()

    matched = []
    for rule in CFR_RULES:
        if any(keyword in text for keyword in rule["keywords"]):
            matched.append(rule)

    return matched

def generate_research_plan(topic: str, model: str = DEFAULT_LLM) -> list[str]:
    prompt = f"""You are a research planner. Create a 3-step research plan to answer: "{topic}".
    Return ONLY 3 short, distinct search queries, one per line.
    """
    try:
        raw = ask_ollama_str(prompt, model=model, temperature=0.3)
        steps = [line.strip().lstrip("- ").lstrip("123. ") for line in raw.split("\n") if line.strip()]
        return steps[:3]
    except Exception:
        return [topic]

def research_answer(topic: str, k_each: int = 6, model: str = DEFAULT_LLM) -> dict:
    """
    Multi-step research: Plan -> Execute -> Synthesize
    """
    # 1. Plan
    plan = generate_research_plan(topic, model=model)
    
    # 2. Execute
    all_hits = []
    seen_rows = set()
    
    # We yield progress updates via a generator, but since we need to return a dict structure
    # compatible with the rest of the app, we'll do the execution first, then return the stream.
    # Ideally, we'd refactor to yield progress events, but for now we'll collect hits first.
    
    for step in plan:
        # Use decompose? Or just search? 
        # Let's keep it simple and just do a search_both for coverage
        step_queries = decompose_query(step, model=model)
        for q in step_queries:
            r = route(q)
            if r == "both":
                hits = search_both(q, k_each=4)
            else:
                hits = search(q, which=r, k=4)
                
            for h in hits:
                key = (h["source"], h["row"])
                if key not in seen_rows:
                    seen_rows.add(key)
                    all_hits.append(h)
    
    # 3. Synthesize
    ctx = build_context_from_hits(all_hits, top_k=15) # Larger context for report
    
    prompt = f"""{SYS}
    
    TASK: Write a comprehensive research report on: "{topic}".
    Use the provided context. Cite sources where possible.
    Structure the report with H2 headers (##).
    
    Context:
    {ctx}
    """
    
    stream = ask_ollama(prompt, model=model, temperature=0.3)
    
    return {
        "mode": "research",
        "plan": plan,
        "hits": all_hits,
        "context_preview": ctx,
        "stream": stream,
        "prompt": prompt
    }
