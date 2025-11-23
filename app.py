# app.py
# ---------------------------------------------------------
# Streamlit GraphRAG Chatbot für GRS-Stilllegung-Ontologie in Neo4j
#
# Lokal:
#   .env kann Neo4j-Creds + optional OPENAI_API_KEY enthalten.
# Cloud:
#   Neo4j-Creds in Streamlit Secrets,
#   OpenAI-Key wird vom User in der Sidebar eingegeben.
# ---------------------------------------------------------

import os
import json
import re
from typing import List, Dict, Any, Tuple

import streamlit as st
from neo4j import GraphDatabase
from openai import OpenAI

# Optional: nur für lokale Entwicklung
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ----------------------------
# Config Loader (Secrets -> ENV)
# ----------------------------
def get_cfg(key: str, default=None):
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)


NEO4J_URI      = get_cfg("NEO4J_URI")
NEO4J_USER     = get_cfg("NEO4J_USER")
NEO4J_PASSWORD = get_cfg("NEO4J_PASSWORD")
NEO4J_DATABASE = get_cfg("NEO4J_DATABASE", "neo4j")

OPENAI_MODEL   = get_cfg("OPENAI_MODEL", "gpt-4.1-mini")


# ----------------------------
# Cached Clients
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_driver(uri: str, user: str, password: str):
    return GraphDatabase.driver(uri, auth=(user, password))

@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str):
    return OpenAI(api_key=api_key)


# ----------------------------
# Neo4j Helper
# ----------------------------
def cypher_query(driver, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    params = params or {}
    with driver.session(database=NEO4J_DATABASE) as session:
        res = session.run(query, params)
        return [r.data() for r in res]


# ----------------------------
# LLM Helper
# ----------------------------
def llm_json(prompt: str) -> Any:
    """Call LLM and parse JSON output safely."""
    client = get_openai_client(st.session_state["OPENAI_API_KEY"])

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You output ONLY valid JSON. No extra text."},
            {"role": "user", "content": prompt},
        ],
    )
    txt = resp.choices[0].message.content.strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}|\[.*\]", txt, re.S)
        if not m:
            return None
        return json.loads(m.group(0))


def extract_entities(question: str) -> List[str]:
    prompt = f"""
Extrahiere aus der folgenden Frage die wichtigsten Ontologie-Begriffe (Subjekte/Objekte),
so wie sie in einem Neo4j-Graphen als :Concept.name vorkommen könnten.
Gib eine JSON-Liste von Strings aus, max. 8 Einträge, ohne Erklärungen.

Frage: {question}
"""
    data = llm_json(prompt)
    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]
    return []


# ----------------------------
# GraphRAG Retrieval
# ----------------------------
def seed_nodes(driver, entities: List[str], limit_per_entity: int = 5) -> List[str]:
    seeds = set()
    for e in entities:
        q = """
        MATCH (c:Concept)
        WHERE toLower(c.name) = toLower($e)
           OR toLower(c.name) CONTAINS toLower($e)
        RETURN c.name AS name
        LIMIT $lim
        """
        rows = cypher_query(driver, q, {"e": e, "lim": limit_per_entity})
        for r in rows:
            seeds.add(r["name"])
    return list(seeds)


def expand_subgraph(driver, seed_names: List[str], hops: int = 2, max_triples: int = 200):
    if not seed_names:
        return []

    q = f"""
    MATCH (s:Concept)-[r*1..{hops}]-(t:Concept)
    WHERE s.name IN $seeds
    WITH r
    UNWIND r AS rel
    WITH DISTINCT rel
    MATCH (a:Concept)-[rel]->(b:Concept)
    RETURN a.name AS s,
           type(rel) AS p,
           b.name AS o,
           rel.sourceDoc AS sourceDoc,
           rel.sourceSection AS sourceSection,
           rel.sourcePage AS sourcePage
    LIMIT $max_triples
    """
    return cypher_query(driver, q, {"seeds": seed_names, "max_triples": max_triples})


def score_triple(triple: Dict[str, Any], question: str) -> float:
    q_tokens = set(re.findall(r"\w+", question.lower()))
    text = f"{triple.get('s','')} {triple.get('p','')} {triple.get('o','')}".lower()
    t_tokens = set(re.findall(r"\w+", text))
    overlap = len(q_tokens & t_tokens)
    return overlap / (len(q_tokens) + 1e-9)


def rank_triples(triples: List[Dict[str, Any]], question: str, k: int = 25):
    scored = [(score_triple(t, question), t) for t in triples]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for s, t in scored[:k] if s > 0] or [t for s, t in scored[:k]]


def build_context(triples: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    lines = []
    sources = []
    for t in triples:
        s = t.get("s")
        p = t.get("p")
        o = t.get("o")
        sec = t.get("sourceSection")
        page = t.get("sourcePage")
        cite = f"{sec}, S.{page}"
        lines.append(f"- {s} —[{p}]→ {o}  (Quelle: {cite})")
        if cite not in sources:
            sources.append(cite)

    context = "\n".join(lines)
    return context, sources


# ----------------------------
# Answering
# ----------------------------
def answer_with_rag(question: str, context: str) -> str:
    client = get_openai_client(st.session_state["OPENAI_API_KEY"])

    system = (
        "Du bist ein Assistent für Stilllegung/Rückbau kerntechnischer Anlagen.\n"
        "Antworte NUR auf Basis des gegebenen Kontexts aus Neo4j.\n"
        "Wenn etwas nicht im Kontext steht, sage klar: 'Nicht im Dokument/Graph abgedeckt'.\n"
        "Nenne immer Quellen als Kapitel/Seite aus dem Kontext.\n"
    )
    user = f"""
KONTEXT (Tripel mit Quellen):
{context}

FRAGE:
{question}

AUFGABE:
- Gib eine präzise, strukturierte Antwort.
- Beziehe dich nur auf den Kontext.
- Füge am Ende 'Quellen:' mit Liste der verwendeten (Kapitel, Seite) hinzu.
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="GRS GraphRAG Chatbot", page_icon="🧠", layout="wide")
st.title("🧠 GRS-GraphRAG Chatbot (Neo4j)")

with st.sidebar:
    st.markdown("### OpenAI")
    default_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.get("OPENAI_API_KEY", default_key),
        help="Wird nur in dieser Session gespeichert."
    )
    if api_key:
        st.session_state["OPENAI_API_KEY"] = api_key
    else:
        st.warning("Bitte OpenAI API Key eingeben, sonst kann der Chatbot nicht antworten.")
        st.stop()

    st.markdown("---")
    st.markdown("### Neo4j Verbindung")
    if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
        st.error("Neo4j Zugangsdaten fehlen. Bitte in Streamlit Secrets/ENV setzen.")
        st.stop()

    st.write("Neo4j URI:", NEO4J_URI)
    st.write("Neo4j User:", NEO4J_USER)
    st.write("Neo4j DB:", NEO4J_DATABASE)
    st.write("LLM Model:", OPENAI_MODEL)

    st.markdown("---")
    hops = st.slider("Graph-Expansion Hops", 1, 3, 2)
    topk = st.slider("Top-K Tripel für Kontext", 5, 50, 25)

# Driver erstellen + kurzer Connection-Test
driver = get_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
try:
    cypher_query(driver, "RETURN 1 AS ok")
except Exception as e:
    st.error(f"Neo4j Verbindung fehlgeschlagen: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Stell eine Frage zur Stilllegung/Rückbau gemäß GRS…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("GraphRAG sucht im Neo4j-Graph…"):
            entities = extract_entities(question)
            seeds = seed_nodes(driver, entities)

            if not seeds:
                rough_entities = list({w for w in re.findall(r"\w+", question) if len(w) > 4})[:6]
                seeds = seed_nodes(driver, rough_entities)

            triples = expand_subgraph(driver, seeds, hops=hops)
            ranked = rank_triples(triples, question, k=topk)
            context, sources = build_context(ranked)
            answer = answer_with_rag(question, context)

        st.markdown(answer)
        with st.expander("🔎 Verwendeter Graph-Kontext"):
            st.markdown(context if context else "_Kein Kontext gefunden._")

    st.session_state.messages.append({"role": "assistant", "content": answer})
