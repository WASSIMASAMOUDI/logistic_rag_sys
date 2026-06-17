#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    TruckMind — Pipeline RAG Logistique (Version Intégrée)
    ======================================================
    Agent LangGraph complet :
    Router → SQL → [ChromaDB] → Analyser → LLM (Groq / Qwen3-32b)
"""

import os
import re
import json
import time
import sqlite3
from pathlib import Path
from typing import Tuple, List, Literal, TypedDict

import chromadb
from groq import Groq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer

# ═══════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

load_dotenv(Path(__file__).parent.parent / ".env")

BASE_DIR = Path(__file__).parent.parent

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("⚠️  Attention: GROQ_API_KEY non définie dans .env")

CHEMIN_BD    = os.getenv("BD_TEST",           str(BASE_DIR / "test/knowledge/truck_logistic.db"))
CHROMA_DIR   = os.getenv("CHROMA_DB_PATH",    str(BASE_DIR / "test/data/chroma_db_logistique"))
QA_INPUT_FILE= os.getenv("QA_DATASET_PATH",   str(BASE_DIR / "test/evaluation/qa_dataset.json"))
OUTPUT_FILE  = os.getenv("RESULTS_PATH",      str(BASE_DIR / "test/evaluation/questionResults.json"))

PDF_FILES = [
    os.getenv("DYNAFLEET_PLAINTEXT_PDF", str(BASE_DIR / "uploads/Dynafleet_plaintext.pdf")),
    os.getenv("VOLVO_LOGISTIQUE_PDF",    str(BASE_DIR / "uploads/volvo_logistique.pdf")),
    os.getenv("ADR_2025_LOGISTIQUE_PDF", str(BASE_DIR / "uploads/ADR_2025_Logistique.pdf")),
]

EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MODELE        = "qwen/qwen3-32b"
TAILLE_CHUNK  = 250
CHEVAUCHEMENT = 63
BATCH_SIZE    = 64
TOP_K         = 5

# ═══════════════════════════════════════════════════════════════════
# 2. PIPELINE CHROMADB
# ═══════════════════════════════════════════════════════════════════

pipeline: dict = {"embed_obj": None, "index_obj": None, "ready": False}


def _get_prefix(filename: str) -> str:
    base = re.sub(r'[^a-zA-Z0-9]', '_', os.path.splitext(filename)[0]).lower()
    if "dynafleet" in base: return "dynafleet"
    if "volvo"     in base: return "volvo"
    if "manuel"    in base: return "manuel"
    if "adr"       in base: return "adr"
    return base


def _charger_pdf(pdf_path: str) -> list:
    if not os.path.exists(pdf_path):
        print(f"  ❌ Introuvable : {pdf_path}")
        return []
    try:
        docs = PyPDFLoader(pdf_path).load()
        for d in docs:
            d.metadata["source_file"] = os.path.basename(pdf_path)
        print(f"  ✅ {len(docs)} page(s) ← {os.path.basename(pdf_path)}")
        return docs
    except Exception as e:
        print(f"  ❌ Erreur chargement PDF : {e}")
        return []


def _charger_modele_embedding() -> HuggingFaceEmbeddings:
    print(f"🔄 Chargement embedding : {EMBED_MODEL}")
    _st_model = SentenceTransformer(EMBED_MODEL, local_files_only=True)
    print(f"   seq_length par défaut : {_st_model.max_seq_length}")
    _st_model.max_seq_length = 512
    print(f"   seq_length appliqué  : {_st_model.max_seq_length}")
    del _st_model

    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    if hasattr(emb.client, "max_seq_length"):
        emb.client.max_seq_length = 512
    print("✅ Modèle embedding prêt (max_seq=512)")
    return emb


def init_pipeline(force_reindex: bool = False, pdf_list: list = None):
    global pipeline

    if pdf_list is None:
        pdf_list = PDF_FILES

    valid_files = [f for f in pdf_list if os.path.exists(f)]

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if not force_reindex:
        try:
            col   = client.get_collection("truck_rag_logistique")
            count = col.count()
            if count > 0:
                emb = _charger_modele_embedding()
                pipeline.update(embed_obj=emb, index_obj=col, ready=True)
                print(f"✅ Pipeline rechargé ({count} chunks existants)")
                return
        except Exception:
            pass

    if not valid_files:
        print("❌ Aucun PDF valide — ChromaDB désactivé")
        return

    print(f"\n📂 Indexation de {len(valid_files)} PDF(s)...")
    all_docs = []
    for f in valid_files:
        all_docs.extend(_charger_pdf(f))

    if not all_docs:
        print("❌ Aucune page chargée")
        return

    # ── Splitter SANS connexion internet ─────────────────────────
    splitter = SentenceTransformersTokenTextSplitter(
        model_name=EMBED_MODEL,
        chunk_size=TAILLE_CHUNK,
        chunk_overlap=CHEVAUCHEMENT,
        model_kwargs={"local_files_only": True},
    )
    chunks = splitter.split_documents(all_docs)
    print(f"✂️  {len(all_docs)} pages → {len(chunks)} chunks "
          f"(taille={TAILLE_CHUNK}, chevauchement={CHEVAUCHEMENT})")

    ids_list, metadata_list, texts_list = [], [], []
    compteurs: dict = {}
    for chunk in chunks:
        source = chunk.metadata.get("source_file", "unknown")
        prefix = _get_prefix(source)
        compteurs[prefix] = compteurs.get(prefix, 0) + 1
        chunk_id = f"{prefix}_{compteurs[prefix]:04d}"
        meta = {**chunk.metadata, "chunk_id": chunk_id}
        ids_list.append(chunk_id)
        metadata_list.append(meta)
        texts_list.append(chunk.page_content)

    emb = _charger_modele_embedding()
    print(f"🧮 Calcul embeddings ({len(texts_list)} chunks, batch={BATCH_SIZE})...")
    all_embeddings = []
    nb_batches = (len(texts_list) - 1) // BATCH_SIZE + 1
    for i in range(0, len(texts_list), BATCH_SIZE):
        batch = texts_list[i:i + BATCH_SIZE]
        print(f"   Batch {i // BATCH_SIZE + 1}/{nb_batches}")
        all_embeddings.extend(emb.embed_documents(batch))

    try:
        client.delete_collection("truck_rag_logistique")
    except Exception:
        pass
    col = client.create_collection("truck_rag_logistique")

    for i in range(0, len(chunks), BATCH_SIZE):
        sl = slice(i, i + BATCH_SIZE)
        col.add(
            embeddings=all_embeddings[sl],
            documents=texts_list[sl],
            metadatas=metadata_list[sl],
            ids=ids_list[sl],
        )

    pipeline.update(embed_obj=emb, index_obj=col, ready=True)
    print(f"\n✅ Pipeline prêt — {len(chunks)} chunks indexés")


# ═══════════════════════════════════════════════════════════════════
# 3. RECHERCHE CHROMADB
# ═══════════════════════════════════════════════════════════════════

def rechercher_dans_chroma(question: str, top_k: int = TOP_K) -> Tuple[str, int]:
    if not pipeline["ready"]:
        print("⚠️  Pipeline non initialisé — lancement...")
        init_pipeline(force_reindex=False)
        if not pipeline["ready"]:
            return "", 0

    vecteur   = pipeline["embed_obj"].embed_query(question)
    resultats = pipeline["index_obj"].query(
        query_embeddings=[vecteur],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents  = resultats["documents"][0] if resultats["documents"] else []
    texte_final = "\n\n---\n\n".join(documents)
    return texte_final, len(documents)


# ═══════════════════════════════════════════════════════════════════
# 4. RECHERCHE SQL
# ═══════════════════════════════════════════════════════════════════

MOTS_STATS = [
    "combien", "nombre", "total", "taux", "pourcentage", "%",
    "moyenne", "moyen", "maximum", "minimum", "max", "min",
    "étendue", "distribution", "répartition", "fréquence",
    "proportion", "statistique", "count", "avg", "sum",
]
MOTS_ROUTE = [
    "route", "trajet", "itinéraire", "distance", "km", "kilomètre",
    "départ", "destination", "arrivée", "durée", "péage", "autoroute",
    "nationale", "urbain", "restriction", "casablanca", "rabat", "fes",
    "marrakech", "tanger", "agadir", "oujda", "meknes", "tetouan",
]
MOTS_CHAUFFEUR = [
    "chauffeur", "conducteur", "driver", "permis", "infraction",
    "heures", "repos", "fatigue", "nom",
]
MOTS_EXPEDITION = [
    "expédition", "livraison", "shipment", "colis", "marchandise",
    "poids", "statut", "retard", "en_transit", "livré", "annulé",
    "date", "véhicule", "camion",
]
MOTS_ALERTES = [
    "retard", "problème", "anomalie", "rouge", "critique",
    "alerte", "warning", "infractions", "fatigue", "risque",
]


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(CHEMIN_BD)


def extraire_shipment_id(texte: str) -> List[str]:
    return [i.upper() for i in re.findall(r'\bSHP-\d{10}\b', texte, re.IGNORECASE)]

def extraire_driver_id(texte: str) -> List[str]:
    return [i.upper() for i in re.findall(r'\bDRV-\d{4}\b', texte, re.IGNORECASE)]

def extraire_vehicule_id(texte: str) -> List[str]:
    return [i.upper() for i in re.findall(r'\bV\d+_\d+\b', texte, re.IGNORECASE)]

def extraire_route_id(texte: str) -> List[str]:
    return [i.upper() for i in re.findall(r'\bRT-[A-Z]{3}-[A-Z]{3}-\d{3}\b', texte, re.IGNORECASE)]


def _contient(question: str, mots: List[str]) -> bool:
    q = question.lower()
    return any(m in q for m in mots)


def _get_fleet_stats(cur) -> str:
    cur.execute("SELECT COUNT(*) FROM shipments")
    total = cur.fetchone()[0]
    cur.execute("SELECT statut, COUNT(*) FROM shipments GROUP BY statut ORDER BY 2 DESC")
    statuts = " | ".join(f"{r[0]}: {r[1]} ({round(r[1]/total*100,1)}%)" for r in cur.fetchall())
    cur.execute("SELECT ROUND(AVG(poids_kg),2), ROUND(MIN(poids_kg),2), ROUND(MAX(poids_kg),2) FROM shipments")
    avg_p, min_p, max_p = cur.fetchone()
    cur.execute("SELECT type_marchandise, COUNT(*) FROM shipments GROUP BY type_marchandise ORDER BY 2 DESC LIMIT 5")
    types = " | ".join(f"{r[0]}: {r[1]}" for r in cur.fetchall())
    cur.execute("SELECT COUNT(*) FROM drivers")
    nb_d = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM routes")
    nb_r = cur.fetchone()[0]
    return (
        f"### FLEET_STATS ({total} expéditions)\n"
        f"  🚛 {nb_d} chauffeurs | {nb_r} routes\n"
        f"  📦 Statuts   : {statuts}\n"
        f"  ⚖️  Poids (kg): Moy={avg_p} | Min={min_p} | Max={max_p}\n"
        f"  📦 Top 5 marchandises : {types}\n"
        f"### FIN FLEET_STATS"
    )


def _stats_detaillees(cur) -> List[str]:
    lignes = []
    cur.execute("""
        SELECT r.depart, r.destination, COUNT(s.id),
               ROUND(AVG(s.poids_kg),0),
               SUM(CASE WHEN s.statut='retard' THEN 1 ELSE 0 END)
        FROM shipments s JOIN routes r ON s.route_id=r.id
        GROUP BY s.route_id ORDER BY 3 DESC LIMIT 5
    """)
    for r in cur.fetchall():
        lignes.append(f"[STATS_ROUTE] {r[0]} → {r[1]} : {r[2]} expéditions | Poids moy: {r[3]} kg | Retards: {r[4]}")
    cur.execute("""
        SELECT d.nom, COUNT(s.id),
               SUM(CASE WHEN s.statut='retard' THEN 1 ELSE 0 END)
        FROM shipments s JOIN drivers d ON s.chauffeur_id=d.id
        GROUP BY s.chauffeur_id HAVING COUNT(s.id)>5
        ORDER BY 3 DESC LIMIT 5
    """)
    for r in cur.fetchall():
        taux = round(r[2]/r[1]*100, 1) if r[1] else 0
        lignes.append(f"[STATS_DRIVER] {r[0]} : {r[1]} missions | Retards: {r[2]} ({taux}%)")
    return lignes


def _rechercher_routes(cur, question: str, ids: List[str], top_k: int) -> List[str]:
    def fmt(r):
        return (f"[Route {r[0]}] {r[1]} → {r[2]} | {r[3]} km | "
                f"Durée: {r[4]}h | Type: {r[5]} | Péages: {r[6]} MAD | Restrictions: {r[7]}")
    if ids:
        cur.execute("SELECT id,depart,destination,distance_km,duree_estimee_h,type_route,peages,restrictions FROM routes WHERE id=?", (ids[0],))
        return [fmt(r) for r in cur.fetchall()]
    q = question.lower()
    cur.execute("SELECT DISTINCT depart FROM routes")
    villes = [row[0].lower() for row in cur.fetchall()]
    trouvees = [v for v in villes if v in q]
    if trouvees:
        rows = []
        for v in trouvees[:2]:
            cur.execute("SELECT id,depart,destination,distance_km,duree_estimee_h,type_route,peages,restrictions FROM routes WHERE LOWER(depart)=? OR LOWER(destination)=? LIMIT ?", (v, v, top_k))
            rows.extend(cur.fetchall())
        return [fmt(r) for r in rows]
    cur.execute("SELECT id,depart,destination,distance_km,duree_estimee_h,type_route,peages,restrictions FROM routes LIMIT ?", (top_k,))
    return [fmt(r) for r in cur.fetchall()]


def _rechercher_chauffeurs(cur, question: str, ids: List[str], top_k: int) -> List[str]:
    def fmt(r):
        repos = "🔴 FATIGUÉ" if r[4] == 0 else ("🟡 ATTENTION" if r[4] < 4 else "🟢 OK")
        infractions = "🔴 ÉLEVÉ" if r[5] >= 3 else ("🟡 MOYEN" if r[5] >= 1 else "🟢 AUCUNE")
        return (f"[Chauffeur {r[0]}] {r[1]} | Permis: {r[2]} | "
                f"Heures conduite: {r[3]}h | Repos restant: {r[4]}h {repos} | Infractions: {r[5]} {infractions}")
    if ids:
        cur.execute("SELECT * FROM drivers WHERE id=?", (ids[0],))
        return [fmt(r) for r in cur.fetchall()]
    q = question.lower()
    cur.execute("SELECT id, nom FROM drivers")
    tous = cur.fetchall()
    trouves = [row for row in tous if any(p in q for p in row[1].lower().split())]
    if trouves:
        result = []
        for row in trouves[:2]:
            cur.execute("SELECT * FROM drivers WHERE id=?", (row[0],))
            result.extend([fmt(r) for r in cur.fetchall()])
        return result
    cur.execute("SELECT * FROM drivers WHERE infractions>=2 OR repos_restant_h=0 ORDER BY infractions DESC, heures_conduite DESC LIMIT ?", (top_k,))
    return [fmt(r) for r in cur.fetchall()]


def _rechercher_expeditions(cur, question: str, shp_ids: List[str], veh_ids: List[str], top_k: int) -> List[str]:
    BASE = """
        SELECT s.id, s.vehicule_id, r.depart, r.destination,
               s.date_depart, s.date_arrivee, s.poids_kg,
               s.type_marchandise, s.statut, d.nom
        FROM   shipments s
        LEFT JOIN routes  r ON s.route_id    = r.id
        LEFT JOIN drivers d ON s.chauffeur_id = d.id
    """
    def fmt(r):
        emoji = "✅" if r[8] == "livre" else ("⏰" if r[8] == "retard" else "🚚")
        return (f"[Expédition {r[0]}] {emoji} {r[8].upper()} | Véhicule: {r[1]} | "
                f"{r[2] or '?'} → {r[3] or '?'} | Départ: {r[4]} → Arrivée: {r[5]} | "
                f"Poids: {r[6]} kg | {r[7]} | Chauffeur: {r[9] or 'N/A'}")
    if shp_ids:
        cur.execute(BASE + " WHERE s.id=?", (shp_ids[0],))
        return [fmt(r) for r in cur.fetchall()]
    if veh_ids:
        cur.execute(BASE + " WHERE s.vehicule_id=? ORDER BY s.date_depart DESC LIMIT ?", (veh_ids[0], top_k))
        return [fmt(r) for r in cur.fetchall()]
    q = question.lower()
    if "retard" in q:
        cur.execute(BASE + " WHERE s.statut='retard' ORDER BY s.date_depart DESC LIMIT ?", (top_k,))
    elif "livr" in q:
        cur.execute(BASE + " WHERE s.statut='livre'  ORDER BY s.date_depart DESC LIMIT ?", (top_k,))
    else:
        cur.execute(BASE + " ORDER BY s.date_depart DESC LIMIT ?", (top_k,))
    return [fmt(r) for r in cur.fetchall()]


def _rechercher_alertes_logistique(cur, top_k: int) -> List[str]:
    lignes = []
    cur.execute("""
        SELECT s.id, s.vehicule_id, r.depart, r.destination, s.date_arrivee, d.nom
        FROM   shipments s
        LEFT JOIN routes  r ON s.route_id    = r.id
        LEFT JOIN drivers d ON s.chauffeur_id = d.id
        WHERE  s.statut = 'retard'
        ORDER  BY s.date_arrivee DESC LIMIT ?
    """, (top_k,))
    for r in cur.fetchall():
        lignes.append(f"[🔴 RETARD] Expédition {r[0]} | Véhicule {r[1]} | {r[2] or '?'} → {r[3] or '?'} | Arrivée: {r[4]} | Chauffeur: {r[5] or 'N/A'}")
    cur.execute("""
        SELECT id, nom, infractions, repos_restant_h, heures_conduite
        FROM   drivers
        WHERE  infractions >= 3 OR repos_restant_h = 0
        ORDER  BY infractions DESC LIMIT ?
    """, (top_k,))
    for r in cur.fetchall():
        lignes.append(f"[🟠 CHAUFFEUR_RISQUE] {r[0]} — {r[1]} | Infractions: {r[2]} | Repos restant: {r[3]}h | Heures conduite: {r[4]}h")
    return lignes


def _fallback_general(cur, question: str, top_k: int) -> List[str]:
    q = question.lower()
    cur.execute("SELECT DISTINCT type_marchandise FROM shipments")
    types = [row[0].lower() for row in cur.fetchall()]
    type_trouve = next((t for t in types if t in q), None)
    if type_trouve:
        cur.execute("""
            SELECT s.id, s.vehicule_id, r.depart, r.destination, s.poids_kg, s.statut
            FROM shipments s LEFT JOIN routes r ON s.route_id=r.id
            WHERE LOWER(s.type_marchandise)=?
            ORDER BY s.date_depart DESC LIMIT ?
        """, (type_trouve, top_k))
        return [f"[{type_trouve.title()}] Expédition {r[0]} | Véhicule {r[1]} | {r[2] or '?'} → {r[3] or '?'} | Poids: {r[4]} kg | Statut: {r[5]}" for r in cur.fetchall()]
    return [_get_fleet_stats(cur)]


def rechercher_dans_sql(question: str, top_k: int = TOP_K) -> Tuple[str, int]:
    conn = _conn()
    cur  = conn.cursor()
    lignes: List[str] = []
    strategies: List[str] = []

    shp_ids = extraire_shipment_id(question)
    drv_ids = extraire_driver_id(question)
    veh_ids = extraire_vehicule_id(question)
    rte_ids = extraire_route_id(question)

    if _contient(question, MOTS_STATS):
        print("  📊 Stratégie A — Stats globales")
        lignes.append(_get_fleet_stats(cur))
        lignes += _stats_detaillees(cur)
        strategies.append("A")
    if _contient(question, MOTS_ROUTE) or rte_ids:
        print("  🗺️  Stratégie B — Routes")
        lignes += _rechercher_routes(cur, question, rte_ids, top_k)
        strategies.append("B")
    if _contient(question, MOTS_CHAUFFEUR) or drv_ids:
        print("  👤 Stratégie C — Chauffeurs")
        lignes += _rechercher_chauffeurs(cur, question, drv_ids, top_k)
        strategies.append("C")
    if _contient(question, MOTS_EXPEDITION) or shp_ids or veh_ids:
        print("  📦 Stratégie D — Expéditions")
        lignes += _rechercher_expeditions(cur, question, shp_ids, veh_ids, top_k)
        strategies.append("D")
    if _contient(question, MOTS_ALERTES):
        print("  🚨 Stratégie E — Alertes logistiques")
        lignes += _rechercher_alertes_logistique(cur, top_k)
        strategies.append("E")
    if not strategies:
        print("  🔄 Stratégie F — Fallback général")
        lignes += _fallback_general(cur, question, top_k)
        strategies.append("F")

    conn.close()
    vus, final = set(), []
    for l in lignes:
        if l not in vus:
            vus.add(l)
            final.append(l)

    print(f"  ✅ Stratégies : {strategies} | {len(final)} résultats")
    return "\n\n---\n\n".join(final), len(final)


# ═══════════════════════════════════════════════════════════════════
# 5. PROMPTS LLM
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
    You are TruckMind Logistics — an expert AI assistant for Volvo truck fleet management.

    ### Identity:
    - You have full knowledge of logistics operations: routes, shipments, drivers, deliveries.
    - You analyze freight data (weights, statuses, delays) and driver safety (hours, rest, infractions).
    - You cross-reference structured data (SQLite) with technical documentation (Dynafleet, Volvo manuals, ADR 2025).
    - Your tone is professional, precise, and logistics-native — speak like an experienced fleet operations manager.

    ### Knowledge Sources:
    1. SQLite / routes    — 14 routes between Moroccan cities
    2. SQLite / drivers   — 25 drivers
    3. SQLite / shipments — 9215 shipments
    4. ChromaDB / docs    — ADR 2025, Dynafleet, Volvo FH/FM

    ### Driver Safety Rules:
    - repos_restant_h = 0  → FATIGUÉ — immediate rest required
    - repos_restant_h < 4  → ATTENTION — monitor closely
    - infractions >= 3     → RISQUE ÉLEVÉ

    ### Rules:
    - Use ONLY the provided context.
    - NEVER say information non disponible unless ALL sources are empty.
    - Delays and driver fatigue are PRIORITY alerts.
    - For statistics: use exact values from FLEET_STATS block.

    ### Output Format:
    - Emergency alerts go FIRST
    - Use bullet points for lists (max 5 per section)
"""

MODELE_PROMPT_UTILISATEUR = """
    ### Contexte
    DONNÉES_SQL:
    {resultat_sql}

    EXTRAITS_DOCS:
    {resultat_vectoriel}

    ### Question
    {question}
"""


def _construire_prompt(etat: dict) -> dict:
    prompt = MODELE_PROMPT_UTILISATEUR.format(
        resultat_sql=etat.get("resultat_sql", "Aucune donnée SQL"),
        resultat_vectoriel=etat.get("resultat_vectoriel", "Aucun extrait technique"),
        question=etat.get("question", "Quel est l'état de la flotte ?"),
    )
    return {"prompt_utilisateur": prompt}


# ═══════════════════════════════════════════════════════════════════
# 6. AGENT LANGGRAPH
# ═══════════════════════════════════════════════════════════════════

class EtatDiagnostic(TypedDict):
    question:            str
    type_requete:        str
    besoin_vector:       bool
    resultat_sql:        str
    resultat_vectoriel:  str
    prompt_utilisateur:  str
    reponse_llm:         str


def noeud_router(etat: EtatDiagnostic) -> EtatDiagnostic:
    q = etat["question"].lower()
    if _contient(etat["question"], MOTS_STATS):
        type_req, besoin_vector = "stats", False
    elif _contient(etat["question"], MOTS_ROUTE) or extraire_route_id(etat["question"]):
        type_req, besoin_vector = "route", True
    elif _contient(etat["question"], MOTS_CHAUFFEUR) or extraire_driver_id(etat["question"]):
        type_req, besoin_vector = "chauffeur", True
    elif (_contient(etat["question"], MOTS_EXPEDITION) or extraire_shipment_id(etat["question"]) or extraire_vehicule_id(etat["question"])):
        type_req, besoin_vector = "expedition", False
    elif _contient(etat["question"], MOTS_ALERTES):
        type_req, besoin_vector = "alerte", False
    elif any(kw in q for kw in ["dynafleet", "tachymètre", "manuel", "volvo", "procédure", "km", "conduite", "telematics", "gps", "rapport"]):
        type_req, besoin_vector = "technique", True
    else:
        type_req, besoin_vector = "général", True
    print(f"  🔀 Router → type: '{type_req}' | vector: {besoin_vector}")
    return {**etat, "type_requete": type_req, "besoin_vector": besoin_vector}


def noeud_sql(etat: EtatDiagnostic) -> EtatDiagnostic:
    contexte, n = rechercher_dans_sql(etat["question"], TOP_K)
    print(f"  🗄️  SQL → {n} résultats")
    return {**etat, "resultat_sql": contexte or "Aucune donnée SQL pertinente."}


def noeud_vector(etat: EtatDiagnostic) -> EtatDiagnostic:
    contexte, n = rechercher_dans_chroma(etat["question"], TOP_K)
    print(f"  📦 ChromaDB → {n} chunks")
    return {**etat, "resultat_vectoriel": contexte or "Aucun extrait trouvé."}


def noeud_skip_vector(etat: EtatDiagnostic) -> EtatDiagnostic:
    print("  ⏭️  ChromaDB ignoré")
    return {**etat, "resultat_vectoriel": ""}


def condition_vector(etat: EtatDiagnostic) -> Literal["vector", "skip_vector"]:
    return "vector" if etat["besoin_vector"] else "skip_vector"


def noeud_analyser(etat: EtatDiagnostic) -> EtatDiagnostic:
    return {**etat, **_construire_prompt(etat)}


groq_client = Groq(api_key=GROQ_API_KEY)

def noeud_llm(etat: EtatDiagnostic) -> EtatDiagnostic:
    completion = groq_client.chat.completions.create(
        model=MODELE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": etat["prompt_utilisateur"]},
        ],
        temperature=0.0,
        max_completion_tokens=2048,
        top_p=0.95,
        reasoning_effort="default",
        stream=True,
        stop=None,
    )
    reponse = ""
    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            reponse += content
    print(f"  🤖 LLM → {len(reponse)} caractères")
    return {**etat, "reponse_llm": reponse.strip()}


def construire_graphe() -> StateGraph:
    graphe = StateGraph(EtatDiagnostic)
    graphe.add_node("router",      noeud_router)
    graphe.add_node("sql",         noeud_sql)
    graphe.add_node("vector",      noeud_vector)
    graphe.add_node("skip_vector", noeud_skip_vector)
    graphe.add_node("analyser",    noeud_analyser)
    graphe.add_node("llm",         noeud_llm)
    graphe.set_entry_point("router")
    graphe.add_edge("router", "sql")
    graphe.add_conditional_edges("sql", condition_vector, {"vector": "vector", "skip_vector": "skip_vector"})
    graphe.add_edge("vector",      "analyser")
    graphe.add_edge("skip_vector", "analyser")
    graphe.add_edge("analyser",    "llm")
    graphe.add_edge("llm",         END)
    return graphe.compile()


agent_logistic = construire_graphe()

def poser_question(question: str) -> str:
    etat_initial: EtatDiagnostic = {
        "question": question, "type_requete": "", "besoin_vector": False,
        "resultat_sql": "", "resultat_vectoriel": "", "prompt_utilisateur": "", "reponse_llm": "",
    }
    etat_final = agent_logistic.invoke(etat_initial)
    return etat_final["reponse_llm"]


# ═══════════════════════════════════════════════════════════════════
# 7. ÉVALUATION
# ═══════════════════════════════════════════════════════════════════

def charger_qa_dataset(chemin: str) -> list:
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"Fichier QA introuvable : {chemin}")
    with open(chemin, "r", encoding="utf-8") as f:
        data = json.load(f)
    qas = data.get("questions_answers", [])
    print(f"✅ QA Dataset chargé : {len(qas)} questions")
    return qas


def _sauvegarder(resultats: list, chemin: str):
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump({"dataset_name": "truck_llm_results_logistique", "results": resultats}, f, ensure_ascii=False, indent=2)


def generer_reponses_llm(qas: list, delai: float = 3.0, chemin_sauvegarde: str = None) -> list:
    resultats = []
    total = len(qas)
    for idx, qa in enumerate(qas, start=1):
        question   = qa.get("question", "")
        answer_ref = qa.get("answer", "")
        difficulty = qa.get("difficulty", "facile")
        source     = qa.get("source_file", "")
        print(f"\n[{idx:02d}/{total}] 🔍 {question[:80]}...")
        reponse_llm = ""
        type_requete = ""
        erreur = None
        for tentative in range(2):
            try:
                etat_initial: EtatDiagnostic = {
                    "question": question, "type_requete": "", "besoin_vector": False,
                    "resultat_sql": "", "resultat_vectoriel": "", "prompt_utilisateur": "", "reponse_llm": "",
                }
                etat_final   = agent_logistic.invoke(etat_initial)
                reponse_llm  = etat_final["reponse_llm"]
                type_requete = etat_final["type_requete"]
                erreur = None
                break
            except Exception as e:
                erreur = str(e)
                print(f"  ⚠️  Tentative {tentative + 1} échouée : {e}")
                if tentative == 0:
                    time.sleep(5)
        if erreur:
            reponse_llm  = f"ERREUR: {erreur}"
            type_requete = "erreur"
        resultats.append({
            "id": idx, "source_file": source, "question": question,
            "difficulty": difficulty, "type_requete": type_requete,
            "reponse_correcte": answer_ref, "reponse_llm": reponse_llm,
        })
        if chemin_sauvegarde:
            _sauvegarder(resultats, chemin_sauvegarde)
            print(f"  💾 Sauvegardé ({idx}/{total})")
        if idx < total:
            time.sleep(delai)
    print(f"\n✅ {len(resultats)} réponses générées.")
    return resultats


def sauvegarder_resultats(resultats: list, chemin: str):
    _sauvegarder(resultats, chemin)
    print(f"✅ Résultats sauvegardés → {chemin}")


# ═══════════════════════════════════════════════════════════════════
# 8. POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 TruckMind Logistique — Pipeline RAG Intégré")
    print("=" * 60)

    init_pipeline(force_reindex=False)

    tests = [
        ("Stats",     "Combien d'expéditions sont en retard ?"),
        ("Route",     "Quelles routes partent de Casablanca ?"),
        ("Chauffeur", "Quels chauffeurs ont des infractions ?"),
        ("Livraison", "Montre-moi les livraisons en retard"),
        ("Alerte",    "Y a-t-il des alertes ou problèmes ?"),
        ("Technique", "Comment fonctionne le système Dynafleet ?"),
    ]

    for label, question in tests:
        print(f"\n{'=' * 60}")
        print(f"TEST — {label} : {question}")
        rep = poser_question(question)
        print(rep[:400], "..." if len(rep) > 400 else "")

    if os.path.exists(QA_INPUT_FILE):
        print(f"\n{'=' * 60}")
        print("📊 Lancement de l'évaluation sur le dataset Q&A...")
        qas = charger_qa_dataset(QA_INPUT_FILE)
        resultats = generer_reponses_llm(qas, delai=15.0, chemin_sauvegarde=OUTPUT_FILE)
        sauvegarder_resultats(resultats, OUTPUT_FILE)
    else:
        print(f"\nℹ️  Pas de fichier QA trouvé ({QA_INPUT_FILE}) — évaluation ignorée")  