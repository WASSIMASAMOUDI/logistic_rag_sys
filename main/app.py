#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smart Delivery System — Flask Backend (LangGraph Pipeline Intégré)"""

import os, re, json, sqlite3, time, sys
import random, string

# Configure console to support utf-8 emojis on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from typing import List, Tuple, Literal, TypedDict
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

# ── Chemins corrigés : tous relatifs à BASE_DIR ──────────────────
def _abs(rel: str) -> str:
    """Convertit un chemin relatif (depuis .env) en chemin absolu depuis BASE_DIR."""
    return str(BASE_DIR / rel)

CHEMIN_BD    = _abs(os.getenv("DB_PATH",          "main/knowledge/truck_logistic.db"))
CHROMA_DIR   = _abs(os.getenv("CHROMA_DB_PATH",   "main/data/chroma_db_logistique"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

PDF_FILES = [
    _abs(os.getenv("DYNAFLEET_PLAINTEXT_PDF", "uploads/Dynafleet_plaintext.pdf")),
    _abs(os.getenv("VOLVO_LOGISTIQUE_PDF",    "uploads/volvo_logistique.pdf")),
    _abs(os.getenv("ADR_2025_LOGISTIQUE_PDF", "uploads/ADR_2025_Logistique.pdf")),
]

EMBED_MODEL   = "all-MiniLM-L6-v2"
MODELE        = "qwen/qwen3-32b"
TAILLE_CHUNK  = 250   # ✅ FIX 5 : maintenant utilisé dans le splitter
CHEVAUCHEMENT = 63    # ✅ FIX 5 : maintenant utilisé dans le splitter
BATCH_SIZE    = 64
TOP_K         = 5

app = Flask(__name__, template_folder="templates", static_folder="static")

# ═══════════════════════════════════════════════════════════════════
# 1. DATABASE
# ═══════════════════════════════════════════════════════════════════

def get_db():
    conn = sqlite3.connect(CHEMIN_BD)
    conn.row_factory = sqlite3.Row
    return conn

def _conn():
    return sqlite3.connect(CHEMIN_BD)

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
        from langchain_community.document_loaders import PyPDFLoader
        docs = PyPDFLoader(pdf_path).load()
        for d in docs:
            d.metadata["source_file"] = os.path.basename(pdf_path)
        print(f"  ✅ {len(docs)} page(s) ← {os.path.basename(pdf_path)}")
        return docs
    except Exception as e:
        print(f"  ❌ Erreur chargement PDF : {e}")
        return []


def _charger_modele_embedding():
    from sentence_transformers import SentenceTransformer
    print(f"🔄 Chargement embedding : {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = 512

    class STEmbeddings:
        def __init__(self, m): self.model = m
        def embed_documents(self, texts): return self.model.encode(texts, normalize_embeddings=True).tolist()
        def embed_query(self, text): return self.model.encode(text, normalize_embeddings=True).tolist()

    print("✅ Modèle embedding prêt")
    return STEmbeddings(model)


def init_pipeline(force_reindex: bool = False, pdf_list: list = None):
    global pipeline
    import chromadb
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
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    all_docs = []
    for f in valid_files:
        all_docs.extend(_charger_pdf(f))
    if not all_docs:
        print("❌ Aucune page chargée"); return

    # ✅ FIX 5 : utilisation des constantes TAILLE_CHUNK et CHEVAUCHEMENT
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TAILLE_CHUNK,
        chunk_overlap=CHEVAUCHEMENT,
    )
    chunks = splitter.split_documents(all_docs)
    ids_list, metadata_list, texts_list = [], [], []
    compteurs: dict = {}
    for chunk in chunks:
        source = chunk.metadata.get("source_file", "unknown")
        prefix = _get_prefix(source)
        compteurs[prefix] = compteurs.get(prefix, 0) + 1
        chunk_id = f"{prefix}_{compteurs[prefix]:04d}"
        ids_list.append(chunk_id)
        metadata_list.append({**chunk.metadata, "chunk_id": chunk_id})
        texts_list.append(chunk.page_content)
    emb = _charger_modele_embedding()
    all_embeddings = []
    nb_batches = (len(texts_list) - 1) // BATCH_SIZE + 1
    for i in range(0, len(texts_list), BATCH_SIZE):
        print(f"   Batch {i // BATCH_SIZE + 1}/{nb_batches}")
        all_embeddings.extend(emb.embed_documents(texts_list[i:i + BATCH_SIZE]))
    try:
        client.delete_collection("truck_rag_logistique")
    except Exception:
        pass
    col = client.create_collection("truck_rag_logistique")
    for i in range(0, len(chunks), BATCH_SIZE):
        sl = slice(i, i + BATCH_SIZE)
        col.add(embeddings=all_embeddings[sl], documents=texts_list[sl],
                metadatas=metadata_list[sl], ids=ids_list[sl])
    pipeline.update(embed_obj=emb, index_obj=col, ready=True)
    print(f"\n✅ Pipeline prêt — {len(chunks)} chunks indexés")


# ═══════════════════════════════════════════════════════════════════
# 3. MOTS-CLÉS
# ═══════════════════════════════════════════════════════════════════

MOTS_STATS = [
    "combien","nombre","total","taux","pourcentage","%","moyenne","moyen",
    "maximum","minimum","max","min","étendue","distribution","répartition",
    "fréquence","proportion","statistique","count","avg","sum",
]
MOTS_ROUTE = [
    "route","trajet","itinéraire","distance","km","kilomètre","départ",
    "destination","arrivée","durée","péage","autoroute","nationale","urbain",
    "restriction","casablanca","rabat","fes","marrakech","tanger","agadir",
    "oujda","meknes","tetouan",
]
MOTS_CHAUFFEUR = [
    "chauffeur","conducteur","driver","permis","infraction","heures","repos","fatigue","nom",
]
MOTS_EXPEDITION = [
    "expédition","livraison","shipment","colis","marchandise","poids","statut",
    "retard","en_transit","livré","annulé","date","véhicule","camion",
]
MOTS_ALERTES = [
    "retard","problème","anomalie","rouge","critique","alerte","warning",
    "infractions","fatigue","risque",
]
MOTS_TECHNIQUE = [
    "dynafleet","tachymètre","manuel","volvo","procédure","conduite",
    "telematics","gps","rapport","adr","fh","fm",
]


def _contient(q: str, mots: List[str]) -> bool:
    return any(m in q.lower() for m in mots)

def extraire_shipment_id(t): return [i.upper() for i in re.findall(r'\bSHP-\d{10}\b', t, re.I)]
def extraire_driver_id(t):   return [i.upper() for i in re.findall(r'\bDRV-\d{4}\b',  t, re.I)]
def extraire_vehicule_id(t): return [i.upper() for i in re.findall(r'\bV\d+_\d+\b',   t, re.I)]
def extraire_route_id(t):    return [i.upper() for i in re.findall(r'\bRT-[A-Z]{3}-[A-Z]{3}-\d{3}\b', t, re.I)]

# ═══════════════════════════════════════════════════════════════════
# 4. HELPERS SQL
# ═══════════════════════════════════════════════════════════════════

def _get_fleet_stats(cur) -> str:
    cur.execute("SELECT COUNT(*) FROM shipments"); total = cur.fetchone()[0]
    cur.execute("SELECT statut, COUNT(*) FROM shipments GROUP BY statut ORDER BY 2 DESC")
    statuts = " | ".join(f"{r[0]}: {r[1]} ({round(r[1]/total*100,1)}%)" for r in cur.fetchall())
    cur.execute("SELECT ROUND(AVG(poids_kg),2), ROUND(MIN(poids_kg),2), ROUND(MAX(poids_kg),2) FROM shipments")
    avg_p, min_p, max_p = cur.fetchone()
    cur.execute("SELECT type_marchandise, COUNT(*) FROM shipments GROUP BY type_marchandise ORDER BY 2 DESC LIMIT 5")
    types = " | ".join(f"{r[0]}: {r[1]}" for r in cur.fetchall())
    cur.execute("SELECT COUNT(*) FROM drivers"); nb_d = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM routes");  nb_r = cur.fetchone()[0]
    return (f"### FLEET_STATS ({total} expéditions)\n"
            f"  🚛 {nb_d} chauffeurs | {nb_r} routes\n"
            f"  📦 Statuts: {statuts}\n"
            f"  ⚖️  Poids(kg): Moy={avg_p} | Min={min_p} | Max={max_p}\n"
            f"  📦 Top 5 marchandises: {types}\n"
            f"### FIN FLEET_STATS")


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
        inf   = "🔴 ÉLEVÉ"   if r[5] >= 3 else ("🟡 MOYEN"    if r[5] >= 1 else "🟢 AUCUNE")
        return (f"[Chauffeur {r[0]}] {r[1]} | Permis: {r[2]} | "
                f"Heures: {r[3]}h | Repos: {r[4]}h {repos} | Infractions: {r[5]} {inf}")
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
        LEFT JOIN routes  r ON s.route_id     = r.id
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


def _rechercher_alertes(cur, top_k: int) -> List[str]:
    lignes = []
    cur.execute("""
        SELECT s.id, s.vehicule_id, r.depart, r.destination, s.date_arrivee, d.nom
        FROM   shipments s
        LEFT JOIN routes  r ON s.route_id     = r.id
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
        lignes.append(f"[🟠 CHAUFFEUR_RISQUE] {r[0]} — {r[1]} | Infractions: {r[2]} | Repos: {r[3]}h | Heures: {r[4]}h")
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
    conn = _conn(); cur = conn.cursor()
    lignes: List[str] = []; strategies: List[str] = []
    shp_ids = extraire_shipment_id(question)
    drv_ids = extraire_driver_id(question)
    veh_ids = extraire_vehicule_id(question)
    rte_ids = extraire_route_id(question)
    if _contient(question, MOTS_STATS):
        lignes.append(_get_fleet_stats(cur))
        lignes += _stats_detaillees(cur)
        strategies.append("A")
    if _contient(question, MOTS_ROUTE) or rte_ids:
        lignes += _rechercher_routes(cur, question, rte_ids, top_k)
        strategies.append("B")
    if _contient(question, MOTS_CHAUFFEUR) or drv_ids:
        lignes += _rechercher_chauffeurs(cur, question, drv_ids, top_k)
        strategies.append("C")
    if _contient(question, MOTS_EXPEDITION) or shp_ids or veh_ids:
        lignes += _rechercher_expeditions(cur, question, shp_ids, veh_ids, top_k)
        strategies.append("D")
    if _contient(question, MOTS_ALERTES):
        lignes += _rechercher_alertes(cur, top_k)
        strategies.append("E")
    if not strategies:
        lignes += _fallback_general(cur, question, top_k)
        strategies.append("F")
    conn.close()
    vus, final = set(), []
    for l in lignes:
        if l not in vus:
            vus.add(l); final.append(l)
    print(f"  ✅ Stratégies SQL : {strategies} | {len(final)} résultats")
    return "\n\n---\n\n".join(final), len(final)


def rechercher_dans_chroma(question: str, top_k: int = TOP_K) -> Tuple[str, int]:
    if not pipeline["ready"]:
        init_pipeline(force_reindex=False)
        if not pipeline["ready"]:
            return "", 0
    vecteur   = pipeline["embed_obj"].embed_query(question)
    resultats = pipeline["index_obj"].query(
        query_embeddings=[vecteur], n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = resultats["documents"][0] if resultats["documents"] else []
    return "\n\n---\n\n".join(documents), len(documents)


# ═══════════════════════════════════════════════════════════════════
# 5. LANGGRAPH STATE + NODES
# ═══════════════════════════════════════════════════════════════════

# ✅ FIX 3 : historique_contexte ajouté au TypedDict
class EtatDiagnostic(TypedDict):
    question:            str
    type_requete:        str
    besoin_vector:       bool
    resultat_sql:        str
    resultat_vectoriel:  str
    prompt_utilisateur:  str
    reponse_llm:         str
    nb_sources:          int
    historique_contexte: str


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

MODELE_PROMPT = """
### Contexte
DONNÉES_SQL:
{resultat_sql}

EXTRAITS_DOCS:
{resultat_vectoriel}

### Historique de la conversation:
{historique_bloc}

### Question:
{question}
"""


# ✅ FIX 2 : suppression de la variable `q` inutilisée, utilisation cohérente de _contient()
def noeud_router(etat: EtatDiagnostic) -> EtatDiagnostic:
    if _contient(etat["question"], MOTS_STATS):
        type_req, besoin_vector = "stats", False
    elif _contient(etat["question"], MOTS_ROUTE) or extraire_route_id(etat["question"]):
        type_req, besoin_vector = "route", True
    elif _contient(etat["question"], MOTS_CHAUFFEUR) or extraire_driver_id(etat["question"]):
        type_req, besoin_vector = "chauffeur", True
    elif (_contient(etat["question"], MOTS_EXPEDITION)
          or extraire_shipment_id(etat["question"])
          or extraire_vehicule_id(etat["question"])):
        type_req, besoin_vector = "expedition", False
    elif _contient(etat["question"], MOTS_ALERTES):
        type_req, besoin_vector = "alerte", False
    elif _contient(etat["question"], MOTS_TECHNIQUE):   # ✅ FIX 2 : utilise _contient() comme les autres
        type_req, besoin_vector = "technique", True
    else:
        type_req, besoin_vector = "général", True
    print(f"  🔀 Router → type: '{type_req}' | vector: {besoin_vector}")
    return {**etat, "type_requete": type_req, "besoin_vector": besoin_vector}


def noeud_sql(etat: EtatDiagnostic) -> EtatDiagnostic:
    contexte, n = rechercher_dans_sql(etat["question"], TOP_K)
    print(f"  🗄️  SQL → {n} résultats")
    return {**etat, "resultat_sql": contexte or "Aucune donnée SQL pertinente.", "nb_sources": n}


def noeud_vector(etat: EtatDiagnostic) -> EtatDiagnostic:
    contexte, n = rechercher_dans_chroma(etat["question"], TOP_K)
    print(f"  📦 ChromaDB → {n} chunks")
    return {**etat, "resultat_vectoriel": contexte or "Aucun extrait trouvé."}


def noeud_skip_vector(etat: EtatDiagnostic) -> EtatDiagnostic:
    print("  ⏭️  ChromaDB ignoré")
    return {**etat, "resultat_vectoriel": ""}


def noeud_analyser(etat: EtatDiagnostic) -> EtatDiagnostic:
    hist = etat.get("historique_contexte", "")
    historique_bloc = f"### Contexte conversation précédente\n{hist}\n\n" if hist else ""
    prompt = MODELE_PROMPT.format(
        resultat_sql=etat.get("resultat_sql", "Aucune donnée SQL"),
        resultat_vectoriel=etat.get("resultat_vectoriel", "Aucun extrait technique"),
        question=etat.get("question", ""),
        historique_bloc=historique_bloc,
    )
    return {**etat, "prompt_utilisateur": prompt}


def noeud_llm(etat: EtatDiagnostic) -> EtatDiagnostic:
    if not GROQ_API_KEY:
        reponse = f"⚠️ GROQ_API_KEY non configurée.\n\n**Données SQL:**\n```\n{etat['resultat_sql'][:1200]}\n```"
        return {**etat, "reponse_llm": reponse}
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
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
        )
        reponse = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                reponse += content
        print(f"  🤖 LLM → {len(reponse)} caractères")
        reponse_nettoyee = re.sub(r'<think>.*?</think>', '', reponse, flags=re.DOTALL).strip()
        return {**etat, "reponse_llm": reponse_nettoyee}
    except Exception as e:
        return {**etat, "reponse_llm": f"❌ Erreur LLM: {e}\n\n**Données:**\n```\n{etat['resultat_sql'][:800]}\n```"}


# ═══════════════════════════════════════════════════════════════════
# 6. CONSTRUCTION DU GRAPHE LANGGRAPH
# ═══════════════════════════════════════════════════════════════════

def _condition_vector(etat: EtatDiagnostic) -> Literal["vector", "skip_vector"]:
    return "vector" if etat["besoin_vector"] else "skip_vector"


def construire_graphe():
    from langgraph.graph import StateGraph, END
    graphe = StateGraph(EtatDiagnostic)
    graphe.add_node("router",      noeud_router)
    graphe.add_node("sql",         noeud_sql)
    graphe.add_node("vector",      noeud_vector)
    graphe.add_node("skip_vector", noeud_skip_vector)
    graphe.add_node("analyser",    noeud_analyser)
    graphe.add_node("llm",         noeud_llm)
    graphe.set_entry_point("router")
    graphe.add_edge("router", "sql")
    graphe.add_conditional_edges(
        "sql", _condition_vector,
        {"vector": "vector", "skip_vector": "skip_vector"}
    )
    graphe.add_edge("vector",      "analyser")
    graphe.add_edge("skip_vector", "analyser")
    graphe.add_edge("analyser",    "llm")
    graphe.add_edge("llm",         END)
    return graphe.compile()


agent_logistic = construire_graphe()
print("✅ Graphe LangGraph compilé")


# ✅ FIX 4 : ternaire corrigé — bot_tronque séparé proprement
def poser_question(question: str, last_user: str = "", last_bot: str = "") -> Tuple[str, str, int]:
    historique_contexte = ""
    if last_user and last_bot:
        bot_tronque = last_bot[:600] + "..." if len(last_bot) > 600 else last_bot
        historique_contexte = (
            f"Utilisateur (dernier échange): {last_user}\n"
            f"Assistant (dernière réponse): {bot_tronque}"
        )
    etat_initial: EtatDiagnostic = {
        "question":            question,
        "type_requete":        "",
        "besoin_vector":       False,
        "resultat_sql":        "",
        "resultat_vectoriel":  "",
        "prompt_utilisateur":  "",
        "reponse_llm":         "",
        "nb_sources":          0,
        "historique_contexte": historique_contexte,
    }
    etat_final = agent_logistic.invoke(etat_initial)
    return etat_final["reponse_llm"], etat_final["type_requete"], etat_final["nb_sources"]


# ═══════════════════════════════════════════════════════════════════
# 7. ROUTES FLASK
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def index(): return render_template("index.html")

@app.route("/chatbot")
def chatbot(): return render_template("chatbot.html")

@app.route("/fleet")
def fleet(): return render_template("fleet.html")


@app.route("/api/stats")
def api_stats():
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM shipments");                                        total  = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM shipments WHERE statut='retard'");                  retards= cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM shipments WHERE statut='livre'");                   livres = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM drivers");                                          nb_d   = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM drivers WHERE infractions>=3 OR repos_restant_h=0"); d_risk = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM routes");                                           nb_r   = cur.fetchone()[0]
    cur.execute("SELECT ROUND(AVG(poids_kg),1) FROM shipments");                          avg_p  = cur.fetchone()[0]
    conn.close()
    return jsonify({
        "total_shipments": total, "retards": retards, "livres": livres,
        "nb_drivers": nb_d, "drivers_risk": d_risk, "nb_routes": nb_r,
        "avg_poids": avg_p,
        "taux_livraison": round(livres/total*100, 1) if total else 0,
        "taux_retard":    round(retards/total*100, 1) if total else 0,
    })


@app.route("/api/shipments_by_type")
def api_by_type():
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT type_marchandise, COUNT(*) FROM shipments GROUP BY type_marchandise ORDER BY 2 DESC")
    data = [{"type": r[0], "count": r[1]} for r in cur.fetchall()]
    conn.close(); return jsonify(data)


@app.route("/api/routes")
def api_routes():
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        SELECT r.id, r.depart, r.destination, r.distance_km, r.duree_estimee_h,
               r.type_route, r.peages,
               COUNT(s.id),
               SUM(CASE WHEN s.statut='retard' THEN 1 ELSE 0 END)
        FROM routes r LEFT JOIN shipments s ON r.id=s.route_id
        GROUP BY r.id ORDER BY 8 DESC
    """)
    data = [{"id":r[0],"depart":r[1],"destination":r[2],"distance_km":r[3],
             "duree_h":r[4],"type":r[5],"peages":r[6],"nb_shipments":r[7],"nb_retards":r[8]}
            for r in cur.fetchall()]
    conn.close(); return jsonify(data)


@app.route("/api/drivers")
def api_drivers():
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        SELECT d.id, d.nom, d.permis, d.heures_conduite, d.repos_restant_h,
               d.infractions, COUNT(s.id)
        FROM drivers d LEFT JOIN shipments s ON d.id=s.chauffeur_id
        GROUP BY d.id ORDER BY d.infractions DESC, d.repos_restant_h ASC
    """)
    data = []
    for r in cur.fetchall():
        risk = "critique" if (r[4]==0 or r[5]>=3) else ("warning" if (r[4]<4 or r[5]>=1) else "ok")
        data.append({"id":r[0],"nom":r[1],"permis":r[2],"heures":r[3],
                     "repos":r[4],"infractions":r[5],"missions":r[6],"risk":risk})
    conn.close(); return jsonify(data)


@app.route("/api/alerts")
def api_alerts():
    conn = get_db(); cur = conn.cursor(); alerts = []
    cur.execute("""
        SELECT s.id, s.vehicule_id, r.depart, r.destination, s.date_arrivee, d.nom
        FROM shipments s
        LEFT JOIN routes  r ON s.route_id     = r.id
        LEFT JOIN drivers d ON s.chauffeur_id = d.id
        WHERE s.statut='retard' ORDER BY s.date_arrivee DESC LIMIT 10
    """)
    for r in cur.fetchall():
        alerts.append({"type":"retard","level":"critical",
                       "message":f"Expédition {r[0]} en retard",
                       "detail":f"{r[2] or '?'} → {r[3] or '?'} | Véhicule: {r[1]} | Chauffeur: {r[5] or 'N/A'}",
                       "date":r[4]})
    cur.execute("""
        SELECT id, nom, infractions, repos_restant_h
        FROM drivers WHERE infractions>=3 OR repos_restant_h=0
        ORDER BY infractions DESC LIMIT 5
    """)
    for r in cur.fetchall():
        alerts.append({"type":"chauffeur","level":"critical" if r[3]==0 else "warning",
                       "message":f"Risque: {r[1]}",
                       "detail":f"Infractions: {r[2]} | Repos: {r[3]}h","date":None})
    conn.close(); return jsonify(alerts)


@app.route("/api/recent_shipments")
def api_recent():
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        SELECT s.id, s.vehicule_id, r.depart, r.destination,
               s.date_depart, s.date_arrivee, s.poids_kg,
               s.type_marchandise, s.statut, d.nom
        FROM shipments s
        LEFT JOIN routes  r ON s.route_id     = r.id
        LEFT JOIN drivers d ON s.chauffeur_id = d.id
        ORDER BY s.date_depart DESC LIMIT 20
    """)
    data = [{"id":r[0],"vehicule":r[1],"depart":r[2],"destination":r[3],
             "date_depart":r[4],"date_arrivee":r[5],"poids":r[6],
             "type":r[7],"statut":r[8],"chauffeur":r[9]}
            for r in cur.fetchall()]
    conn.close(); return jsonify(data)

# ═══════════════════════════════════════════════════════════════════
# 8. MANAGE ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route("/manage")
def manage(): return render_template("manage.html")


@app.route("/api/manage/route", methods=["POST"])
def api_add_route():
    data = request.get_json()
    required = ["depart", "destination", "distance_km", "duree_h"]
    for field in required:
        if not data.get(field):
            return jsonify({"error": f"Champ manquant: {field}"}), 400

    depart = data["depart"].strip()
    dest   = data["destination"].strip()
    if depart == dest:
        return jsonify({"error": "Départ et destination identiques"}), 400

    conn = get_db(); cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM routes")
    idx = cur.fetchone()[0] + 1
    route_id = f"RT-{depart[:3].upper()}-{dest[:3].upper()}-{idx:03d}"
    while True:
        cur.execute("SELECT id FROM routes WHERE id=?", (route_id,))
        if not cur.fetchone():
            break
        idx += 1
        route_id = f"RT-{depart[:3].upper()}-{dest[:3].upper()}-{idx:03d}"

    try:
        cur.execute("""
            INSERT INTO routes (id, depart, destination, distance_km, duree_estimee_h, type_route, peages, restrictions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            route_id, depart, dest,
            float(data["distance_km"]),
            float(data["duree_h"]),
            data.get("type_route", "nationale"),
            float(data.get("peages", 0)),
            data.get("restrictions", "aucune"),
        ))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "route_id": route_id}), 201
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500


@app.route("/api/manage/shipment", methods=["POST"])
def api_add_shipment():
    data = request.get_json()
    required = ["vehicule_id", "route_id", "chauffeur_id", "date_depart", "date_arrivee", "poids_kg"]
    for field in required:
        if not data.get(field):
            return jsonify({"error": f"Champ manquant: {field}"}), 400

    conn = get_db(); cur = conn.cursor()

    cur.execute("SELECT id FROM routes WHERE id=?", (data["route_id"],))
    if not cur.fetchone():
        conn.close()
        return jsonify({"error": "Route introuvable"}), 404

    cur.execute("SELECT id FROM drivers WHERE id=?", (data["chauffeur_id"],))
    if not cur.fetchone():
        conn.close()
        return jsonify({"error": "Chauffeur introuvable"}), 404

    suffix = ''.join(random.choices(string.digits, k=10))
    shipment_id = f"SHP-{suffix}"
    while True:
        cur.execute("SELECT id FROM shipments WHERE id=?", (shipment_id,))
        if not cur.fetchone():
            break
        suffix = ''.join(random.choices(string.digits, k=10))
        shipment_id = f"SHP-{suffix}"

    try:
        cur.execute("""
            INSERT INTO shipments
              (id, vehicule_id, route_id, date_depart, date_arrivee, poids_kg, type_marchandise, statut, chauffeur_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            shipment_id,
            data["vehicule_id"].strip(),
            data["route_id"],
            data["date_depart"],
            data["date_arrivee"],
            float(data["poids_kg"]),
            data.get("type_marchandise", "General"),
            data.get("statut", "en_transit"),
            data["chauffeur_id"],
        ))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "shipment_id": shipment_id}), 201
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500


@app.route("/api/manage/driver/<driver_id>", methods=["PATCH"])
def api_update_driver(driver_id):
    data = request.get_json()
    conn = get_db(); cur = conn.cursor()

    cur.execute("SELECT id FROM drivers WHERE id=?", (driver_id,))
    if not cur.fetchone():
        conn.close()
        return jsonify({"error": "Chauffeur introuvable"}), 404

    fields, values = [], []
    if "heures_conduite" in data:
        fields.append("heures_conduite=?"); values.append(float(data["heures_conduite"]))
    if "repos_restant_h" in data:
        fields.append("repos_restant_h=?"); values.append(float(data["repos_restant_h"]))
    if "infractions" in data:
        fields.append("infractions=?"); values.append(int(data["infractions"]))

    if not fields:
        conn.close()
        return jsonify({"error": "Aucun champ à modifier"}), 400

    try:
        values.append(driver_id)
        cur.execute(f"UPDATE drivers SET {', '.join(fields)} WHERE id=?", values)
        conn.commit()
        conn.close()
        return jsonify({"success": True, "driver_id": driver_id}), 200
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500

# ═══════════════════════════════════════════════════════════════════
# 9. CHAT HISTORY (JSON)
# ═══════════════════════════════════════════════════════════════════

HISTORY_FILE = Path(__file__).parent / "data" / "chat_history.json"
MAX_MESSAGES  = 10

def _load_history() -> dict:
    os.makedirs(HISTORY_FILE.parent, exist_ok=True)
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"conversations": [], "active_id": None}

def _save_history(data: dict):
    os.makedirs(HISTORY_FILE.parent, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _new_conv_id() -> str:
    return f"conv_{int(time.time())}_{random.randint(100, 999)}"


@app.route("/api/history", methods=["GET"])
def api_get_history():
    data = _load_history()
    convs = [
        {
            "id":      c["id"],
            "title":   c.get("title", "Conversation"),
            "count":   len(c.get("messages", [])),
            "created": c.get("created", ""),
        }
        for c in data["conversations"]
    ]
    return jsonify({"conversations": convs, "active_id": data.get("active_id")})


@app.route("/api/history/<conv_id>", methods=["GET"])
def api_get_conversation(conv_id):
    data = _load_history()
    for c in data["conversations"]:
        if c["id"] == conv_id:
            return jsonify(c)
    return jsonify({"error": "Conversation introuvable"}), 404


@app.route("/api/history/new", methods=["POST"])
def api_new_conversation():
    data  = _load_history()
    cid   = _new_conv_id()
    title = request.get_json(force=True).get("title", "Nouvelle conversation")
    data["conversations"].append({
        "id":       cid,
        "title":    title,
        "created":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "messages": [],
    })
    data["active_id"] = cid
    _save_history(data)
    return jsonify({"id": cid, "title": title}), 201


@app.route("/api/history/<conv_id>/delete", methods=["DELETE"])
def api_delete_conversation(conv_id):
    data = _load_history()
    data["conversations"] = [c for c in data["conversations"] if c["id"] != conv_id]
    if data.get("active_id") == conv_id:
        data["active_id"] = data["conversations"][-1]["id"] if data["conversations"] else None
    _save_history(data)
    return jsonify({"success": True})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload  = request.get_json()
    question = payload.get("message", "").strip()
    conv_id  = payload.get("conv_id")
    last_user_msg = payload.get("last_user", "").strip()
    last_bot_msg  = payload.get("last_bot",  "").strip()

    if not question:
        return jsonify({"error": "Message vide"}), 400

    data = _load_history()

    conv = None
    if conv_id:
        for c in data["conversations"]:
            if c["id"] == conv_id:
                conv = c; break

    if conv is None:
        conv_id = _new_conv_id()
        conv = {
            "id":       conv_id,
            "title":    question[:50],
            "created":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "messages": [],
        }
        data["conversations"].append(conv)
        data["active_id"] = conv_id

    if len(conv["messages"]) >= MAX_MESSAGES:
        new_conv_id = _new_conv_id()
        new_conv = {
            "id":       new_conv_id,
            "title":    question[:50],
            "created":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "messages": [],
        }
        data["conversations"].append(new_conv)
        data["active_id"] = new_conv_id
        conv    = new_conv
        conv_id = new_conv_id
        _save_history(data)
        return jsonify({"new_conversation": True, "conv_id": conv_id}), 200

    print(f"\nQuestion: {question[:80]}")
    reponse, type_req, nb_sources = poser_question(
        question,
        last_user=last_user_msg,
        last_bot=last_bot_msg,
    )

    conv["messages"].append({"role": "user",      "text": question,
                              "time": time.strftime("%H:%M")})
    conv["messages"].append({"role": "assistant", "text": reponse,
                              "time": time.strftime("%H:%M"),
                              "type": type_req, "sources": nb_sources})
    _save_history(data)

    return jsonify({
        "response":        reponse,
        "type_requete":    type_req,
        "sources":         nb_sources,
        "conv_id":         conv_id,
        "msg_count":       len(conv["messages"]),
        "max_messages":    MAX_MESSAGES,
    })


if __name__ == "__main__":
    print("=" * 60)
    print("🚚 Smart Delivery System — LangGraph Pipeline")
    print("=" * 60)
    init_pipeline(force_reindex=False)
    print("🌐 Serveur démarré → http://localhost:5000")
    # ✅ FIX 1 : use_reloader=False évite le double chargement du modèle embedding
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)