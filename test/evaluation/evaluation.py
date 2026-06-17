"""
Module d'évaluation manuelle par LLM-juges.
Score final = moyenne(Gemini + Claude + DeepSeek) pondérée par difficulté.
Variables d'environnement chargées depuis .env
"""

import os
import json
import numpy as np
from dotenv import load_dotenv

# ── Chargement .env ───────────────────────────────────────────────
load_dotenv()

OUTPUT_FILE  = os.getenv("RESULTS_PATH",  "test/evaluation/questionResults.json")
RAPPORT_PATH = os.getenv("RAPPORT_FINAL", "test/evaluation/rapport_final.json")
EVAL_DIR     = os.path.dirname(RAPPORT_PATH)

# ── Constantes ────────────────────────────────────────────────────
MODELES_EVALUATION = ["Gemini", "Claude", "DeepSeek"]

COEFF_DIFFICULTE = {
    "facile":    1.0,
    "moyen":     1.5,
    "difficile": 2.0,
}


# ══════════════════════════════════════════════════════════════════
#  Utilitaires
# ══════════════════════════════════════════════════════════════════

def sauvegarder_resultats(resultats: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)


def _saisir_note(label: str) -> float:
    """Saisie et validation d'une note entre 0 et 5."""
    while True:
        try:
            note = float(input(f"  Note {label} (0-5) : "))
            if 0.0 <= note <= 5.0:
                return note
            print("  ⚠️  Valeur invalide — entrez un nombre entre 0 et 5.")
        except ValueError:
            print("  ⚠️  Entrée invalide — veuillez saisir un nombre.")


def _note_vers_100(note: float) -> float:
    """Convertit une note /5 en score /100."""
    return round((note / 5.0) * 100.0, 4)


# ══════════════════════════════════════════════════════════════════
#  Saisie des notes + calcul du score par question
# ══════════════════════════════════════════════════════════════════

def saisir_notes_et_calculer(
    resultats: list,
    modeles: list = None
) -> list:
    """
    Pour chaque question :
      - Affiche : question, réponse correcte, réponse LLM
      - Demande une note /5 pour chaque modèle (Gemini, Claude, DeepSeek)
      - Score final = moyenne des 3 scores /100
    """
    if modeles is None:
        modeles = MODELES_EVALUATION

    resultats_enrichis = []
    total = len(resultats)

    print("=" * 72)
    print("🏆  SAISIE DES NOTES ÉVALUATEURS")
    print(f"    Modèles  : {' + '.join(modeles)}")
    print(f"    Score final = moyenne des {len(modeles)} évaluateurs (/100)")
    print("=" * 72)

    for res in resultats:
        idx        = res["id"]
        question   = res["question"]
        difficulty = res.get("difficulty", "facile")
        coeff      = COEFF_DIFFICULTE.get(difficulty, 1.0)
        reponse    = res.get("reponse_llm", "")
        type_req   = res.get("type_requete", "?")

        print(f"\n{'─' * 72}")
        print(f"  Question [{idx:02d}/{total}]  |  "
              f"Type : {type_req:10}  |  "
              f"Difficulté : {difficulty.upper():8}  |  "
              f"Coeff : ×{coeff}")
        print(f"{'─' * 72}")
        print(f"  ❓ {question}")
        print(f"  ✅ Réponse correcte : {res['reponse_correcte']}")
        print(f"  🤖 Réponse LLM      : "
              f"{reponse[:250]}{'...' if len(reponse) > 250 else ''}")
        print()

        # ── Notes des évaluateurs ─────────────────────────────────
        evaluations    = {}
        scores_sur_100 = []

        for modele in modeles:
            note      = _saisir_note(modele)
            score_100 = _note_vers_100(note)
            evaluations[modele] = {
                "note_sur_5":    note,
                "score_sur_100": score_100,
            }
            scores_sur_100.append(score_100)
            print(f"    → {modele:<10} : {note}/5  →  {score_100:.2f}/100")

        # ── Score final = moyenne des évaluateurs ─────────────────
        score_final = round(sum(scores_sur_100) / len(scores_sur_100), 4)

        print()
        print(f"  ⭐ Score final [{idx:02d}] = "
              f"({' + '.join(f'{s:.2f}' for s in scores_sur_100)}) "
              f"/ {len(scores_sur_100)} = {score_final:.2f}/100")

        enrichi                         = res.copy()
        enrichi["evaluations"]          = evaluations
        enrichi["score_final_question"] = score_final
        resultats_enrichis.append(enrichi)

    print(f"\n{'═' * 72}")
    print(f"✅ {len(resultats_enrichis)} questions évaluées.")
    return resultats_enrichis


# ══════════════════════════════════════════════════════════════════
#  Score global pondéré par difficulté
# ══════════════════════════════════════════════════════════════════

def calculer_score_global(resultats: list) -> dict:
    """
    Retourne :
      - score pondéré /100 pour chaque évaluateur
      - SCORE_FINAL_GLOBAL = moyenne pondérée des scores finaux
    """
    somme_ponderee       = {m: 0.0 for m in MODELES_EVALUATION}
    somme_ponderee_final = 0.0
    total_coeff          = 0.0

    for res in resultats:
        coeff = COEFF_DIFFICULTE.get(res.get("difficulty", "facile"), 1.0)
        evals = res.get("evaluations", {})
        s_fin = res.get("score_final_question", 0.0)

        somme_ponderee_final += s_fin * coeff
        total_coeff          += coeff

        for m in MODELES_EVALUATION:
            if m in evals:
                somme_ponderee[m] += evals[m].get("score_sur_100", 0.0) * coeff

    scores = {
        m: round(somme_ponderee[m] / total_coeff, 2) if total_coeff > 0 else 0.0
        for m in MODELES_EVALUATION
    }
    scores["SCORE_FINAL_GLOBAL"] = (
        round(somme_ponderee_final / total_coeff, 2) if total_coeff > 0 else 0.0
    )
    return scores


# ══════════════════════════════════════════════════════════════════
#  Rapport final
# ══════════════════════════════════════════════════════════════════

def afficher_rapport_final(resultats: list) -> None:
    """
    Affiche :
      - Scores globaux /100 par évaluateur
      - Score moyen par niveau de difficulté
      - Tableau récapitulatif par question
    Sauvegarde rapport_final.json et questionResults.json.
    """
    modeles        = MODELES_EVALUATION
    scores_globaux = calculer_score_global(resultats)

    # ── Scores globaux /100 ───────────────────────────────────────
    print("\n" + "═" * 70)
    print("🏆  SCORES GLOBAUX /100  (Σ note×coeff / Σ 5×coeff × 100)")
    print("═" * 70)
    for modele, score in scores_globaux.items():
        barre = "█" * int(score / 5)
        print(f"  {modele:<20} : {score:6.2f}/100  {barre}")

    # ── Score final moyen par difficulté ──────────────────────────
    print("\n" + "─" * 70)
    print("📊  SCORE FINAL MOYEN par niveau de difficulté")
    print("─" * 70)

    stats_diff = {}
    for diff in ["facile", "moyen", "difficile"]:
        valeurs = [
            r["score_final_question"]
            for r in resultats
            if r.get("difficulty") == diff and "score_final_question" in r
        ]
        if valeurs:
            moy = round(float(np.mean(valeurs)), 2)
            stats_diff[diff] = moy
            print(f"  {diff.capitalize():<12} : {moy:6.2f}/100  "
                  f"(n={len(valeurs)} | "
                  f"min={min(valeurs):.2f} | "
                  f"max={max(valeurs):.2f})")

    # ── Tableau par question ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("📋  DÉTAIL PAR QUESTION")
    print("─" * 70)

    header = f"{'ID':>3}  {'Diff':<10}  "
    for m in modeles:
        header += f"{m:>10}  "
    header += f"{'Final':>7}  Question"
    print(header)
    print("─" * 70)

    for res in resultats:
        idx         = res["id"]
        diff        = res.get("difficulty", "?")[:6]
        evals       = res.get("evaluations", {})
        score_final = res.get("score_final_question", 0.0)
        q_short     = res["question"][:40] + "..."

        ligne = f"{idx:>3}  {diff:<10}  "
        for m in modeles:
            note     = evals.get(m, {}).get("note_sur_5", "-")
            note_str = f"{note:.1f}/5" if isinstance(note, float) else str(note)
            ligne   += f"{note_str:>10}  "
        ligne += f"{score_final:>7.2f}  {q_short}"
        print(ligne)

    # ── Sauvegarde rapport_final.json ─────────────────────────────
    rapport = {
        "scores_globaux":       scores_globaux,
        "score_moyen_par_diff": stats_diff,
        "total_questions":      len(resultats),
        "coefficients":         COEFF_DIFFICULTE,
        "modeles_evalues":      modeles,
        "details":              resultats,
    }

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(RAPPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)

    sauvegarder_resultats(resultats, OUTPUT_FILE)

    print(f"\n✅ Rapport final sauvegardé       → {RAPPORT_PATH}")
    print(f"✅ questionResults.json mis à jour → {OUTPUT_FILE}")


# ══════════════════════════════════════════════════════════════════
#  Point d'entrée
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Supporte list directe ou dict avec clé "results" / "questions" / "data"
    if isinstance(data, list):
        resultats_bruts = data
    elif isinstance(data, dict):
        # Cherche la première valeur qui est une liste
        resultats_bruts = next(
            (v for v in data.values() if isinstance(v, list)), None
        )
        if resultats_bruts is None:
            raise ValueError(
                f"Structure JSON inattendue — clés disponibles : {list(data.keys())}"
            )
    else:
        raise ValueError("Le fichier JSON doit contenir une liste ou un objet.")

    resultats_notes = saisir_notes_et_calculer(resultats_bruts)
    sauvegarder_resultats(resultats_notes, OUTPUT_FILE)
    print("✅ Évaluations sauvegardées dans questionResults.json.")

    afficher_rapport_final(resultats_notes)