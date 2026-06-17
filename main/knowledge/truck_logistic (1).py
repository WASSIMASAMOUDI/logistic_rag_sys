#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volvo Maroc Logistics - Database Builder
Crée une base de données SQLite avec les tables routes, drivers, shipments
à partir du fichier volvo_maroc_logistics.xlsx.

Utilisation :
    python build_logistics_db.py

Les chemins sont lus depuis le fichier .env :
    BD_TEST     = chemin de la base SQLite
    XLSX_VOLVO  = chemin du fichier Excel d'entrée
"""

import pandas as pd
import sqlite3
import os
from datetime import timedelta
from dotenv import load_dotenv

# ============================================================
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# ============================================================
load_dotenv()  # Lit le fichier .env à la racine

DB_PATH   = os.getenv("BD_TEST", "test/knowledge/truck_logistic.db")
XLSX_PATH = os.getenv("XLSX_VOLVO", "uploads/volvo_maroc_logistics.xlsx")

# ============================================================
# CORRESPONDANCE PORT -> VILLE (utilisée pour le champ depart)
# ============================================================
PORT_TO_CITY = {
    'PORT01': 'Casablanca',
    'PORT02': 'Rabat',
    'PORT03': 'Fes',
    'PORT04': 'Casablanca',
    'PORT05': 'Tanger',
    'PORT06': 'Meknes',
    'PORT07': 'Agadir',
    'PORT08': 'Oujda',
    'PORT09': 'Marrakech',
    'PORT10': 'Tetouan',
    'PORT11': 'Laayoune',
}

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def get_route_type(distance):
    """Classifie le type de route selon la distance."""
    if distance <= 100:
        return 'urbain'
    elif distance <= 500:
        return 'nationale'
    else:
        return 'autoroute'


def get_toll_cost(distance, route_type):
    """Estime le coût des péages en MAD."""
    if route_type == 'autoroute':
        return round(distance * 0.25, 2)
    elif route_type == 'nationale':
        return round(distance * 0.05, 2)
    else:
        return 0.0


def get_restrictions(distance, route_type):
    """Construit la chaîne des restrictions selon la route."""
    restrictions = []
    if route_type == 'autoroute':
        restrictions.append('poids_max_40t')
    if distance > 1000:
        restrictions.append('gabarit_renforce')
    if distance > 500:
        restrictions.append('vehicule_articule')
    return '; '.join(restrictions) if restrictions else 'aucune'


def count_infractions(inf_text):
    """Compte le nombre d'infractions à partir du texte."""
    if pd.isna(inf_text) or str(inf_text).strip() == 'Aucune':
        return 0
    return str(inf_text).count(';') + 1


def calc_arrival_date(depart_date, transit_days, advance_days, retard_days):
    """Calcule la date d'arrivée à partir de la date de départ et des ajustements."""
    if pd.isna(depart_date):
        return None
    base_date = pd.to_datetime(depart_date)
    actual_days = int(transit_days) - int(advance_days) + int(retard_days)
    arrival = base_date + timedelta(days=max(actual_days, 0))
    return arrival.strftime('%Y-%m-%d')


# ============================================================
# CONSTRUCTION PRINCIPALE
# ============================================================

def build_database():
    print("=" * 60)
    print("VOLVO MAROC LOGISTICS - CONSTRUCTION DE LA BASE")
    print("=" * 60)

    # --- 1. Création du dossier de sortie ---
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    # Suppression de l'ancienne base si elle existe
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"[INFO] Ancienne base supprimée : {DB_PATH}")

    # --- 2. Lecture du fichier Excel ---
    print(f"\n[1/6] Chargement du fichier Excel : {XLSX_PATH}")
    df = pd.read_excel(XLSX_PATH)
    print(f"       {len(df):,} lignes, {len(df.columns)} colonnes chargées")

    # Ajout de la ville de départ
    df['depart_ville'] = df['Port_depart'].map(PORT_TO_CITY)

    # --- 3. Construction de la table ROUTES ---
    print("\n[2/6] Construction de la table ROUTES...")

    routes_agg = df.groupby(
        ['Port_depart', 'Port_arrivee', 'Ville_arrivee', 'depart_ville']
    ).agg({
        'Distance_km': 'first',
        'Heures_transit': 'first',
        'TPT_contractuel_jours': 'first'
    }).reset_index()

    routes_rows = []
    route_lookup = {}   # (Port_depart, Port_arrivee, Ville_arrivee) -> route_id

    for idx, row in routes_agg.iterrows():
        depart = row['depart_ville']
        destination = row['Ville_arrivee']
        distance = float(row['Distance_km']) if pd.notna(row['Distance_km']) else 0.0
        duree = float(row['Heures_transit']) if pd.notna(row['Heures_transit']) else 0.0
        route_type = get_route_type(distance)

        route_id = f"RT-{depart[:3].upper()}-{destination[:3].upper()}-{idx+1:03d}"

        routes_rows.append({
            'id': route_id,
            'depart': depart,
            'destination': destination,
            'distance_km': distance,
            'duree_estimee_h': duree,
            'type_route': route_type,
            'peages': get_toll_cost(distance, route_type),
            'restrictions': get_restrictions(distance, route_type)
        })

        key = (row['Port_depart'], row['Port_arrivee'], row['Ville_arrivee'])
        route_lookup[key] = route_id

    routes_df = pd.DataFrame(routes_rows)
    print(f"       {len(routes_df)} routes uniques créées")

    # --- 4. Construction de la table DRIVERS ---
    print("\n[3/6] Construction de la table DRIVERS...")

    drivers_data = df[['chauffeur_id', 'nom', 'permis',
                       'heures_conduite', 'repos_restant_h', 'infractions'
                      ]].drop_duplicates(subset=['chauffeur_id'])

    drivers_rows = []
    for _, row in drivers_data.iterrows():
        drivers_rows.append({
            'id': str(row['chauffeur_id']),
            'nom': str(row['nom']),
            'permis': str(row['permis']) if pd.notna(row['permis']) else 'C1',
            'heures_conduite': float(row['heures_conduite']) if pd.notna(row['heures_conduite']) else 0.0,
            'repos_restant_h': float(row['repos_restant_h']) if pd.notna(row['repos_restant_h']) else 0.0,
            'infractions': count_infractions(row['infractions'])
        })

    drivers_df = pd.DataFrame(drivers_rows)
    print(f"       {len(drivers_df):,} chauffeurs uniques créés")

    # --- 5. Construction de la table SHIPMENTS ---
    print("\n[4/6] Construction de la table SHIPMENTS...")

    STATUS_MAP = {
        'EN AVANCE': 'livre',
        'A TEMPS': 'livre',
        'RETARD': 'retard',
        'ANNULE': 'annule'
    }

    shipments_rows = []
    for _, row in df.iterrows():
        key = (row['Port_depart'], row['Port_arrivee'], row['Ville_arrivee'])
        route_id = route_lookup.get(key, 'RT-UNK')

        depart_date = pd.to_datetime(row['Date_commande']).strftime('%Y-%m-%d') \
            if pd.notna(row['Date_commande']) else None

        transit_days = row['Jours_transit_estimes'] if pd.notna(row['Jours_transit_estimes']) else 1
        advance_days = row['Avance_jours'] if pd.notna(row['Avance_jours']) else 0
        retard_days = row['Retard_jours'] if pd.notna(row['Retard_jours']) else 0

        arrival_date = calc_arrival_date(row['Date_commande'], transit_days,
                                          advance_days, retard_days)

        statut = STATUS_MAP.get(str(row['Statut_livraison']).strip(), 'en_transit')

        shipments_rows.append({
            'id': f"SHP-{int(row['Order_ID']):010d}",
            'vehicule_id': str(row['Transporteur']) if pd.notna(row['Transporteur']) else 'V44_3',
            'route_id': route_id,
            'date_depart': depart_date,
            'date_arrivee': arrival_date,
            'poids_kg': float(row['Poids_tonnes']) * 1000 if pd.notna(row['Poids_tonnes']) else 0.0,
            'type_marchandise': str(row['type_marchandise']) if pd.notna(row['type_marchandise']) else 'General',
            'statut': statut,
            'chauffeur_id': str(row['chauffeur_id']) if pd.notna(row['chauffeur_id']) else 'UNK'
        })

    shipments_df = pd.DataFrame(shipments_rows)
    print(f"       {len(shipments_df):,} expéditions créées")

    # --- 6. Création de la base SQLite ---
    print("\n[5/6] Création de la base SQLite...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Table routes
    cursor.execute('''
        CREATE TABLE routes (
            id              TEXT PRIMARY KEY,
            depart          TEXT,
            destination     TEXT,
            distance_km     REAL,
            duree_estimee_h REAL,
            type_route      TEXT,
            peages          REAL,
            restrictions    TEXT
        );
    ''')

    # Table drivers
    cursor.execute('''
        CREATE TABLE drivers (
            id              TEXT PRIMARY KEY,
            nom             TEXT,
            permis          TEXT,
            heures_conduite REAL,
            repos_restant_h REAL,
            infractions     INTEGER
        );
    ''')

    # Table shipments (avec clés étrangères)
    cursor.execute('''
        CREATE TABLE shipments (
            id              TEXT PRIMARY KEY,
            vehicule_id     TEXT,
            route_id        TEXT,
            date_depart     TEXT,
            date_arrivee    TEXT,
            poids_kg        REAL,
            type_marchandise TEXT,
            statut          TEXT,
            chauffeur_id    TEXT,
            FOREIGN KEY (route_id)     REFERENCES routes(id),
            FOREIGN KEY (chauffeur_id) REFERENCES drivers(id)
        );
    ''')

    # Insertion des données
    routes_df.to_sql('routes', conn, if_exists='append', index=False)
    drivers_df.to_sql('drivers', conn, if_exists='append', index=False)
    shipments_df.to_sql('shipments', conn, if_exists='append', index=False)

    # Index pour les performances
    cursor.execute('CREATE INDEX idx_shipments_route   ON shipments(route_id);')
    cursor.execute('CREATE INDEX idx_shipments_driver  ON shipments(chauffeur_id);')
    cursor.execute('CREATE INDEX idx_shipments_statut  ON shipments(statut);')
    cursor.execute('CREATE INDEX idx_routes_depart     ON routes(depart);')
    cursor.execute('CREATE INDEX idx_routes_destination ON routes(destination);')

    conn.commit()

    # --- 7. Vérification ---
    print("\n[6/6] Vérification de la base...")

    cursor.execute("SELECT COUNT(*) FROM routes")
    n_routes = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM drivers")
    n_drivers = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM shipments")
    n_shipments = cursor.fetchone()[0]

    cursor.execute('''
        SELECT COUNT(*) FROM shipments s
        LEFT JOIN routes r ON s.route_id = r.id
        WHERE r.id IS NULL
    ''')
    orphan_routes = cursor.fetchone()[0]

    cursor.execute('''
        SELECT COUNT(*) FROM shipments s
        LEFT JOIN drivers d ON s.chauffeur_id = d.id
        WHERE d.id IS NULL
    ''')
    orphan_drivers = cursor.fetchone()[0]

    conn.close()

    # --- Résumé ---
    print("\n" + "=" * 60)
    print("BASE DE DONNÉES CRÉÉE AVEC SUCCÈS")
    print("=" * 60)
    print(f"  Chemin :       {os.path.abspath(DB_PATH)}")
    print(f"  Routes :       {n_routes:,}")
    print(f"  Chauffeurs :   {n_drivers:,}")
    print(f"  Expéditions :  {n_shipments:,}")
    print(f"  Clés orphelines : routes={orphan_routes}, chauffeurs={orphan_drivers}")
    print("=" * 60)

    return True


if __name__ == '__main__':
    build_database()