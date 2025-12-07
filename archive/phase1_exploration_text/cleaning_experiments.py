"""
Module pour gérer les expériences de comparaison de configurations de nettoyage de texte.

Ce module fournit des fonctions pour:
- Définir plusieurs configurations de nettoyage de texte
- Appliquer ces configurations à un dataset
- Comparer les performances de classification avec différentes stratégies de preprocessing
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import regex as reg
import html
import unicodedata
from ftfy import fix_text
from .cleaning import global_text_cleaner


def get_cleaning_configs():
    """
    Retourne les 6 configurations de nettoyage à comparer.

    Returns:
        dict: Dictionnaire {nom_config: dict_parametres}
              avec les 6 configurations (v0 à v5)

    Configurations:
        - v0_no_cleaning: Nettoyage minimal (baseline)
        - v1_basic: Nettoyage basique (HTML, unicode, lowercase)
        - v2_structural: v1 + fusions structurelles (dimensions, unités, durées)
        - v3_boilerplate: v2 + suppression du boilerplate
        - v4_no_digits: v3 + suppression des chiffres isolés
        - v5_no_letters: v3 + suppression des lettres isolées
    """

    # Config 0: Minimal cleaning (baseline with almost no preprocessing)
    # Only basic whitespace normalization, keep everything else intact
    config_v0 = {
        "use_basic_cleaning": False,
        "normalize_x_dimensions": False,
        "normalize_units": False,
        "normalize_durations": False,
        "normalize_age_ranges": False,
        "tag_year_numbers": False,
        "remove_boilerplate": False,
        "remove_nltk_stops": False,
        "remove_custom_stops": False,
        "remove_single_digit": False,
        "remove_single_letter": False,
        "lowercase": False,
    }

    # Config 1: Basic cleaning only
    # HTML removal, unicode normalization, lowercasing
    config_v1 = {
        "use_basic_cleaning": True,
        "normalize_x_dimensions": False,
        "normalize_units": False,
        "normalize_durations": False,
        "normalize_age_ranges": False,
        "tag_year_numbers": False,
        "remove_boilerplate": False,
        "remove_nltk_stops": False,
        "remove_custom_stops": False,
        "remove_single_digit": False,
        "remove_single_letter": False,
        "lowercase": True,
    }

    # Config 2: Structural normalization + basic cleaning
    # Merge dimensions (22x11), units (500g), durations (24h), age ranges
    config_v2 = {
        "use_basic_cleaning": True,
        "normalize_x_dimensions": True,
        "normalize_units": True,
        "normalize_durations": True,
        "normalize_age_ranges": True,
        "tag_year_numbers": False,
        "remove_boilerplate": False,
        "remove_nltk_stops": False,
        "remove_custom_stops": False,
        "remove_single_digit": False,
        "remove_single_letter": False,
        "lowercase": True,
    }

    # Config 3: Config 2 + boilerplate removal
    # Add removal of common HTML artifacts and template phrases
    config_v3 = {
        "use_basic_cleaning": True,
        "normalize_x_dimensions": True,
        "normalize_units": True,
        "normalize_durations": True,
        "normalize_age_ranges": True,
        "tag_year_numbers": False,
        "remove_boilerplate": True,
        "remove_nltk_stops": False,
        "remove_custom_stops": False,
        "remove_single_digit": False,
        "remove_single_letter": False,
        "lowercase": True,
    }

    # Config 4: Config 3 + remove single digits
    # Test impact of removing isolated digits (e.g., "1", "2", "5")
    config_v4 = {
        "use_basic_cleaning": True,
        "normalize_x_dimensions": True,
        "normalize_units": True,
        "normalize_durations": True,
        "normalize_age_ranges": True,
        "tag_year_numbers": False,
        "remove_boilerplate": True,
        "remove_nltk_stops": False,
        "remove_custom_stops": False,
        "remove_single_digit": True,  # ← différence ici
        "remove_single_letter": False,
        "lowercase": True,
    }

    # Config 5: Config 3 + remove single letters
    # Test impact of removing isolated letters (e.g., "a", "l", "s")
    config_v5 = {
        "use_basic_cleaning": True,
        "normalize_x_dimensions": True,
        "normalize_units": True,
        "normalize_durations": True,
        "normalize_age_ranges": True,
        "tag_year_numbers": False,
        "remove_boilerplate": True,
        "remove_nltk_stops": False,
        "remove_custom_stops": False,
        "remove_single_digit": False,
        "remove_single_letter": True,  # ← différence ici
        "lowercase": True,
    }

    # Store configs in a dictionary for iteration
    configs = {
        "v0_no_cleaning": config_v0,
        "v1_basic": config_v1,
        "v2_structural": config_v2,
        "v3_boilerplate": config_v3,
        "v4_no_digits": config_v4,
        "v5_no_letters": config_v5,
    }

    return configs


def build_text_raw_column(df):
    """
    Crée la colonne 'text_raw' par concaténation de designation + description.

    Args:
        df: DataFrame contenant les colonnes 'designation' et 'description'

    Returns:
        DataFrame: Le DataFrame avec la nouvelle colonne 'text_raw' ajoutée

    Note:
        Modifie le DataFrame en place ET le retourne pour chaînage.
    """
    df["text_raw"] = (
        df["designation"].fillna("").astype(str).str.strip() + " " +
        df["description"].fillna("").astype(str).str.strip()
    ).str.strip()

    print(f"✓ Colonne 'text_raw' créée : {len(df)} lignes")

    return df


def add_cleaned_variants(df, configs, verbose=True):
    """
    Applique chaque configuration de nettoyage et ajoute les colonnes au DataFrame.

    Args:
        df: DataFrame contenant la colonne 'text_raw'
        configs: Dictionnaire {nom_config: dict_parametres}
        verbose: Si True, affiche la progression

    Returns:
        DataFrame: Le DataFrame avec les nouvelles colonnes de texte nettoyé

    Note:
        Crée des colonnes nommées selon les clés du dictionnaire configs
        (par ex: v0_no_cleaning, v1_basic, etc.)
    """
    if verbose:
        print("Création des variantes de texte nettoyé...\n")

    for variant_name, config in configs.items():
        if verbose:
            print(f"  Traitement {variant_name}...", end=" ")

        # Appliquer global_text_cleaner avec cette configuration
        df[variant_name] = df["text_raw"].apply(
            lambda x: global_text_cleaner(x, **config)
        )

        if verbose:
            # Statistiques rapides
            avg_len = df[variant_name].str.len().mean()
            print(f"✓ (longueur moyenne: {avg_len:.0f} caractères)")

    if verbose:
        print(f"\n✓ {len(configs)} variantes créées")
        print(f"Colonnes disponibles : {list(configs.keys())}")

    return df


def run_cleaning_experiment(
    df,
    configs,
    label_column="prdtypecode",
    test_size=0.2,
    random_state=42,
    tfidf_params=None,
    logreg_params=None,
    verbose=True
):
    """
    Lance l'expérience complète de comparaison des configurations de nettoyage.

    Pour chaque configuration:
    1. Crée un split train/validation (MÊME split pour toutes les configs)
    2. Entraîne un pipeline TF-IDF + LogisticRegression
    3. Évalue sur validation avec F1 score (weighted)

    Args:
        df: DataFrame contenant les colonnes de texte nettoyé et les labels
        configs: Dictionnaire {nom_config: dict_parametres}
        label_column: Nom de la colonne contenant les labels (default: "prdtypecode")
        test_size: Proportion du split validation (default: 0.2)
        random_state: Seed pour la reproductibilité (default: 42)
        tfidf_params: Paramètres pour TfidfVectorizer (default: config optimisée)
        logreg_params: Paramètres pour LogisticRegression (default: config optimisée)
        verbose: Si True, affiche la progression

    Returns:
        pd.DataFrame: Résultats avec colonnes:
            - configuration: nom de la config
            - f1_weighted: score F1 pondéré sur validation
            - basic_cleaning: booléen
            - structural_merges: booléen
            - boilerplate_removal: booléen
            - remove_single_digit: booléen
            - remove_single_letter: booléen
    """

    # Paramètres par défaut pour TF-IDF
    if tfidf_params is None:
        tfidf_params = {
            "max_features": 10000,
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
        }

    # Paramètres par défaut pour LogisticRegression
    if logreg_params is None:
        logreg_params = {
            "C": 2.0,
            "max_iter": 1000,
            "random_state": random_state,
            "solver": "lbfgs",
            "multi_class": "multinomial",
        }

    # Préparer y (labels)
    y = df[label_column].values

    # Créer UN SEUL split pour garantir la comparaison équitable
    if verbose:
        print("Création du split train/validation (fixe pour toutes les configs)...")

    train_idx, valid_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    y_train = y[train_idx]
    y_valid = y[valid_idx]

    if verbose:
        print(f"  Train: {len(train_idx)} échantillons")
        print(f"  Validation: {len(valid_idx)} échantillons")
        print()

    # Liste pour stocker les résultats
    results = []

    # Boucle sur chaque configuration
    for variant_name in configs.keys():
        if verbose:
            print("=" * 70)
            print(f"CONFIGURATION : {variant_name.upper()}")
            print("=" * 70)

        # Préparer X pour cette variante
        X_text = df[variant_name].values
        X_train_text = X_text[train_idx]
        X_valid_text = X_text[valid_idx]

        # Créer le pipeline : TF-IDF + LogisticRegression
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('clf', LogisticRegression(**logreg_params))
        ])

        # Entraînement
        if verbose:
            print("  Entraînement en cours...", end=" ")
        pipeline.fit(X_train_text, y_train)
        if verbose:
            print("✓")

        # Prédiction sur validation
        if verbose:
            print("  Évaluation sur validation...", end=" ")
        y_pred = pipeline.predict(X_valid_text)
        f1_weighted = f1_score(y_valid, y_pred, average='weighted')
        if verbose:
            print(f"✓")
            print(f"\n  → F1 Score (weighted) : {f1_weighted:.4f}")
            print()

        # Extraire les flags de configuration pour l'analyse
        config = configs[variant_name]

        # Sauvegarder les résultats
        results.append({
            "configuration": variant_name,
            "f1_weighted": f1_weighted,
            "basic_cleaning": config["use_basic_cleaning"],
            "structural_merges": (
                config["normalize_x_dimensions"] or
                config["normalize_units"] or
                config["normalize_durations"] or
                config["normalize_age_ranges"]
            ),
            "boilerplate_removal": config["remove_boilerplate"],
            "remove_single_digit": config["remove_single_digit"],
            "remove_single_letter": config["remove_single_letter"],
        })

    if verbose:
        print("=" * 70)
        print("✓ ÉVALUATION TERMINÉE POUR TOUTES LES CONFIGURATIONS")
        print("=" * 70)

    # Créer le DataFrame de résultats et trier par F1 score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("f1_weighted", ascending=False).reset_index(drop=True)

    return results_df


def print_experiment_analysis(results_df):
    """
    Affiche une analyse détaillée des résultats de l'expérience.

    Args:
        results_df: DataFrame retourné par run_cleaning_experiment()
    """

    print("\n" + "=" * 80)
    print("TABLEAU COMPARATIF : IMPACT DES STRATÉGIES DE NETTOYAGE")
    print("=" * 80)
    print()
    print(results_df.to_string(index=False))
    print()

    # Identifier la meilleure configuration
    best_config = results_df.iloc[0]

    print("=" * 80)
    print("MEILLEURE CONFIGURATION")
    print("=" * 80)
    print(f"\nConfiguration        : {best_config['configuration']}")
    print(f"F1 Score (weighted)  : {best_config['f1_weighted']:.4f}")
    print(f"Basic cleaning       : {best_config['basic_cleaning']}")
    print(f"Structural merges    : {best_config['structural_merges']}")
    print(f"Boilerplate removal  : {best_config['boilerplate_removal']}")
    print(f"Remove single digits : {best_config['remove_single_digit']}")
    print(f"Remove single letters: {best_config['remove_single_letter']}")
    print()

    # Analyse comparative
    print("=" * 80)
    print("ANALYSE COMPARATIVE")
    print("=" * 80)
    print()

    # Helper function to get F1 score for a config
    def get_f1(config_name):
        return results_df[results_df['configuration'] == config_name]['f1_weighted'].values[0]

    # Comparer v0 (no cleaning) vs v1 (basic)
    v0_f1 = get_f1('v0_no_cleaning')
    v1_f1 = get_f1('v1_basic')
    print(f"1. Impact du nettoyage basique (v1 vs v0):")
    print(f"   v0 (no cleaning): {v0_f1:.4f}")
    print(f"   v1 (basic)      : {v1_f1:.4f}")
    print(f"   → Gain          : {v1_f1 - v0_f1:+.4f} ({(v1_f1/v0_f1 - 1)*100:+.2f}%)")
    print()

    # Comparer v1 (basic) vs v2 (structural)
    v2_f1 = get_f1('v2_structural')
    print(f"2. Impact des fusions structurelles (v2 vs v1):")
    print(f"   v1 (basic)      : {v1_f1:.4f}")
    print(f"   v2 (structural) : {v2_f1:.4f}")
    print(f"   → Gain          : {v2_f1 - v1_f1:+.4f} ({(v2_f1/v1_f1 - 1)*100:+.2f}%)")
    print()

    # Comparer v2 (structural) vs v3 (boilerplate)
    v3_f1 = get_f1('v3_boilerplate')
    print(f"3. Impact de la suppression du boilerplate (v3 vs v2):")
    print(f"   v2 (structural) : {v2_f1:.4f}")
    print(f"   v3 (boilerplate): {v3_f1:.4f}")
    print(f"   → Gain          : {v3_f1 - v2_f1:+.4f} ({(v3_f1/v2_f1 - 1)*100:+.2f}%)")
    print()

    # Comparer v3 vs v4 (remove_single_digit)
    v4_f1 = get_f1('v4_no_digits')
    print(f"4. Impact de la suppression des chiffres isolés (v4 vs v3):")
    print(f"   v3 (boilerplate): {v3_f1:.4f}")
    print(f"   v4 (no digits)  : {v4_f1:.4f}")
    print(f"   → Gain          : {v4_f1 - v3_f1:+.4f} ({(v4_f1/v3_f1 - 1)*100:+.2f}%)")
    print()

    # Comparer v3 vs v5 (remove_single_letter)
    v5_f1 = get_f1('v5_no_letters')
    print(f"5. Impact de la suppression des lettres isolées (v5 vs v3):")
    print(f"   v3 (boilerplate): {v3_f1:.4f}")
    print(f"   v5 (no letters) : {v5_f1:.4f}")
    print(f"   → Gain          : {v5_f1 - v3_f1:+.4f} ({(v5_f1/v3_f1 - 1)*100:+.2f}%)")
    print()

    print("=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print()
    print("Les résultats montrent que:")
    print()

    if v1_f1 > v0_f1:
        print("  ✓ Le nettoyage basique (HTML, unicode, lowercase) améliore les performances.")
    else:
        print("  ✗ Le nettoyage basique dégrade légèrement les performances.")

    if v2_f1 > v1_f1:
        print("  ✓ Les fusions structurelles (dimensions, unités, durées) sont bénéfiques.")
    else:
        print("  ✗ Les fusions structurelles n'apportent pas de gain significatif.")

    if v3_f1 > v2_f1:
        print("  ✓ La suppression du boilerplate améliore la classification.")
    else:
        print("  ✗ La suppression du boilerplate a un impact neutre ou négatif.")

    if v4_f1 < v3_f1:
        print("  ⚠ La suppression des chiffres isolés dégrade les performances (comme attendu).")
    else:
        print("  ✓ Surprenant : la suppression des chiffres isolés améliore les performances.")

    if v5_f1 < v3_f1:
        print("  ⚠ La suppression des lettres isolées dégrade les performances.")
    else:
        print("  ✓ La suppression des lettres isolées améliore les performances.")

    print()
    print(f"La configuration optimale identifiée est : {best_config['configuration']}")
    print(f"F1 Score : {best_config['f1_weighted']:.4f}")
    print()


def plot_experiment_results(results_df):
    """
    Crée un graphique en barres des résultats de l'expérience.

    Args:
        results_df: DataFrame retourné par run_cleaning_experiment()

    Note:
        Requiert matplotlib. Si non disponible, affiche un message d'avertissement.
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        configs = results_df['configuration'].values
        f1_scores = results_df['f1_weighted'].values

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9']
        bars = ax.bar(range(len(configs)), f1_scores, color=colors, alpha=0.8, edgecolor='black')

        # Annoter chaque barre avec le score
        for i, (bar, score) in enumerate(zip(bars, f1_scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax.set_xlabel('Configuration de nettoyage', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score (weighted)', fontsize=12, fontweight='bold')
        ax.set_title('Comparaison des stratégies de nettoyage de texte\n(TF-IDF + LogisticRegression)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([min(f1_scores) * 0.95, max(f1_scores) * 1.02])

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("(matplotlib non disponible pour la visualisation)")


#==============================================================================
# MODULE: ABLATION STUDY (GRANULAR CLEANING TEST)
# ==============================================================================

def granular_cleaner(text, keep_html_tags=False, keep_punctuation=False, skip_unescape=False):
    """
    Custom cleaner allowing granular control to disable specific cleaning steps 
    for ablation testing. 
    Base logic is 'v1_keep_case' (Basic Cleaning but NO Lowercase).
    """
    if pd.isna(text): return ""
    s = str(text)

    # A. Remove HTML Tags
    if not keep_html_tags:
        s = reg.sub(r"<[^>]+>", " ", s)

    # B. Decode & Fix Encoding
    if not skip_unescape:
        s = html.unescape(s)
        s = fix_text(s)
        s = unicodedata.normalize("NFC", s)

    # C. Remove Punctuation
    if not keep_punctuation:
        s = reg.sub(r"(?<!\d)\.(?!\d)", " ", s)  # Non-numeric dots
        s = reg.sub(r"(?<!\S)-(?!\S)", " ", s)   # Isolated hyphens
        s = reg.sub(r"(?<!\S):(?!\S)", " ", s)   # Isolated colons
        s = reg.sub(r"(?<!\S)·(?!\S)", " ", s)   # Middle dots
        s = reg.sub(r"(?<!\S)/(?!\S)", " ", s)   # Isolated slashes
        s = reg.sub(r"(?<!\S)\+(?!\S)", " ", s)   # Isolated plus signs
        s = s.replace("////", " ")

    # D. Final Cleanup
    # CRITICAL: Always keep case (Lowercase = False) for this study
    s = reg.sub(r"\s+", " ", s).strip()
    return s


def _create_dummy_config(description):
    """Helper to create a full config dictionary to satisfy run_cleaning_experiment expectations."""
    return {
        "use_basic_cleaning": True,
        "lowercase": False,
        "normalize_x_dimensions": False, "normalize_units": False,
        "normalize_durations": False, "normalize_age_ranges": False,
        "tag_year_numbers": False, "remove_boilerplate": False, 
        "remove_nltk_stops": False, "remove_custom_stops": False,
        "remove_single_digit": False, "remove_single_letter": False,
        "_description": description
    }


def run_detailed_ablation_study(df, random_state=42):
    """
    Runs a detailed ablation study to determine the impact of specific cleaning steps:
    1. Lowercase (v1_keep_case as baseline)
    2. HTML removal
    3. Punctuation removal
    4. Encoding fix
    
    Returns:
        pd.DataFrame: The experiment results sorted by F1 score.
    """
    print("1. Generating granular text variants...")

    # 1. Baseline: v1_keep_case (No Lowercase)
    if 'v1_keep_case' not in df.columns:
        print("  - Processing v1_keep_case (Baseline)...")
        df['v1_keep_case'] = df['text_raw'].apply(
            lambda x: granular_cleaner(x, keep_html_tags=False, keep_punctuation=False, skip_unescape=False)
        )
    else:
        print("  - v1_keep_case already exists.")

    # 2. Test: Keep HTML
    print("  - Processing test_keep_html...")
    df['test_keep_html'] = df['text_raw'].apply(
        lambda x: granular_cleaner(x, keep_html_tags=True, keep_punctuation=False, skip_unescape=False)
    )

    # 3. Test: Keep Punctuation
    print("  - Processing test_keep_punct...")
    df['test_keep_punct'] = df['text_raw'].apply(
        lambda x: granular_cleaner(x, keep_html_tags=False, keep_punctuation=True, skip_unescape=False)
    )

    # 4. Test: Skip Unescape
    print("  - Processing test_skip_unescape...")
    df['test_skip_unescape'] = df['text_raw'].apply(
        lambda x: granular_cleaner(x, keep_html_tags=False, keep_punctuation=False, skip_unescape=True)
    )

    # Define configs
    granular_configs = {
        "v1_keep_case": _create_dummy_config("Baseline (No Lowercase)"),
        "test_keep_html": _create_dummy_config("Keep HTML Tags"),
        "test_keep_punct": _create_dummy_config("Keep Punctuation"),
        "test_skip_unescape": _create_dummy_config("Skip Unescape/Fix")
    }

    print("\n2. Running benchmark...")
    results = run_cleaning_experiment(
        df=df,
        configs=granular_configs,
        random_state=random_state,
        verbose=True
    )

    # Automatic Analysis
    print("\n" + "="*80)
    print("DETAILED ANALYSIS: WHICH CLEANING STEP IS USEFUL?")
    print("="*80)
    
    base_row = results[results['configuration'] == 'v1_keep_case']
    if not base_row.empty:
        base_score = base_row['f1_weighted'].values[0]
        print(f"Baseline Score (v1_keep_case): {base_score:.6f}\n")

        for idx, row in results.iterrows():
            name = row['configuration']
            score = row['f1_weighted']
            if name == 'v1_keep_case': continue
            
            diff = score - base_score
            if diff > 0.0001:
                print(f"🚀 {name}: {score:.6f} (+{diff:.6f}) -> improving score! The removed step was harmful.")
            elif diff < -0.0001:
                print(f"📉 {name}: {score:.6f} ({diff:.6f}) -> score dropped. The removed step is NECESSARY.")
            else:
                print(f"➖ {name}: {score:.6f} -> No significant impact.")
    
    return results