"""
Benchmark experiments for text cleaning strategies.

This module orchestrates ablation studies to measure the marginal impact
of each preprocessing step on classification performance.

Experimental Design:
1. Baseline: Raw data (no preprocessing)
2. Single-step tests: Enable only ONE cleaning option at a time
3. Fair comparison: Same train/test split (random_state=42) for all configs
4. Simple model: TF-IDF + LogisticRegression (focus on preprocessing effects)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from .xiaosong_cleaning import clean_text, get_available_options


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset(data_dir="../data"):
    """
    Load Rakuten dataset and create text_raw column.

    Args:
        data_dir: Path to data directory

    Returns:
        pd.DataFrame: Dataset with columns including 'text_raw' and 'prdtypecode'
    """
    print("=" * 80)
    print("Loading Rakuten dataset...")
    print("=" * 80)

    # Load features and labels
    X_train = pd.read_csv(f"{data_dir}/X_train_update.csv", index_col=0)
    Y_train = pd.read_csv(f"{data_dir}/Y_train_CVw08PX.csv", index_col=0)

    # Merge on index
    df = X_train.join(Y_train, how="inner")

    # Create text_raw: designation + " " + description
    df["text_raw"] = (
        df["designation"].fillna("").astype(str).str.strip() + " " +
        df["description"].fillna("").astype(str).str.strip()
    ).str.strip()

    print(f"✓ Dataset loaded: {df.shape[0]:,} samples")
    print(f"✓ Columns: {df.columns.tolist()}")
    print(f"✓ Classes: {df['prdtypecode'].nunique()} unique product types")
    print(f"✓ Text_raw created (avg length: {df['text_raw'].str.len().mean():.0f} chars)")
    print()

    return df


# =============================================================================
# Experiment Configuration
# =============================================================================

def define_experiments():
    """
    Define all experiment configurations for ablation study.

    Returns:
        list of dict: Each dict contains:
            - 'name': Human-readable experiment name
            - 'config': Dictionary of cleaning options (kwargs for clean_text)
            - 'group': Category (for organizing results)

    Experiment Design:
        - Group 0: Baseline (no preprocessing)
        - Group 1: Encoding & Unicode fixes
        - Group 2: HTML & structure cleanup
        - Group 3: Case transformation
        - Group 4: Structural merges (preserve semantic units)
        - Group 5: Punctuation handling
        - Group 6: Token filtering
    """
    experiments = []

    # -------------------------------------------------------------------------
    # Group 0: Baseline
    # -------------------------------------------------------------------------

    experiments.append({
        "name": "baseline_raw",
        "group": "0_Baseline",
        "config": {}  # All options False by default
    })

    # -------------------------------------------------------------------------
    # Group 1: Encoding & Unicode
    # -------------------------------------------------------------------------

    experiments.append({
        "name": "fix_encoding",
        "group": "1_Encoding",
        "config": {"fix_encoding": True}
    })

    experiments.append({
        "name": "unescape_html",
        "group": "1_Encoding",
        "config": {"unescape_html": True}
    })

    experiments.append({
        "name": "normalize_unicode",
        "group": "1_Encoding",
        "config": {"normalize_unicode": True}
    })

    # Combo: All encoding fixes
    experiments.append({
        "name": "all_encoding_fixes",
        "group": "1_Encoding",
        "config": {
            "fix_encoding": True,
            "unescape_html": True,
            "normalize_unicode": True
        }
    })

    # -------------------------------------------------------------------------
    # Group 2: HTML & Structure
    # -------------------------------------------------------------------------

    experiments.append({
        "name": "remove_html_tags",
        "group": "2_HTML",
        "config": {"remove_html_tags": True}
    })

    experiments.append({
        "name": "remove_boilerplate",
        "group": "2_HTML",
        "config": {"remove_boilerplate": True}
    })

    # -------------------------------------------------------------------------
    # Group 3: Case Transformation
    # -------------------------------------------------------------------------

    experiments.append({
        "name": "lowercase",
        "group": "3_Case",
        "config": {"lowercase": True}
    })

    # -------------------------------------------------------------------------
    # Group 4: Structural Merges
    # -------------------------------------------------------------------------

    experiments.append({
        "name": "merge_dimensions",
        "group": "4_Merges",
        "config": {"merge_dimensions": True}
    })

    experiments.append({
        "name": "merge_units",
        "group": "4_Merges",
        "config": {"merge_units": True}
    })

    experiments.append({
        "name": "merge_durations",
        "group": "4_Merges",
        "config": {"merge_durations": True}
    })

    experiments.append({
        "name": "merge_age_ranges",
        "group": "4_Merges",
        "config": {"merge_age_ranges": True}
    })

    experiments.append({
        "name": "tag_years",
        "group": "4_Merges",
        "config": {"tag_years": True}
    })

    # Combo: All structural merges
    experiments.append({
        "name": "all_merges",
        "group": "4_Merges",
        "config": {
            "merge_dimensions": True,
            "merge_units": True,
            "merge_durations": True,
            "merge_age_ranges": True
        }
    })

    # -------------------------------------------------------------------------
    # Group 5: Punctuation
    # -------------------------------------------------------------------------

    experiments.append({
        "name": "remove_punctuation",
        "group": "5_Punctuation",
        "config": {"remove_punctuation": True}
    })

    # -------------------------------------------------------------------------
    # Group 6: Token Filtering
    # -------------------------------------------------------------------------

    experiments.append({
        "name": "remove_stopwords",
        "group": "6_Filtering",
        "config": {"remove_stopwords": True}
    })

    experiments.append({
        "name": "remove_single_letters",
        "group": "6_Filtering",
        "config": {"remove_single_letters": True}
    })

    experiments.append({
        "name": "remove_single_digits",
        "group": "6_Filtering",
        "config": {"remove_single_digits": True}
    })

    experiments.append({
        "name": "remove_pure_punct_tokens",
        "group": "6_Filtering",
        "config": {"remove_pure_punct_tokens": True}
    })

    # -------------------------------------------------------------------------
    # Group 7: Common Combinations (for reference)
    # -------------------------------------------------------------------------

    # Traditional "clean" approach
    experiments.append({
        "name": "traditional_cleaning",
        "group": "7_Combos",
        "config": {
            "fix_encoding": True,
            "unescape_html": True,
            "normalize_unicode": True,
            "remove_html_tags": True,
            "lowercase": True,
            "remove_punctuation": True,
            "remove_stopwords": True
        }
    })

    # Conservative approach (encoding + HTML only)
    experiments.append({
        "name": "conservative_cleaning",
        "group": "7_Combos",
        "config": {
            "fix_encoding": True,
            "unescape_html": True,
            "normalize_unicode": True,
            "remove_html_tags": True
        }
    })

    # Merges only (no removal)
    experiments.append({
        "name": "merges_only",
        "group": "7_Combos",
        "config": {
            "merge_dimensions": True,
            "merge_units": True,
            "merge_durations": True,
            "merge_age_ranges": True
        }
    })

    return experiments


# =============================================================================
# Benchmark Execution
# =============================================================================

def run_benchmark(
    df,
    experiments=None,
    test_size=0.2,
    random_state=42,
    tfidf_max_features=10000,
    tfidf_ngram_range=(1, 2),
    verbose=True
):
    """
    Run benchmark experiments to compare cleaning strategies.

    Args:
        df: DataFrame with 'text_raw' and 'prdtypecode' columns
        experiments: List of experiment configs (if None, uses define_experiments())
        test_size: Proportion for validation split
        random_state: Random seed for reproducibility
        tfidf_max_features: Max features for TF-IDF
        tfidf_ngram_range: N-gram range for TF-IDF
        verbose: Print progress

    Returns:
        pd.DataFrame: Results with columns:
            - experiment: Experiment name
            - group: Experiment group
            - f1_weighted: F1 score (weighted)
            - accuracy: Accuracy score
            - delta_f1: Difference vs baseline
            - delta_pct: Percentage change vs baseline
    """
    if experiments is None:
        experiments = define_experiments()

    if verbose:
        print("=" * 80)
        print("BENCHMARK CONFIGURATION")
        print("=" * 80)
        print(f"Total experiments   : {len(experiments)}")
        print(f"Test size           : {test_size}")
        print(f"Random state        : {random_state}")
        print(f"TF-IDF max features : {tfidf_max_features:,}")
        print(f"TF-IDF n-gram range : {tfidf_ngram_range}")
        print("=" * 80)
        print()

    # Prepare labels
    y = df["prdtypecode"].values

    # Create single train/test split (shared across all experiments)
    if verbose:
        print("Creating train/test split...")

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    y_train = y[train_idx]
    y_test = y[test_idx]

    if verbose:
        print(f"  Train: {len(train_idx):,} samples")
        print(f"  Test : {len(test_idx):,} samples")
        print()

    # Store results
    results = []
    baseline_f1 = None

    # Run experiments
    for i, exp in enumerate(experiments, 1):
        exp_name = exp["name"]
        exp_group = exp["group"]
        exp_config = exp["config"]

        if verbose:
            print(f"[{i}/{len(experiments)}] {exp_name}")
            print(f"  Group : {exp_group}")
            print(f"  Config: {exp_config if exp_config else 'None (raw data)'}")

        # Apply cleaning to ALL data first
        if verbose:
            print("  Cleaning text...", end=" ")

        df[f"text_clean_{exp_name}"] = df["text_raw"].apply(
            lambda x: clean_text(x, **exp_config)
        )

        if verbose:
            avg_len = df[f"text_clean_{exp_name}"].str.len().mean()
            print(f"✓ (avg length: {avg_len:.0f} chars)")

        # Extract train/test using shared indices
        X_train_text = df[f"text_clean_{exp_name}"].values[train_idx]
        X_test_text = df[f"text_clean_{exp_name}"].values[test_idx]

        # Build pipeline
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=tfidf_max_features,
                ngram_range=tfidf_ngram_range,
                min_df=2,
                max_df=0.95,
                lowercase=False,  # Cleaning function handles this
                sublinear_tf=True
            )),
            ("clf", LogisticRegression(
                C=2.0,
                max_iter=1000,
                random_state=random_state,
                solver="lbfgs",
                multi_class="multinomial"
            ))
        ])

        # Train
        if verbose:
            print("  Training...", end=" ")
        pipeline.fit(X_train_text, y_train)
        if verbose:
            print("✓")

        # Evaluate
        if verbose:
            print("  Evaluating...", end=" ")
        y_pred = pipeline.predict(X_test_text)
        f1 = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)
        if verbose:
            print("✓")

        # Calculate delta vs baseline
        if exp_name == "baseline_raw":
            baseline_f1 = f1
            delta_f1 = 0.0
            delta_pct = 0.0
        else:
            delta_f1 = f1 - baseline_f1 if baseline_f1 else 0.0
            delta_pct = (delta_f1 / baseline_f1 * 100) if baseline_f1 else 0.0

        if verbose:
            print(f"  → F1 Score: {f1:.6f} | Accuracy: {acc:.4f}", end="")
            if exp_name != "baseline_raw":
                symbol = "🚀" if delta_f1 > 0 else "📉" if delta_f1 < 0 else "➖"
                print(f" | Δ vs baseline: {symbol} {delta_f1:+.6f} ({delta_pct:+.2f}%)")
            else:
                print(" | [BASELINE]")
            print()

        # Store result
        results.append({
            "experiment": exp_name,
            "group": exp_group,
            "f1_weighted": f1,
            "accuracy": acc,
            "delta_f1": delta_f1,
            "delta_pct": delta_pct
        })

        # Clean up temporary column to save memory
        df.drop(columns=[f"text_clean_{exp_name}"], inplace=True)

    if verbose:
        print("=" * 80)
        print("✓ BENCHMARK COMPLETED")
        print("=" * 80)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    return results_df


# =============================================================================
# Results Analysis
# =============================================================================

def analyze_results(results_df, top_n=10):
    """
    Print detailed analysis of benchmark results.

    Args:
        results_df: Results DataFrame from run_benchmark()
        top_n: Number of top/bottom experiments to highlight
    """
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS ANALYSIS")
    print("=" * 80)
    print()

    # Overall summary
    baseline = results_df[results_df["experiment"] == "baseline_raw"].iloc[0]
    print(f"Baseline F1 Score: {baseline['f1_weighted']:.6f}")
    print()

    # Top improvements
    print(f"🚀 TOP {top_n} IMPROVEMENTS:")
    print("-" * 80)
    top_improvements = results_df[results_df["experiment"] != "baseline_raw"].nlargest(top_n, "delta_f1")
    for i, row in top_improvements.iterrows():
        print(f"  {row['experiment']:30s} | F1: {row['f1_weighted']:.6f} | "
              f"Δ: {row['delta_f1']:+.6f} ({row['delta_pct']:+.2f}%) | Group: {row['group']}")
    print()

    # Bottom performers
    print(f"📉 TOP {top_n} DEGRADATIONS:")
    print("-" * 80)
    bottom_performers = results_df[results_df["experiment"] != "baseline_raw"].nsmallest(top_n, "delta_f1")
    for i, row in bottom_performers.iterrows():
        print(f"  {row['experiment']:30s} | F1: {row['f1_weighted']:.6f} | "
              f"Δ: {row['delta_f1']:+.6f} ({row['delta_pct']:+.2f}%) | Group: {row['group']}")
    print()

    # Group-wise analysis
    print("📊 GROUP-WISE SUMMARY:")
    print("-" * 80)
    group_stats = results_df.groupby("group").agg({
        "delta_f1": ["mean", "max", "min"],
        "experiment": "count"
    }).round(6)
    print(group_stats)
    print()

    print("=" * 80)


def save_results(results_df, output_path="benchmark_results.csv"):
    """
    Save benchmark results to CSV.

    Args:
        results_df: Results DataFrame
        output_path: Output file path
    """
    results_df.to_csv(output_path, index=False)
    print(f"✓ Results saved to: {output_path}")
