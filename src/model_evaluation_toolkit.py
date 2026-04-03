"""
model_evaluation_toolkit
=======================
Model Training, Evaluation & Comparison Toolkit
-------------------------------------------------
A curated collection of functions for safely training, tuning, evaluating,
and statistically comparing binary classification pipelines in Jupyter notebooks.

All functions produce publication-ready styled HTML tables and inline
Base64-encoded figures using a consistent visual language aligned with
``eda_toolkit.py``.

Visual Language
---------------
- Colour palette  : matplotlib ``tab10`` (multi-model ROC curves)
- Warm heatmap    : ``YlOrRd`` (|ΔAUC| lower triangle)
- White upper triangle for p-values
- Bold axis labels, tight layout, 100 dpi default output

Pipeline Architecture
---------------------
All pipelines follow the same leakage-free preprocessing structure:

    Raw data
        → OneHotEncoder (categorical columns, ``remainder='passthrough'``)
        → IterativeImputer / MICE (random_state=42)
        → RobustScaler (numeric columns by position)
        → Estimator

This guarantees **zero data leakage**, as all transformations are fit
exclusively on training data.

Language Support
----------------
Most functions include a ``spanish`` parameter for bilingual outputs:

    spanish=False → English (default)
    spanish=True  → Spanish

This affects:
- Metric names
- Table headers
- Plot labels

Dependencies
------------
    pip install scikit-learn pandas numpy matplotlib scipy statsmodels

Functions
---------
Model Training & Selection
    train_model
        Hyperparameter tuning using Stratified Cross-Validation with a
        Top-5 ranked configuration table.

Model Evaluation
    evaluate_model
        Train a final pipeline with fixed hyperparameters and report
        Train vs Test metrics (with overfitting diagnostic).

    compare_models_table
        Compare multiple fitted models on a shared test set.
        Outputs a ranked table sorted by ROC-AUC (descending) with
        Accuracy as tie-breaker.

    compare_leakage_vs_clean
        Didactic comparison between a leaked and a clean pipeline to
        quantify performance inflation due to data leakage.

Visualisation
    plot_roc_curves
        Overlay ROC curves for multiple models using a consistent
        visual style.

Statistical Comparison
    _delong_roc_variance
        Internal helper for computing AUC variance (DeLong method).

    delong_roc_test
        Pairwise statistical test for comparing ROC-AUC values.

    plot_delong_matrix
        Hybrid heatmap showing:
            - Lower triangle → |ΔAUC|
            - Upper triangle → corrected p-values (Holm method)

        Supports bilingual labeling.

Notes
-----
- All pipelines are designed to be **reproducible and leakage-free**.
- Visual outputs are optimised for **academic publication**.
- Designed to integrate seamlessly into **EDA + modeling workflows**.
"""

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
__version__ = "1.0.0"

__author__ = (
    "Paula Andrea Gómez Vargas <apaulag@uninorte.edu.co>, "
    "Juan Camilo Mendoza Arango <cjarango@uninorte.edu.co>, "
    "Miguel Ángel Pérez Vargas <vargasmiguel@uninorte.edu.co>"
)


# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import io
import base64

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
)
from statsmodels.stats.multitest import multipletests
from IPython.display import display, HTML

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    "train_model",
    "evaluate_model",
    "compare_leakage_vs_clean",
    "plot_roc_curves",
    "delong_roc_test",
    "plot_delong_matrix",
    "compare_models_table",
]


# ---------------------------------------------------------------------------
# Constants — column groups (mirror your project's feature schema)
# ---------------------------------------------------------------------------

NUMERIC_COLS: list[str] = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CATEGORICAL_COLS: list[str] = ["Sex", "ChestPainType", "RestingECG",
                                "ExerciseAngina", "ST_Slope"]

_TABLE_STYLES = [
    {"selector": "",   "props": [("margin-left", "auto"), ("margin-right", "auto"),
                                  ("width", "auto")]},
    {"selector": "th", "props": [("background-color", "#f2f2f2"), ("color", "black"),
                                  ("border", "1px solid black"), ("padding", "10px"),
                                  ("text-align", "center")]},
    {"selector": "td", "props": [("border", "1px solid black"), ("padding", "10px"),
                                  ("text-align", "center")]},
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render_figure(fig: plt.Figure, dpi: int = 100) -> None:
    """
    Encode *fig* as a Base64 PNG and display it as centred inline HTML.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to serialise and display.
    dpi : int, optional
        Export resolution (default ``100``).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    encoded = base64.b64encode(buf.getbuffer()).decode("ascii")
    display(HTML(
        f'<div style="text-align: center; width: 100%;">'
        f'<img src="data:image/png;base64,{encoded}"></div>'
    ))


# ===========================================================================
# 1. MODEL TRAINING & EVALUATION
# ===========================================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    estimador,
    malla_parametros: dict,
    nombre_paso: str = "model",
    scoring: str = "roc_auc",
):
    """
    Perform hyperparameter tuning using stratified cross-validation and
    display a ranked table of the top 5 configurations.

    This function builds a leakage-free sklearn Pipeline following the
    project's standard preprocessing architecture:

        OneHotEncoder → IterativeImputer → RobustScaler → estimator

    A GridSearchCV is applied with StratifiedKFold (10 splits), and the
    best configurations are displayed as a styled HTML table.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training labels (binary).
    estimador : sklearn estimator
        Model to tune (must support sklearn interface).
    malla_parametros : dict
        Parameter grid using sklearn's ``step__param`` convention.
    nombre_paso : str, optional
        Name of the estimator step inside the pipeline (default ``"model"``).
    scoring : str, optional
        Scoring metric for GridSearchCV (default ``"roc_auc"``).

    Returns
    -------
    grid_search : sklearn.model_selection.GridSearchCV
        Fitted GridSearch object containing all results.

    Notes
    -----
    - Uses **StratifiedKFold (n=10)** for robust performance estimation.
    - Outputs a **Top-5 ranking table** with mean ± std scores.
    - Designed for exploratory model selection prior to final evaluation.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> grid = train_model(
    ...     X_train, y_train,
    ...     SVC(probability=True),
    ...     {"model__C": [0.1, 1, 10]},
    ... )
    """
    # ------------------------------------------------------------------
    # 1. Pipeline (consistent with module standard)
    # ------------------------------------------------------------------
    pipeline = Pipeline([
        ("encoding", ColumnTransformer([
            ("onehot", OneHotEncoder(handle_unknown="ignore",
                                     sparse_output=False), CATEGORICAL_COLS)
        ], remainder="passthrough")),
        ("imputer", IterativeImputer(random_state=42)),
        ("scaler", ColumnTransformer([
            ("robust", RobustScaler(),
             list(range(-len(NUMERIC_COLS), 0)))
        ], remainder="passthrough")),
        (nombre_paso, estimador),
    ])

    # ------------------------------------------------------------------
    # 2. Stratified Cross-Validation
    # ------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # ------------------------------------------------------------------
    # 3. Grid Search
    # ------------------------------------------------------------------
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=malla_parametros,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    grid_search.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 4. Results processing
    # ------------------------------------------------------------------
    resultados = pd.DataFrame(grid_search.cv_results_)

    params_df = pd.json_normalize(resultados["params"])
    params_df.columns = [col.split("__")[-1] for col in params_df.columns]

    metric_name = scoring.upper().replace("_", "-")

    resultados[metric_name] = resultados.apply(
        lambda x: f"{x['mean_test_score']:.3f} ± {x['std_test_score']:.3f}",
        axis=1
    )

    df_ranking = pd.concat(
        [params_df, resultados[[metric_name, "rank_test_score"]]],
        axis=1
    )

    df_top5 = (
        df_ranking
        .sort_values("rank_test_score")
        .head(5)
        .drop(columns=["rank_test_score"])
    )

    # ------------------------------------------------------------------
    # 5. Styled output (consistent visual language)
    # ------------------------------------------------------------------
    styled = (
        df_top5.style
        .hide(axis="index")
        .set_table_styles(_TABLE_STYLES)
    )

    display(HTML(
        "<div style='text-align: center; width: 100%; margin-top: 10px;'>"
        + styled.to_html()
        + "</div>"
    ))

    return grid_search

def evaluate_model(
    estimator,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    alert_threshold: float = 0.03,
    spanish: bool = False,
) -> Pipeline:
    """
    Train a production pipeline once with fixed hyperparameters and evaluate performance.

    Constructs a leakage-free ``sklearn.Pipeline`` that applies:

        encoding → imputation → scaling → estimator

    ensuring that all transformations are fitted exclusively on training data.

    A styled HTML table is displayed with five evaluation metrics for both
    training and test sets. Differences exceeding *alert_threshold* are flagged
    with a dagger (†) as a visual overfitting warning.

    Parameters
    ----------
    estimator : sklearn estimator
        An unfitted classifier that supports ``predict_proba`` 
        (e.g. ``SVC(probability=True)``, ``LogisticRegression()``).

    params : dict
        Hyperparameter dictionary. Prefixes are automatically mapped to
        ``model__*`` inside the pipeline.

    X_train : pd.DataFrame
        Training feature matrix.

    y_train : pd.Series
        Training labels (binary 0/1).

    X_test : pd.DataFrame
        Test feature matrix.

    y_test : pd.Series
        Test labels (binary 0/1).

    alert_threshold : float, optional
        Absolute Train–Test difference above which a metric is flagged
        with † (default ``0.03``).

    spanish : bool, optional
        If True, displays metric names and table labels in Spanish.
        Default is ``False``.

    Returns
    -------
    pipeline : sklearn.Pipeline
        Fully fitted pipeline ready for inference via ``predict`` or
        ``predict_proba``.

    Notes
    -----
    Pipeline structure::

        OneHotEncoder → IterativeImputer → RobustScaler → estimator

    Numeric columns are scaled using negative indexing (last *n* columns),
    making the pipeline robust to varying numbers of encoded features.

    Language behaviour
    ------------------
    - ``spanish=False`` → English labels (default)
    - ``spanish=True``  → Spanish labels:

        Accuracy   → Exactitud  
        Precision  → Precisión  
        Recall     → Sensibilidad  
        Gap        → Diferencia  

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> pipeline = evaluate_model(
    ...     SVC(probability=True),
    ...     params={"svc__C": 1.0, "svc__kernel": "rbf"},
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test,  y_test=y_test,
    ... )

    >>> # Spanish output
    >>> pipeline = evaluate_model(
    ...     SVC(probability=True),
    ...     params={"svc__C": 1.0},
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test,  y_test=y_test,
    ...     spanish=True
    ... )
    """
    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    pipeline = Pipeline([
        ("encoding", ColumnTransformer([
            ("onehot", OneHotEncoder(handle_unknown="ignore",
                                     sparse_output=False), CATEGORICAL_COLS)
        ], remainder="passthrough")),
        ("imputer", IterativeImputer(random_state=42)),
        ("scaler", ColumnTransformer([
            ("robust", RobustScaler(),
             list(range(-len(NUMERIC_COLS), 0)))
        ], remainder="passthrough")),
        ("model", estimator),
    ])

    clean_params = {f"model__{k.split('__')[-1]}": v for k, v in params.items()}
    pipeline.set_params(**clean_params)

    pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    y_tr_pred = pipeline.predict(X_train)
    y_tr_prob = pipeline.predict_proba(X_train)[:, 1]

    y_te_pred = pipeline.predict(X_test)
    y_te_prob = pipeline.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------
    # Language configuration
    # ------------------------------------------------------------------
    if spanish:
        metric_names = ["Exactitud", "Precisión", "Sensibilidad", "F1-Score", "ROC-AUC"]
        col_train = "Entrenamiento"
        col_test  = "Prueba"
        col_gap   = "Diferencia"
    else:
        metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        col_train = "Train"
        col_test  = "Test"
        col_gap   = "Gap"

    metric_fns = [
        (metric_names[0], accuracy_score,  y_tr_pred, y_te_pred),
        (metric_names[1], precision_score, y_tr_pred, y_te_pred),
        (metric_names[2], recall_score,    y_tr_pred, y_te_pred),
        (metric_names[3], f1_score,        y_tr_pred, y_te_pred),
        (metric_names[4], roc_auc_score,   y_tr_prob, y_te_prob),
    ]

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    rows = []
    for name, fn, tr_arg, te_arg in metric_fns:
        tr_val = fn(y_train, tr_arg)
        te_val = fn(y_test,  te_arg)
        rows.append({
            "Metric": name,
            col_train: tr_val,
            col_test: te_val
        })

    df_summary = pd.DataFrame(rows)
    df_summary[col_gap] = (df_summary[col_train] - df_summary[col_test]).abs()

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def _flag_gap(val: float) -> str:
        return f"{val:.3f}†" if val > alert_threshold else f"{val:.3f}"

    styled = (
        df_summary.style
        .hide(axis="index")
        .format({
            col_train: "{:.3f}",
            col_test: "{:.3f}",
            col_gap: _flag_gap
        })
        .set_table_styles(_TABLE_STYLES)
    )

    display(HTML(
        "<div style='text-align: center; width: 100%; margin-top: 10px;'>"
        + styled.to_html()
        + "</div>"
    ))

    return pipeline


# ===========================================================================
# 2. LEAKAGE vs. CLEAN PIPELINE COMPARISON
# ===========================================================================

def compare_leakage_vs_clean(
    X_train_leaked: pd.DataFrame,
    X_test_leaked: pd.DataFrame,
    y_train_leaked: pd.Series,
    y_test_leaked: pd.Series,
    params_leaked: dict,
    X_train_clean: pd.DataFrame,
    X_test_clean: pd.DataFrame,
    y_train_clean: pd.Series,
    y_test_clean: pd.Series,
    params_clean: dict,
    spanish: bool = True,
) -> None:
    """
    Train and evaluate two SVC pipelines — one with data leakage, one without —
    and display a side-by-side metric comparison table.

    This function is designed as a **didactic diagnostic tool** to quantify
    the optimistic bias introduced when preprocessing is incorrectly applied
    using information from both training and test sets.

    The *leaked* pipeline intentionally fits the scaler on the combined
    dataset (train ∪ test), introducing data leakage.

    The *clean* pipeline follows the correct, leakage-free architecture
    (see :func:`evaluate_model`).

    Parameters
    ----------
    X_train_leaked, X_test_leaked : pd.DataFrame
        Feature matrices for the leaked experiment. Must include
        the extra ``"leaky_feature"`` column.
    y_train_leaked, y_test_leaked : pd.Series
        Labels for the leaked experiment.
    params_leaked : dict
        Hyperparameters for the leaked SVC pipeline.
    X_train_clean, X_test_clean : pd.DataFrame
        Feature matrices for the clean experiment.
    y_train_clean, y_test_clean : pd.Series
        Labels for the clean experiment.
    params_clean : dict
        Hyperparameters for the clean SVC pipeline.
    spanish : bool, optional
        If True, display metric names and table headers in Spanish.
        Default is True.

    Notes
    -----
    The leaked model is intentionally incorrect and should **never** be used
    in production. It is included only for educational purposes.

    The "Difference" column is computed as ``Leaked − Clean``.
    Positive values indicate inflated performance due to leakage.

    Examples
    --------
    >>> compare_leakage_vs_clean(
    ...     X_train_l, X_test_l, y_train_l, y_test_l, params_leaked,
    ...     X_train_c, X_test_c, y_train_c, y_test_c, params_clean,
    ...     spanish=True
    ... )
    """

    # ------------------------------------------------------------------
    # Translation dictionaries
    # ------------------------------------------------------------------
    metric_translation = {
        "Accuracy": "Exactitud",
        "Precision": "Precisión",
        "Recall": "Sensibilidad",
        "F1-Score": "F1-Score",
        "ROC-AUC": "ROC-AUC",
    }

    column_translation = {
        "Metric": "Métrica",
        "Leaked Model": "Modelo con Fuga",
        "Clean Model": "Modelo Limpio",
        "Difference": "Diferencia",
    }

    # ------------------------------------------------------------------
    # 1. Leaked pipeline
    # ------------------------------------------------------------------
    numeric_leaked = NUMERIC_COLS + ["leaky_feature"]

    scaler_leaked = ColumnTransformer([
        ("robust", RobustScaler(), numeric_leaked)
    ], remainder="passthrough")

    scaler_leaked.fit(pd.concat([X_train_leaked, X_test_leaked]))

    passthrough_cols = [c for c in X_train_leaked.columns if c not in numeric_leaked]
    all_cols = numeric_leaked + passthrough_cols

    X_tr_l = pd.DataFrame(scaler_leaked.transform(X_train_leaked), columns=all_cols)
    X_te_l = pd.DataFrame(scaler_leaked.transform(X_test_leaked), columns=all_cols)

    pipe_leaked = Pipeline([
        ("encoding", ColumnTransformer([
            ("onehot", OneHotEncoder(handle_unknown="ignore",
                                    sparse_output=False), CATEGORICAL_COLS)
        ], remainder="passthrough")),
        ("imputer", IterativeImputer(random_state=42)),
        ("svc",     SVC(probability=True)),
    ])

    pipe_leaked.set_params(**params_leaked)
    pipe_leaked.fit(X_tr_l, y_train_leaked)

    y_pred_l = pipe_leaked.predict(X_te_l)
    y_prob_l = pipe_leaked.predict_proba(X_te_l)[:, 1]

    # ------------------------------------------------------------------
    # 2. Clean pipeline
    # ------------------------------------------------------------------
    pipe_clean = Pipeline([
        ("encoding", ColumnTransformer([
            ("onehot", OneHotEncoder(handle_unknown="ignore",
                                    sparse_output=False), CATEGORICAL_COLS)
        ], remainder="passthrough")),
        ("imputer", IterativeImputer(random_state=42)),
        ("scaler", ColumnTransformer([
            ("robust", RobustScaler(), list(range(-len(NUMERIC_COLS), 0)))
        ], remainder="passthrough")),
        ("svc", SVC(probability=True)),
    ])

    pipe_clean.set_params(**params_clean)
    pipe_clean.fit(X_train_clean, y_train_clean)

    y_pred_c = pipe_clean.predict(X_test_clean)
    y_prob_c = pipe_clean.predict_proba(X_test_clean)[:, 1]

    # ------------------------------------------------------------------
    # 3. Metrics
    # ------------------------------------------------------------------
    metric_specs = [
        ("Accuracy",  accuracy_score,  False),
        ("Precision", precision_score, False),
        ("Recall",    recall_score,    False),
        ("F1-Score",  f1_score,        False),
        ("ROC-AUC",   roc_auc_score,   True),
    ]

    rows = []
    for name, fn, use_proba in metric_specs:
        val_l = fn(y_test_leaked, y_prob_l if use_proba else y_pred_l)
        val_c = fn(y_test_clean,  y_prob_c if use_proba else y_pred_c)

        metric_name = metric_translation[name] if spanish else name

        rows.append({
            "Metric": metric_name,
            "Leaked Model": f"{val_l:.3f}",
            "Clean Model":  f"{val_c:.3f}",
            "Difference":   f"{val_l - val_c:.3f}",
        })

    df_comparison = pd.DataFrame(rows)

    # Translate column names
    if spanish:
        df_comparison.rename(columns=column_translation, inplace=True)

    # ------------------------------------------------------------------
    # 4. Styling
    # ------------------------------------------------------------------
    styled = (
        df_comparison.style
        .hide(axis="index")
        .set_table_styles(_TABLE_STYLES)
    )

    display(HTML(
        "<div style='text-align: center; width: 100%; margin-top: 10px;'>"
        + styled.to_html()
        + "</div>"
    ))


def compare_models_table(
    models: list[tuple[str, Pipeline]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    spanish: bool = False,
) -> None:
    """
    Compare multiple fitted models on a common test set and display
    a styled performance table sorted by ROC-AUC.

    Models are ranked by:
        1. ROC-AUC (descending)
        2. Accuracy (descending, tie-breaker)

    Parameters
    ----------
    models : list of (str, Pipeline)
        List of tuples containing model name and fitted pipeline.

    X_test : pd.DataFrame
        Test feature matrix.

    y_test : pd.Series
        True binary labels.

    spanish : bool, optional
        If True, displays metric names in Spanish.

    Notes
    -----
    Metrics computed:
        - Accuracy
        - Precision
        - Recall
        - F1-Score
        - ROC-AUC
    """

    # ------------------------------------------------------------------
    # Language configuration
    # ------------------------------------------------------------------
    if spanish:
        metric_names = ["Exactitud", "Precisión", "Sensibilidad", "F1-Score", "ROC-AUC"]
        model_col = "Modelo"
        sort_auc = "ROC-AUC"
        sort_acc = "Exactitud"
    else:
        metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        model_col = "Model"
        sort_auc = "ROC-AUC"
        sort_acc = "Accuracy"

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    rows = []

    for name, pipeline in models:

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        row = {
            model_col: name,
            metric_names[0]: accuracy_score(y_test, y_pred),
            metric_names[1]: precision_score(y_test, y_pred),
            metric_names[2]: recall_score(y_test, y_pred),
            metric_names[3]: f1_score(y_test, y_pred),
            metric_names[4]: roc_auc_score(y_test, y_prob),
        }

        rows.append(row)

    df_results = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Sorting (key part)
    # ------------------------------------------------------------------
    df_results = df_results.sort_values(
        by=[sort_auc, sort_acc],
        ascending=[False, False]
    )

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    styled = (
        df_results.style
        .format({col: "{:.3f}" for col in df_results.columns if col != model_col})
        .hide(axis="index")
        .set_table_styles(_TABLE_STYLES)
    )

    display(HTML(
        "<div style='text-align: center; width: 100%; margin-top: 10px;'>"
        + styled.to_html()
        + "</div>"
    ))

# ===========================================================================
# 3. ROC CURVE OVERLAY
# ===========================================================================

def plot_roc_curves(
    models: list[tuple[str, Pipeline]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    chart_title: str | None = None,
    legend_title: str = "Model",
    show_legend: bool = True,
    tick_size: int = 11,
    label_size: int = 12,
    label_pad: int = 12,
    legend_size: int = 11,
    legend_title_size: int = 12,
    spanish: bool = False,
) -> None:
    """
    Overlay ROC curves for multiple fitted pipelines on a single axes.

    Each curve is computed from ``predict_proba`` using the positive class
    probabilities. Colours follow matplotlib's ``tab10`` palette for
    consistent multi-model visualization.

    Parameters
    ----------
    models : list of (str, Pipeline)
        List of fitted pipelines with display names.

    X_test : pd.DataFrame
        Test feature matrix.

    y_test : pd.Series
        True binary labels.

    chart_title : str, optional
        Title of the plot. Omitted if ``None``.

    legend_title : str, optional
        Title of the legend (default ``"Model"``).

    show_legend : bool, optional
        If True, displays the legend. If False, hides it.
        Default is ``True``.

    tick_size : int, optional
        Font size for both x and y axis ticks (default ``11``).

    label_size : int, optional
        Font size for axis labels (default ``12``).

    label_pad : int, optional
        Padding for axis labels (default ``12``).

    legend_size : int, optional
        Font size for legend labels (default ``11``).

    legend_title_size : int, optional
        Font size for legend title (default ``12``).

    spanish : bool, optional
        If True, displays axis labels and legend title in Spanish.
        Default is ``False``.

    Notes
    -----
    Language behaviour:

    - ``spanish=False`` → English labels
    - ``spanish=True``  → Spanish labels

    Examples
    --------
    >>> plot_roc_curves(
    ...     [("SVC", pipe_svc), ("LR", pipe_lr)],
    ...     X_test, y_test,
    ...     chart_title="ROC Comparison",
    ...     show_legend=True,
    ...     spanish=True
    ... )
    """
    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    colors = plt.cm.tab10.colors

    # ------------------------------------------------------------------
    # Plot ROC curves
    # ------------------------------------------------------------------
    for i, (name, pipeline) in enumerate(models):
        y_probs = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)

        ax.plot(
            fpr, tpr,
            lw=2.5,
            label=name,
            color=colors[i % len(colors)],
            alpha=0.9
        )

    # ------------------------------------------------------------------
    # Language handling
    # ------------------------------------------------------------------
    if spanish:
        xlabel = "Tasa de falsos positivos (1 − Especificidad)"
        ylabel = "Tasa de verdaderos positivos (Sensibilidad)"
        legend_title_final = "Modelo" if legend_title == "Model" else legend_title
    else:
        xlabel = "False Positive Rate (1 − Specificity)"
        ylabel = "True Positive Rate (Sensitivity)"
        legend_title_final = legend_title

    # ------------------------------------------------------------------
    # Titles and labels
    # ------------------------------------------------------------------
    if chart_title:
        ax.set_title(chart_title, fontsize=label_size + 2, fontweight="bold", pad=20)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    ax.set_xlabel(xlabel, fontsize=label_size, fontweight="bold", labelpad=label_pad)
    ax.set_ylabel(ylabel, fontsize=label_size, fontweight="bold", labelpad=label_pad)

    # Tick control
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    # ------------------------------------------------------------------
    # Legend control
    # ------------------------------------------------------------------
    if show_legend:
        ax.legend(
            title=legend_title_final,
            loc="lower right",
            frameon=False,
            fontsize=legend_size,
            title_fontsize=legend_title_size,
            labelspacing=0.7
        )

    # ------------------------------------------------------------------
    # Grid and layout
    # ------------------------------------------------------------------
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    _render_figure(fig)


# ===========================================================================
# 4. DELONG'S TEST — PAIRWISE AUC COMPARISON
# ===========================================================================

def _delong_roc_variance(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute AUC and its variance via structural components (DeLong 1988).

    This is a private helper used by :func:`delong_roc_test`.  It ranks
    predictions and derives the V₁₀ and V₀₁ structural components needed
    for the variance and covariance estimates.

    Parameters
    ----------
    ground_truth : np.ndarray, shape (n,)
        True binary labels.
    predictions : np.ndarray, shape (n,)
        Predicted probabilities for the positive class.

    Returns
    -------
    auc : float
        Area under the ROC curve (equivalent to Mann-Whitney U statistic).
    var_auc : float
        Estimated variance of the AUC.
    v10 : np.ndarray
        Structural component for positive-class observations.
    v01 : np.ndarray
        Structural component for negative-class observations.

    References
    ----------
    DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988).
    Comparing the areas under two or more correlated receiver operating
    characteristic curves: a nonparametric approach.
    *Biometrics*, 44(3), 837–845.
    """
    order        = np.argsort(predictions)
    ground_truth = ground_truth[order]
    predictions  = predictions[order]

    n_pos = int(np.sum(ground_truth))
    n_neg = len(ground_truth) - n_pos

    ranks = stats.rankdata(predictions)

    v10 = (
        ranks[ground_truth == 1]
        - stats.rankdata(predictions[ground_truth == 1])
    ) / n_neg

    v01 = (
        n_pos + 1
        - (ranks[ground_truth == 0]
           - stats.rankdata(predictions[ground_truth == 0]))
    ) / n_pos

    auc     = float(np.mean(v10))
    var_auc = (np.var(v10, ddof=1) / n_pos) + (np.var(v01, ddof=1) / n_neg)

    return auc, var_auc, v10, v01


def delong_roc_test(
    ground_truth,
    probas_model1,
    probas_model2,
) -> float:
    """
    Pairwise DeLong test for the equality of two AUC values.

    Applies the non-parametric method of DeLong et al. (1988) to test
    H₀: AUC₁ = AUC₂ against H₁: AUC₁ ≠ AUC₂ (two-tailed).

    Both classifiers must have been evaluated on the **same** test set so
    that their predictions are paired and the covariance between the two
    AUC estimators can be estimated.

    Parameters
    ----------
    ground_truth : array-like, shape (n,)
        True binary labels (0 or 1).
    probas_model1 : array-like, shape (n,)
        Predicted probabilities for the positive class from model 1.
    probas_model2 : array-like, shape (n,)
        Predicted probabilities for the positive class from model 2.

    Returns
    -------
    p_value : float
        Two-tailed p-value for H₀: AUC₁ = AUC₂.
        Returns ``1.0`` if both models produce identical predictions.

    References
    ----------
    DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988).
    Comparing the areas under two or more correlated receiver operating
    characteristic curves: a nonparametric approach.
    *Biometrics*, 44(3), 837–845.

    Examples
    --------
    >>> p = delong_roc_test(y_test,
    ...                     pipe_a.predict_proba(X_test)[:, 1],
    ...                     pipe_b.predict_proba(X_test)[:, 1])
    >>> print(f"DeLong p-value: {p:.4f}")
    """
    gt = np.asarray(ground_truth)
    p1 = np.asarray(probas_model1)
    p2 = np.asarray(probas_model2)

    auc1, var1, v10_1, v01_1 = _delong_roc_variance(gt, p1)
    auc2, var2, v10_2, v01_2 = _delong_roc_variance(gt, p2)

    cov12 = (
        np.cov(v10_1, v10_2, ddof=1)[0, 1] / len(v10_1)
        + np.cov(v01_1, v01_2, ddof=1)[0, 1] / len(v01_1)
    )

    denom = np.sqrt(var1 + var2 - 2 * cov12)
    if denom == 0:
        return 1.0  # Identical models — H₀ cannot be rejected

    z       = (auc1 - auc2) / denom
    p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))

    return float(p_value)


# ===========================================================================
# 5. DELONG PAIRWISE MATRIX — HYBRID HEATMAP
# ===========================================================================

def plot_delong_matrix(
    models: list[tuple[str, Pipeline]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    alpha: float = 0.05,
    figsize: tuple[int, int] = (10, 10),
    chart_title: str | None = None,
    spanish: bool = False,
) -> None:
    """
    Hybrid heatmap comparing all pairs of models via DeLong's AUC test.

    The matrix is divided into two triangles:

    * Lower triangle — absolute AUC difference |ΔAUC|
    * Upper triangle — Holm-corrected p-values

    Parameters
    ----------
    spanish : bool, optional
        If True, displays labels in Spanish.
    """
    names    = [m[0] for m in models]
    n_models = len(names)

    # ------------------------------------------------------------------
    # Collect predicted probabilities and AUCs
    # ------------------------------------------------------------------
    probas = {name: pipe.predict_proba(X_test)[:, 1] for name, pipe in models}
    aucs   = {name: roc_auc_score(y_test, prob) for name, prob in probas.items()}

    # ------------------------------------------------------------------
    # Pairwise DeLong tests
    # ------------------------------------------------------------------
    y_true = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)

    pair_results = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            m1, m2 = names[i], names[j]
            p_val  = delong_roc_test(y_true, probas[m1], probas[m2])
            diff   = abs(aucs[m1] - aucs[m2])
            pair_results.append({"M1": m1, "M2": m2, "p_raw": p_val, "delta_auc": diff})

    df_pairs = pd.DataFrame(pair_results)

    # ------------------------------------------------------------------
    # Holm correction
    # ------------------------------------------------------------------
    df_pairs["p_corrected"] = multipletests(df_pairs["p_raw"], method="holm")[1]

    # ------------------------------------------------------------------
    # Build matrices
    # ------------------------------------------------------------------
    heat_matrix  = pd.DataFrame(np.nan, index=names, columns=names)
    annot_matrix = pd.DataFrame("",     index=names, columns=names)

    for _, row in df_pairs.iterrows():
        p_fin  = row["p_corrected"]
        d_auc  = row["delta_auc"]
        star   = "*" if p_fin < alpha else ""
        p_text = f"{p_fin:.3f}{star}" if p_fin >= 0.001 else f"<0.001{star}"

        heat_matrix.loc[row["M2"], row["M1"]]  = d_auc
        annot_matrix.loc[row["M2"], row["M1"]] = f"{d_auc:.3f}"
        annot_matrix.loc[row["M1"], row["M2"]] = p_text

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    mask_upper = np.triu(np.ones_like(heat_matrix, dtype=bool), k=1)

    sns.heatmap(
        heat_matrix,
        mask=mask_upper,
        annot=annot_matrix,
        fmt="",
        cmap="YlOrRd",
        cbar_kws={"shrink": 0.7, "pad": 0.05},
        square=True,
        linewidths=1.5,
        linecolor="white",
        ax=ax,
        annot_kws={"fontweight": "bold", "fontsize": 10},
    )

    # Upper triangle overlay (p-values)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            rect = plt.Rectangle((j, i), 1, 1,
                                 fill=True, facecolor="white", edgecolor="none", zorder=0)
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.5,
                    annot_matrix.iloc[i, j],
                    ha="center", va="center",
                    fontsize=10)

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)

    # Colorbar label (Spanish support)
    cbar = ax.collections[0].colorbar
    if spanish:
        label = "Diferencia absoluta de AUC (|ΔAUC|)"
    else:
        label = "Absolute AUC Difference (|ΔAUC|)"

    cbar.set_label(label, labelpad=25, fontweight="bold", fontsize=13)

    if chart_title:
        ax.set_title(chart_title, fontsize=14, fontweight="bold", pad=25)

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=13)

    plt.tight_layout()
    _render_figure(fig, dpi = 150)