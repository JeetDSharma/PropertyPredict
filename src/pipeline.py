"""Data loading, cleansing, featurization, and model pipeline definitions.

Task: predict aqueous solubility (log mol/L) of small molecules from their
structure. Dataset is ESOL / Delaney (1,128 measured compounds).

Everything that learns from data -- imputation, scaling, feature selection --
lives inside a scikit-learn Pipeline so that it is fit on training folds only.
Nothing here touches a test fold.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

RDLogger.DisableLog("rdApp.*")

RANDOM_STATE = 42
# Anchored to the repo root rather than the working directory, so the app
# loads the same data no matter where the process was launched from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "esol.csv")
TARGET = "measured log solubility in mols per litre"

# The six descriptors shipped with the dataset. These define the "no feature
# engineering" baseline: what you would model with straight out of the CSV.
BASELINE_FEATURES = [
    "Minimum Degree",
    "Molecular Weight",
    "Number of H-Bond Donors",
    "Number of Rings",
    "Number of Rotatable Bonds",
    "Polar Surface Area",
]

# Ipc grows factorially with molecule size and overflows to inf on larger
# structures, so it is excluded rather than imputed.
EXCLUDED_DESCRIPTORS = {"Ipc"}


# --------------------------------------------------------------------------
# Load + cleanse
# --------------------------------------------------------------------------

def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame, iqr_factor: float = 3.0) -> tuple[pd.DataFrame, dict]:
    """Cleanse the raw table and return (clean_df, report_of_what_was_removed).

    Four steps, in order:
      1. Drop the leakage column. "ESOL predicted log solubility" is another
         model's prediction of the target, published alongside it. Leaving it
         in gives near-perfect scores that mean nothing.
      2. Drop structures RDKit cannot parse.
      3. Collapse duplicate structures. The same molecule appears in this
         dataset under multiple common names (Guaiacol / o-Methoxyphenol,
         Dialifos / Dialifor, ...). The SMILES strings differ as text but
         describe the same compound, so they only match after canonicalization.
         Left in, they put the same molecule in a training fold and a test
         fold at once -- the model would be scored on a molecule it memorized.
         Duplicates are merged and their measurements averaged.
      4. Drop extreme target outliers (beyond `iqr_factor` x IQR).
    """
    report = {"rows_in": len(df)}

    df = df.drop(columns=["ESOL predicted log solubility in mols per litre"])

    before = len(df)
    df = df.assign(canonical_smiles=df["smiles"].apply(_canonical))
    df = df[df["canonical_smiles"].notna()]
    report["unparseable_smiles_removed"] = before - len(df)

    before = len(df)
    agg = {c: "first" for c in df.columns if c != "canonical_smiles"}
    agg[TARGET] = "mean"  # synonym pairs disagree slightly; average them
    df = df.groupby("canonical_smiles", as_index=False, sort=False).agg(agg)
    report["duplicate_structures_merged"] = before - len(df)

    q1, q3 = df[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
    before = len(df)
    df = df[df[TARGET].between(lo, hi)]
    report["target_outliers_removed"] = before - len(df)
    report["outlier_bounds"] = (round(lo, 2), round(hi, 2))
    report["rows_out"] = len(df)

    return df.reset_index(drop=True), report


def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


# --------------------------------------------------------------------------
# Featurize
# --------------------------------------------------------------------------

def rdkit_descriptors(smiles: pd.Series) -> pd.DataFrame:
    """Compute the full RDKit descriptor block (~210 columns) per molecule."""
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        rows.append(Descriptors.CalcMolDescriptors(mol))
    X = pd.DataFrame(rows, index=smiles.index)
    X = X.drop(columns=[c for c in EXCLUDED_DESCRIPTORS if c in X.columns])
    # A few descriptors return +/-inf on edge-case structures; mark them missing
    # and let the pipeline's imputer fill them from the training fold's median.
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.dropna(axis=1, how="all")


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    """Add ratio features motivated by how solubility actually works.

    Solubility is a competition between a molecule's polar surface (which the
    water likes) and its size and greasiness (which the water does not). Raw
    descriptors carry the numerator and denominator separately; these give the
    model the ratio directly.
    """
    X = X.copy()
    eps = 1e-6
    X["eng_polarity_density"] = X["TPSA"] / (X["MolWt"] + eps)
    X["eng_mass_per_heavy_atom"] = X["MolWt"] / (X["HeavyAtomCount"] + eps)
    X["eng_logp_per_heavy_atom"] = X["MolLogP"] / (X["HeavyAtomCount"] + eps)
    X["eng_hbond_total"] = X["NumHDonors"] + X["NumHAcceptors"]
    X["eng_aromatic_fraction"] = X["NumAromaticRings"] / (X["RingCount"] + 1)
    return X


def build_datasets(path: str = DATA_PATH):
    """Return (X_baseline, X_full, y, clean_report)."""
    df, report = clean(load_raw(path))
    y = df[TARGET].to_numpy()
    X_baseline = df[BASELINE_FEATURES].copy()
    X_full = engineer(rdkit_descriptors(df["canonical_smiles"]))
    return X_baseline, X_full, y, report, df


# --------------------------------------------------------------------------
# Feature selection (leak-free: fit inside the pipeline)
# --------------------------------------------------------------------------

class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one of every pair of features correlated above `threshold`.

    RDKit ships many near-duplicate descriptors (several molecular-weight
    variants, several surface-area variants). Dropping the redundancy shortens
    the feature matrix without losing signal. Correlations are computed on the
    training fold only.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr)
        upper = np.triu(np.abs(corr), k=1)
        drop = set()
        for i, j in zip(*np.where(upper > self.threshold)):
            if i not in drop:
                drop.add(j)
        self.keep_ = np.array([i for i in range(X.shape[1]) if i not in drop])
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]

    def get_support(self):
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.keep_] = True
        return mask


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------

def _preprocessor():
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("drop_constant", VarianceThreshold(threshold=0.0)),
        ("decorrelate", CorrelationFilter(threshold=0.95)),
        ("scale", StandardScaler()),
    ]


def baseline_pipeline() -> Pipeline:
    """Ordinary least squares on the six stock descriptors. Nothing tuned."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LinearRegression()),
    ])


def candidate_models() -> dict[str, tuple[Pipeline, dict]]:
    """Return {name: (pipeline, hyperparameter search space)}."""
    return {
        "Ridge": (
            Pipeline(_preprocessor() + [("model", Ridge(random_state=None))]),
            {
                "decorrelate__threshold": [0.90, 0.95, 0.99],
                "model__alpha": loguniform(1e-2, 1e3),
            },
        ),
        "Random forest": (
            Pipeline(_preprocessor() + [(
                "model",
                RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            )]),
            {
                "decorrelate__threshold": [0.90, 0.95, 0.99],
                "model__n_estimators": randint(200, 600),
                "model__max_depth": [None, 8, 14, 20],
                "model__min_samples_leaf": randint(1, 6),
                "model__max_features": uniform(0.2, 0.6),
            },
        ),
        "XGBoost": (
            Pipeline(_preprocessor() + [(
                "model",
                XGBRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    tree_method="hist",
                    objective="reg:squarederror",
                ),
            )]),
            {
                "decorrelate__threshold": [0.90, 0.95, 0.99],
                "model__n_estimators": randint(200, 800),
                "model__max_depth": randint(3, 9),
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.5, 0.5),
                "model__min_child_weight": randint(1, 8),
                "model__reg_lambda": loguniform(0.1, 20),
            },
        ),
    }


# --------------------------------------------------------------------------
# Plain-language descriptor names (for the non-technical report and the UI)
# --------------------------------------------------------------------------

PLAIN_NAMES = {
    "MolLogP": "Greasiness (oil/water preference)",
    "TPSA": "Polar surface area",
    "MolWt": "Molecular weight",
    "HeavyAtomMolWt": "Molecular weight (excluding hydrogens)",
    "ExactMolWt": "Exact molecular weight",
    "NumHDonors": "Hydrogen-bond donors",
    "NumHAcceptors": "Hydrogen-bond acceptors",
    "NumRotatableBonds": "Flexible bonds",
    "RingCount": "Number of rings",
    "NumAromaticRings": "Aromatic rings",
    "FractionCSP3": "Fraction of saturated carbons",
    "HeavyAtomCount": "Number of heavy atoms",
    "LabuteASA": "Molecular surface area",
    "BertzCT": "Structural complexity",
    "qed": "Drug-likeness score",
    "MolMR": "Molar refractivity (polarizability)",
    "BalabanJ": "Molecular branching index",
    "NumValenceElectrons": "Valence electron count",
    "MaxPartialCharge": "Strongest positive atomic charge",
    "MinPartialCharge": "Strongest negative atomic charge",
    "eng_polarity_density": "Polar surface per unit mass (engineered)",
    "eng_mass_per_heavy_atom": "Average atom weight (engineered)",
    "eng_logp_per_heavy_atom": "Greasiness per atom (engineered)",
    "eng_hbond_total": "Total hydrogen-bond sites (engineered)",
    "eng_aromatic_fraction": "Share of rings that are aromatic (engineered)",
}


# RDKit ships whole families of binned surface-area and connectivity
# descriptors. Naming them one by one is hopeless; naming the family is not.
_FAMILIES = [
    (r"^PEOE_VSA(\d+)$", "Surface area carrying charge, band {0}"),
    (r"^SlogP_VSA(\d+)$", "Surface area by greasiness, band {0}"),
    (r"^SMR_VSA(\d+)$", "Surface area by polarizability, band {0}"),
    (r"^EState_VSA(\d+)$", "Surface area by electronic state, band {0}"),
    (r"^VSA_EState(\d+)$", "Electronic state by surface area, band {0}"),
    (r"^Chi(\d+)[nv]$", "Branching pattern, order {0}"),
    (r"^Chi(\d+)$", "Branching pattern, order {0}"),
    (r"^Kappa(\d+)$", "Molecular shape index {0}"),
    (r"^BCUT2D_MW(LOW|HI)$", "Heaviest/lightest atom contribution"),
    (r"^BCUT2D_CHG(LO|HI)$", "Most/least charged atom contribution"),
    (r"^BCUT2D_LOGP(LOW|HI)$", "Greasiest/least greasy atom contribution"),
    (r"^BCUT2D_MR(LOW|HI)$", "Most/least polarizable atom contribution"),
    (r"^fr_(.+)$", "Count of {0} groups"),
    (r"^Num(.+)$", "Number of {0}"),
]


def plain_name(descriptor: str) -> str:
    """Translate an RDKit descriptor name into something a chemist-but-not-
    modeller can read. Falls back to the raw name rather than inventing one."""
    if descriptor in PLAIN_NAMES:
        return PLAIN_NAMES[descriptor]
    for pattern, template in _FAMILIES:
        m = re.match(pattern, descriptor)
        if m:
            arg = m.group(1).replace("_", " ").lower()
            return template.format(arg) + f" ({descriptor})"
    return descriptor
