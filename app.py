"""PropertyPredict — a small UI over the trained solubility model.

Run:  streamlit run app.py

Built for someone who knows chemistry but not machine learning: paste a
structure, get a number with units they recognize, and see which properties
drove the answer.
"""

from __future__ import annotations

import os
import sys

import joblib
import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from pipeline import build_datasets, engineer, plain_name, rdkit_descriptors  # noqa: E402

RDLogger.DisableLog("rdApp.*")

MODEL_PATH = "models/best_model.joblib"
SURFACE, INK, INK_MUTED, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de", "#2a78d6"

EXAMPLES = {
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Glucose": "OCC1OC(O)C(O)C(O)C1O",
    "Naphthalene": "c1ccc2ccccc2c1",
    "DDT (persistent pollutant)": "Clc1ccc(C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl)cc1",
    "Ethanol": "CCO",
}

# Rough bands used in pharmacopoeias, in log mol/L.
BANDS = [
    (-1.0, "Very soluble", "Dissolves readily in water."),
    (-2.0, "Soluble", "Dissolves well enough for most aqueous work."),
    (-4.0, "Slightly soluble", "Needs a co-solvent or formulation help."),
    (-6.0, "Poorly soluble", "Largely stays out of the water phase."),
    (-99.0, "Practically insoluble", "Effectively does not dissolve; expect it to persist."),
]


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_training_targets():
    _, _, y, _, _ = build_datasets()
    return y


def featurize(smiles_list, feature_order):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    ok = [i for i, m in enumerate(mols) if m is not None]
    if not ok:
        return None, mols
    valid = pd.Series([smiles_list[i] for i in ok], index=ok)
    X = engineer(rdkit_descriptors(valid)).reindex(columns=feature_order)
    return X, mols


def band_for(logs):
    for cutoff, label, blurb in BANDS:
        if logs >= cutoff:
            return label, blurb
    return BANDS[-1][1], BANDS[-1][2]


def distribution_chart(y, value):
    fig, ax = plt.subplots(figsize=(6, 1.9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.hist(y, bins=45, color=GRID, zorder=2)
    ax.axvline(value, color=BLUE, linewidth=2.5, zorder=3)
    ax.annotate("this molecule", xy=(value, ax.get_ylim()[1] * 0.82),
                xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=BLUE, va="center")
    ax.set_yticks([])
    ax.set_xlabel("Solubility of the 1,117 training molecules (log mol/L)",
                  fontsize=9, color=INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------

st.set_page_config(page_title="PropertyPredict", page_icon="🧪", layout="centered")
st.title("🧪 PropertyPredict")
st.caption("Predicting how well a molecule dissolves in water, from its structure alone.")

bundle = load_model()
if bundle is None:
    st.error("No trained model found. Run `python src/run.py` first.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Model", bundle["model_name"])
c2.metric("Typical error", f"± {bundle['cv_rmse']:.2f} log units")
c3.metric("Trained on", f"{bundle['n_train']:,} molecules")

single, batch = st.tabs(["One molecule", "A whole list"])

with single:
    choice = st.selectbox("Start from an example", list(EXAMPLES) + ["(type my own)"])
    default = EXAMPLES.get(choice, "")
    smiles = st.text_input("SMILES structure", value=default,
                           placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")

    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("That is not a structure RDKit can read. Check the SMILES string.")
        else:
            X, _ = featurize([smiles], bundle["features"])
            pred = float(bundle["model"].predict(X)[0])
            label, blurb = band_for(pred)

            from rdkit.Chem import Descriptors
            mw = Descriptors.MolWt(mol)
            mg_per_l = (10 ** pred) * mw * 1000

            left, right = st.columns([1, 1.3])
            with left:
                st.image(Draw.MolToImage(mol, size=(280, 280)))
            with right:
                st.metric("Predicted solubility", f"{pred:.2f} log mol/L")
                st.metric("Roughly", f"{mg_per_l:,.0f} mg/L" if mg_per_l >= 1
                          else f"{mg_per_l:.3f} mg/L")
                st.markdown(f"**{label}** — {blurb}")

            lo, hi = pred - bundle["cv_rmse"], pred + bundle["cv_rmse"]
            st.caption(
                f"The model is typically within {bundle['cv_rmse']:.2f} log units, "
                f"so read this as roughly {lo:.2f} to {hi:.2f}."
            )

            st.pyplot(distribution_chart(load_training_targets(), pred))

            with st.expander("Properties this molecule has"):
                keys = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
                        "NumRotatableBonds", "RingCount", "NumAromaticRings",
                        "eng_polarity_density", "eng_hbond_total"]
                st.dataframe(
                    pd.DataFrame({
                        "Property": [plain_name(k) for k in keys],
                        "Value": [round(float(X.iloc[0][k]), 3) for k in keys],
                    }),
                    hide_index=True, width="stretch",
                )

with batch:
    st.write("One SMILES per line.")
    text = st.text_area("Structures", height=160,
                        value="\n".join(list(EXAMPLES.values())[:4]))
    if st.button("Predict all", type="primary"):
        lines = [s.strip() for s in text.splitlines() if s.strip()]
        X, mols = featurize(lines, bundle["features"])
        if X is None:
            st.error("None of those lines parsed as a structure.")
        else:
            preds = bundle["model"].predict(X)
            out = pd.DataFrame({"SMILES": [lines[i] for i in X.index]})
            out["Predicted log mol/L"] = preds.round(2)
            out["Interpretation"] = [band_for(p)[0] for p in preds]
            failed = [lines[i] for i, m in enumerate(mols) if m is None]
            st.dataframe(out, hide_index=True, width="stretch")
            if failed:
                st.warning(f"Could not read {len(failed)} line(s): {', '.join(failed)}")
            st.download_button("Download as CSV", out.to_csv(index=False),
                               "predictions.csv", "text/csv")

st.divider()
st.caption(
    "Trained on the ESOL/Delaney measured-solubility dataset. Predictions are "
    "estimates for triage and screening, not a substitute for measurement."
)
