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
from rdkit.Chem import rdDepictor

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from pipeline import build_datasets, engineer, plain_name, rdkit_descriptors  # noqa: E402

RDLogger.DisableLog("rdApp.*")

MODEL_PATH = os.path.join(REPO_ROOT, "models", "best_model.joblib")
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


# Atom colours for the structure drawing. Carbon is left implicit, as is
# conventional in a skeletal formula.
ATOM_COLORS = {
    "O": "#c0392b", "N": "#2a78d6", "S": "#b7950b", "P": "#d35400",
    "F": "#16a085", "Cl": "#16a085", "Br": "#8e44ad", "I": "#6c3483",
}


def _bond_normal(p, q, scale):
    """Unit normal to the p->q vector, scaled -- offsets parallel bond lines."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    return -dy / n * scale, dx / n * scale


def draw_molecule(mol, size=3.0):
    """Render a skeletal formula with matplotlib.

    rdkit.Chem.Draw links against X11 (libXrender), which is absent from the
    deployment image, so the structure is drawn here from the 2D coordinates
    rdDepictor produces -- that module has no such dependency.
    """
    mol = Chem.Mol(mol)
    try:
        # Kekulize so aromatic rings come back as explicit alternating bonds.
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        pass
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pos = {a.GetIdx(): (conf.GetAtomPosition(a.GetIdx()).x,
                        conf.GetAtomPosition(a.GetIdx()).y)
           for a in mol.GetAtoms()}

    fig, ax = plt.subplots(figsize=(size, size))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    labelled = {i for i, a in enumerate(mol.GetAtoms()) if a.GetSymbol() != "C"}

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        p, q = pos[i], pos[j]
        # Stop bonds short of atom labels so the text is not struck through.
        dx, dy = q[0] - p[0], q[1] - p[1]
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        ux, uy = dx / n, dy / n
        p = (p[0] + ux * (0.20 if i in labelled else 0.0),
             p[1] + uy * (0.20 if i in labelled else 0.0))
        q = (q[0] - ux * (0.20 if j in labelled else 0.0),
             q[1] - uy * (0.20 if j in labelled else 0.0))

        order = bond.GetBondTypeAsDouble()
        if order == 2:
            offsets, width = (1, -1), 1.4
        elif order == 3:
            offsets, width = (1, 0, -1), 1.3
        else:
            offsets, width = (0,), 1.6
        ox, oy = _bond_normal(p, q, 0.055 if order == 2 else 0.075)
        for s_ in offsets:
            ax.plot([p[0] + s_ * ox, q[0] + s_ * ox],
                    [p[1] + s_ * oy, q[1] + s_ * oy],
                    color=INK, linewidth=width, solid_capstyle="round", zorder=2)

    for idx in labelled:
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        h = atom.GetTotalNumHs()
        label = sym + ("H" if h == 1 else f"H{h}" if h > 1 else "")
        x, y = pos[idx]
        ax.text(x, y, label, ha="center", va="center", fontsize=11,
                color=ATOM_COLORS.get(sym, INK), zorder=3,
                bbox=dict(boxstyle="round,pad=0.12", facecolor=SURFACE,
                          edgecolor="none"))

    xs = [c[0] for c in pos.values()]
    ys = [c[1] for c in pos.values()]
    pad = 0.6
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    return fig


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
                structure = draw_molecule(mol)
                st.pyplot(structure)
                plt.close(structure)
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

            dist = distribution_chart(load_training_targets(), pred)
            st.pyplot(dist)
            plt.close(dist)

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
