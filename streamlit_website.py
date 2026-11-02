from pathlib import Path
from pickle import load

import pandas as pd
import streamlit as st
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from src.featurize import featurize_single

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
DNA_BASES = set("ACGTU")

EC_CLASSES = {
    "Oxidoreductases": "Catalyze oxidation-reduction (electron transfer) reactions.",
    "Transferases": "Transfer a functional group from one molecule to another.",
    "Hydrolases": "Break chemical bonds using water (hydrolysis).",
    "Lyases": "Break bonds without hydrolysis or oxidation, often forming a double bond or ring.",
    "Isomerases": "Rearrange atoms within a single molecule to form an isomer.",
    "Ligases": "Join two molecules together, typically using ATP.",
    "Translocases": "Move molecules or ions across membranes.",
}

EXAMPLE_SEQUENCES = {
    "— pick an example —": "",
    "Lysozyme C (Hydrolase)": (
        "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSR"
        "WWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
    ),
    "Triosephosphate isomerase (Isomerase)": (
        "APSRKFFVGGNWKMNGRKQSLGELIGTLNAAKVPADTEVVCAPPTAYIDFARQKLDPKIAV"
        "AAQNCYKVTNGAFTGEISPGMIKDCGATWVVLGHSERRHVFGESDELIGQKVAHALAEGLGVIACIGEK"
        "LDEREAGITEKVVFEQTKVIADNVKDWSKVVLAYEPVWAIGTGKTATPQQAQEVHEKLRGWLKSNVSD"
        "AVAQSTRIIYGGSVTGATCKELASQPDVDGFLVGGASLKPEFVDIINAKQ"
    ),
}


@st.cache_resource
def load_model(model_name: str):
    with open(MODELS_DIR / f"{model_name}.pkl", "rb") as file:
        return load(file)


@st.cache_resource
def load_scaler():
    with open(MODELS_DIR / "scaler.pkl", "rb") as file:
        return load(file)


@st.cache_resource
def available_models() -> list:
    return sorted(p.stem for p in MODELS_DIR.glob("*.pkl") if p.stem != "scaler")


def clean_sequence(raw: str) -> str:
    return "".join(raw.split()).upper()


def validate_protein(sequence: str) -> str | None:
    if not sequence:
        return "Enter a sequence first."
    invalid = sorted(set(sequence) - AMINO_ACIDS)
    if invalid:
        return f"Not a recognized amino acid sequence — unexpected character(s): {', '.join(invalid)}"
    if len(sequence) < 10:
        return "Sequence is too short to featurize reliably (10+ residues recommended)."
    return None


st.set_page_config(page_title="Enzyme Class Predictor", page_icon="🧬", layout="centered")

st.title("🧬 Enzyme Class Predictor")
st.markdown(
    "Paste a protein (or DNA) sequence and this app predicts which of the "
    "**7 top-level Enzyme Commission (EC) classes** it most likely belongs to, "
    "using a classifier trained on sequence-derived features — amino acid "
    "composition, molecular weight, hydrophobicity (GRAVY), predicted secondary "
    "structure, and 3-mer frequencies."
)

models = available_models()
if not models:
    st.error(
        "No trained models found in `models/`. Run `python -m src.train` "
        "from the project root to generate them (see the README)."
    )
    st.stop()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", models, index=models.index("RandomForestClassifier") if "RandomForestClassifier" in models else 0)
    dna_input = st.toggle("Input is DNA, not protein", value=False)
    st.divider()
    st.subheader("EC classes")
    for ec, description in EC_CLASSES.items():
        st.caption(f"**{ec}** — {description}")

if "sequence_text" not in st.session_state:
    st.session_state.sequence_text = ""


def _apply_example():
    st.session_state.sequence_text = EXAMPLE_SEQUENCES[st.session_state.example_choice]


st.selectbox(
    "Try an example sequence",
    list(EXAMPLE_SEQUENCES.keys()),
    key="example_choice",
    on_change=_apply_example,
)

# A fixed `key` (rather than a dynamic label) keeps this widget's identity —
# and therefore whatever the user typed — stable across the DNA/protein toggle.
sequence_input = st.text_area(
    "DNA sequence" if dna_input else "Protein sequence",
    key="sequence_text",
    height=140,
    placeholder="e.g. MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR...",
)

predict = st.button("Predict", type="primary")

if predict:
    sequence = clean_sequence(sequence_input)

    if dna_input:
        if sequence and set(sequence) - DNA_BASES:
            st.error(f"Not a recognized DNA sequence — unexpected character(s): {', '.join(sorted(set(sequence) - DNA_BASES))}")
            st.stop()
        if not sequence:
            st.error("Enter a sequence first.")
            st.stop()
        try:
            sequence = str(Seq(sequence).translate(to_stop=True))
        except Exception as exc:
            st.error(f"Couldn't translate this DNA sequence: {exc}")
            st.stop()
        st.caption(f"Translated protein: `{sequence}`")

    error = validate_protein(sequence)
    if error:
        st.error(error)
        st.stop()

    clf = load_model(model_name)
    scaler = load_scaler()
    embedding = featurize_single(sequence, scaler)

    try:
        prediction = clf.predict(embedding)[0]
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.success(f"Predicted class: **{prediction}**")
    st.caption(EC_CLASSES.get(prediction, ""))

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(embedding)[0]
        proba_df = pd.DataFrame({"EC class": clf.classes_, "Probability": proba}).sort_values(
            "Probability", ascending=False
        )
        st.bar_chart(proba_df.set_index("EC class"))

    with st.expander("Sequence properties"):
        analysis = ProteinAnalysis(sequence)
        col1, col2, col3 = st.columns(3)
        col1.metric("Length", f"{len(sequence)} aa")
        col2.metric("Molecular weight", f"{analysis.molecular_weight() / 1000:.1f} kDa")
        col3.metric("GRAVY (hydrophobicity)", f"{analysis.gravy():.2f}")

st.divider()
st.caption(
    "Models trained on reviewed UniProtKB sequences per top-level EC class. "
    "See the [README](https://github.com/johnbruvold/protein-prediction) for methodology and limitations."
)
