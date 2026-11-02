# Enzyme Class Predictor

A Streamlit app that predicts which of the 7 top-level [Enzyme Commission (EC)
classes](https://en.wikipedia.org/wiki/List_of_enzymes) a protein sequence
belongs to, from the sequence alone.

| EC class | Function |
|---|---|
| Oxidoreductases | Catalyze oxidation-reduction (electron transfer) reactions |
| Transferases | Transfer a functional group from one molecule to another |
| Hydrolases | Break chemical bonds using water (hydrolysis) |
| Lyases | Break bonds without hydrolysis or oxidation |
| Isomerases | Rearrange atoms within a molecule to form an isomer |
| Ligases | Join two molecules together, typically using ATP |
| Translocases | Move molecules or ions across membranes |

## How it works

1. **Data** ([`src/data.py`](src/data.py)) — reviewed sequences per EC class are
   pulled from UniProtKB.
2. **Features** ([`src/featurize.py`](src/featurize.py)) — each sequence is
   turned into a numeric vector: amino acid composition (20 features),
   molecular weight, mean flexibility, GRAVY (hydrophobicity), predicted
   secondary structure fraction (helix/turn/sheet), and normalized 3-mer
   frequencies (8,000 features). The vector is scaled with a `StandardScaler`
   fit once on the training set, then reused — unseen — at inference time.
3. **Models** ([`src/train.py`](src/train.py)) — a Random Forest and an SVM
   (RBF kernel) are trained on 2,000 sampled sequences per class.
4. **App** ([`streamlit_website.py`](streamlit_website.py)) — paste a
   sequence (protein or DNA, translated on the fly), pick a model, get a
   prediction with per-class probabilities.

## Running it

```bash
conda env create -f environment.yml
conda activate protein-clf
```

or with pip:

```bash
pip install -r requirements.txt
```

`data/` isn't checked into the repo — the raw/cleaned sequence data runs into
the hundreds of MB. The Random Forest model and scaler ship in `models/` so
the app works out of the box; the SVM (500MB+, too large for git and for
Streamlit Community Cloud's free-tier memory) doesn't. Regenerate either
locally if you want to retrain or add the SVM back to the model picker:

```bash
python -m src.data    # fetch sequences from UniProt into data/
python -m src.train   # extract features, train models, save models/*.pkl + scaler.pkl
```

Then launch the app:

```bash
streamlit run streamlit_website.py
```

### Deploying on Streamlit Community Cloud

1. Push this repo to GitHub (it already ships the Random Forest model + scaler,
   so no training step is needed on the deploy side).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at this repo, `main` branch, main file path
   `protein-prediction/streamlit_website.py` (or `streamlit_website.py` if
   deploying from the `protein-prediction` repo directly rather than the
   `protein-predictor` wrapper that embeds it as a submodule).
3. If deploying from the wrapper repo, enable "Clone submodules" in the app's
   advanced settings, since `protein-prediction` is linked in as a git
   submodule.

## Project layout

```
src/
  data.py        # fetch sequences per EC class from UniProtKB
  featurize.py   # sequence -> feature vector (+ scaling)
  train.py       # train/test split, model training, saves models/ + scaler.pkl
notebooks/       # exploratory data analysis and early modeling
streamlit_website.py
```

## Limitations

- Classifying only at the top EC level (1–7), not full EC numbers.
- Trained on reviewed UniProtKB entries only, which skews toward
  well-studied organisms and protein families.
- 3-mer frequency features make for a high-dimensional, mostly sparse
  feature space; results on sequences very unlike the training distribution
  (e.g. very short fragments) should be treated as low-confidence.
