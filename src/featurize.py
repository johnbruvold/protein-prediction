from itertools import product

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import StandardScaler

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
KMERS = ["".join(p) for p in product(AMINO_ACIDS, repeat=3)]
FEATURE_NAMES = (
    list(AMINO_ACIDS)
    + ["MolecularWeight", "Flexibility", "GRAVY", "Helix", "Turn", "Sheet"]
    + KMERS
)


def kmers_freq(seq: str, k: int = 3) -> list:
    counts = {kmer: 0 for kmer in KMERS}
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if kmer in counts:
            counts[kmer] += 1
    length = max(len(seq), 1)
    return [counts[kmer] / length for kmer in KMERS]


def _sequence_features(seq: str) -> list:
    """Raw (unscaled) physicochemical + k-mer features for a single sequence."""
    analysis = ProteinAnalysis(seq)
    aa_percent = analysis.amino_acids_percent
    features = [aa_percent.get(aa, 0.0) for aa in AMINO_ACIDS]
    features.append(analysis.molecular_weight())
    features.append(float(np.mean(analysis.flexibility())))
    features.append(analysis.gravy())
    features.extend(analysis.secondary_structure_fraction())
    features.extend(kmers_freq(seq))
    return features


def extract_features(sequences) -> pd.DataFrame:
    """Raw (unscaled) features for one sequence (str) or many (iterable of str)."""
    if isinstance(sequences, str):
        sequences = [sequences]
    rows = [_sequence_features(seq) for seq in sequences]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def featurize(AA_df: pd.DataFrame, scaler: StandardScaler = None):
    """Extract + scale features for a DataFrame with a "Sequence" column.

    If `scaler` is given, its existing fit is reused (e.g. test/inference data)
    and only the scaled features are returned. Otherwise a new StandardScaler
    is fit on this data and returned alongside the features — use this path
    only for training data, since fitting a scaler per-call means train and
    inference data would otherwise be scaled inconsistently.
    """
    raw = extract_features(AA_df["Sequence"])
    if scaler is None:
        scaler = StandardScaler().fit(raw)
        return scaler.transform(raw), scaler
    return scaler.transform(raw)


def featurize_single(sequence: str, scaler: StandardScaler) -> np.ndarray:
    """Extract + scale features for one sequence using an already-fit scaler.

    A scaler fit on a single sample has zero variance for every feature, which
    collapses the output to all-zeros regardless of the input — the caller
    must pass the StandardScaler that was fit on the training set.
    """
    raw = extract_features(sequence)
    return scaler.transform(raw)
