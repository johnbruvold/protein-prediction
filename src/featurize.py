from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import numpy as np
from itertools import product
from sklearn.preprocessing import StandardScaler 

def kmers_freq(seq,k=3):
    kmers = ["".join(p) for p in product("ACDEFGHIKLMNPQRSTVWY",repeat=3)]
    freq = {kmer:0 for kmer in kmers}
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in freq:
            freq[kmer] += 1
    return pd.Series([v /len(seq) for v in freq.values()])

def featurize(AA_df:pd.DataFrame,kmers_k=3):
    """"Returns an array of scaled and transformed features from the sequence Dataframe"""
    #secondary_structures = ["Helix","Turn","Sheet"]
    features_array = []
    AAs = AA_df["Sequence"].apply(lambda x: ProteinAnalysis(x).amino_acids_percent)

    for amino_acid in "ACDEFGHIKLMNPQRSTVWY":
        features_array.append(AAs.apply(lambda x: x.get(amino_acid,0)))

    features_array.append(AA_df["Sequence"].apply(lambda x: ProteinAnalysis(x).molecular_weight()).to_list())
    features_array.append(AA_df["Sequence"].apply(lambda x: np.mean(ProteinAnalysis(x).flexibility())).to_list())
    features_array.append(AA_df["Sequence"].apply(lambda x: ProteinAnalysis(x).gravy()).to_list())


    secondary_structures = AA_df["Sequence"].apply(lambda x: ProteinAnalysis(x).secondary_structure_fraction())
    for ss in range(3):
        features_array.append(secondary_structures.apply(lambda x: x[ss]).to_list())
    kmers = AA_df["Sequence"].apply(kmers_freq).to_numpy()
    features_array = np.array(features_array).T
    features_array = np.append(features_array,kmers,axis=1)
    features_array = StandardScaler().fit_transform(features_array)
    return features_array

def featurize_single(Sequence:str):
    features_array = []
    AAs = ProteinAnalysis(Sequence).amino_acids_percent
    for amino_acid in "ACDEFGHIKLMNPQRSTVWY":
        features_array.append(AAs.get(amino_acid,0))
    features_array.append(ProteinAnalysis(Sequence).molecular_weight())
    features_array.append(np.mean(ProteinAnalysis(Sequence).flexibility()))
    features_array.append(ProteinAnalysis(Sequence).gravy())


    secondary_structures = ProteinAnalysis(Sequence).secondary_structure_fraction()
    for ss in range(3):
        features_array.append(secondary_structures[ss])
    kmers = kmers_freq(Sequence)
    features_array = np.array(features_array).T
    features_array = np.append(features_array,kmers).reshape(-1, 1).T
    features_array = StandardScaler().fit_transform(features_array)
    return features_array