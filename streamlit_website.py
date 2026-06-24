import streamlit as st
from pickle import load
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from src.featurize import featurize_single
import pandas as pd
import os
from Bio.Seq import Seq


proj_path = Path.cwd().parent 
models = os.listdir(proj_path/ "protein-prediction" / "models")

with st.container():
    sequence = st.chat_input("Sequence?")

with st.expander("Settings"):
    model = st.selectbox('Model:', [m[:-4] for m in models if ".pkl" in m])
    DNA_input = st.toggle("Input is DNA?")
result = st.empty()
if sequence:
    with open(proj_path/ "protein-prediction" / "models" /f"{model}.pkl","rb") as file:
        clf = load(file)
    if DNA_input:
        try:
            sequence = Seq(sequence).translate(to_stop=True)
        except:
            result.write("Improper DNA")
    embedding = featurize_single(sequence)
    try:
        result.write(f"{clf.predict(embedding)[0]}")
    except:
        result.write("Bad Enzyme")

    #/Users/johnbruvold/Desktop/protein-predictor/protein-prediction/models