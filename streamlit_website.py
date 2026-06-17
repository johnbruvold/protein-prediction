import streamlit as st
from pickle import load
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from src.featurize import featurize_single
import pandas as pd

proj_path = Path.cwd().parent 

with open(proj_path/ "protein-prediction" / "models" /"RandomForestClassifier.pkl","rb") as file:
    clf = load(file)

sequence = st.chat_input("Sequence?")
if sequence:
    st.write(f"{clf.predict(featurize_single(sequence))}")#/Users/johnbruvold/Desktop/protein-predictor/protein-prediction/models