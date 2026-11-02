from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENZYME_CLASSES = {
    "Oxidoreductases": "1.-.-.-",
    "Transferases": "2.-.-.-",
    "Hydrolases": "3.-.-.-",
    "Lyases": "4.-.-.-",
    "Isomerases": "5.-.-.-",
    "Ligases": "6.-.-.-",
    "Translocases": "7.-.-.-",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def get_by_ec(ec: str) -> None:
    """Fetch reviewed UniProtKB sequences for an EC class into data/RawSequences/."""
    ec_class_code = ENZYME_CLASSES.get(ec, False)
    if not ec_class_code:
        return

    out_path = PROJECT_ROOT / "data" / "RawSequences" / f"{ec}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    url = (
        "https://rest.uniprot.org/uniprotkb/search?"
        "fields=accession%2Cid%2Cprotein_name%2Corganism_name%2Clength%2Csequence"
        f"&format=tsv&query=%28%28*%29+AND+%28reviewed%3Atrue%29+AND+%28ec%3A{ec_class_code}%29%29&size=500"
    )
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return
    df = pd.read_csv(url, delimiter="\t")

    while response.status_code == 200 and response.headers.get("Link", False):
        try:
            url = response.headers.get("Link")[1:-13]
            response = requests.get(url, headers=HEADERS)
            df = pd.concat([df, pd.read_csv(url, delimiter="\t")])
        except Exception:
            break

    df.to_csv(out_path)


def clean_sequences(ec: str) -> None:
    """Filter out short/non-canonical/duplicate sequences, writing to data/CleanedSequences/."""
    raw_path = PROJECT_ROOT / "data" / "RawSequences" / f"{ec}.csv"
    if not raw_path.exists():
        return

    out_path = PROJECT_ROOT / "data" / "CleanedSequences" / f"{ec}Cleaned.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    df = df[(df["Length"] >= 50) & ~(df["Sequence"].str.contains("[OUXB]", regex=True))]
    df = df.drop_duplicates("Sequence")
    df.to_csv(out_path)


if __name__ == "__main__":
    for enzyme_class in ENZYME_CLASSES:
        print(f"Fetching {enzyme_class}...")
        get_by_ec(enzyme_class)
        print(f"Cleaning {enzyme_class}...")
        clean_sequences(enzyme_class)
