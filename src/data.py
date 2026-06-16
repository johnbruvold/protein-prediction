import requests
import pandas as pd
from pathlib import Path


enzyme_classes = {
    "Oxidoreductases": "1.-.-.-",
    "Transferases": "2.-.-.-",
    "Hydrolases": "3.-.-.-",
    "Lyases": "4.-.-.-",
    "Isomerases": "5.-.-.-",
    "Ligases": "6.-.-.-",
    "Translocases": "7.-.-.-"
}

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

def get_by_ec(ec):
    ec_class_code = enzyme_classes.get(ec,False)
    proj_path = Path.cwd().parent
    if not(ec_class_code):
        return
    url=f"https://rest.uniprot.org/uniprotkb/search?fields=accession%2Cid%2Cprotein_name%2Corganism_name%2Clength%2Csequence&format=tsv&query=%28%28*%29+AND+%28reviewed%3Atrue%29+AND+%28ec%3A{ec_class_code}%29%29&size=500"
    cr = requests.get(url,headers=headers)
    if cr.status_code != 200:
        return
    df = pd.read_csv(url,delimiter="\t")
    while (cr.status_code == 200) and ( cr.headers.get('Link',False)):
        try:
            url = cr.headers.get('Link')[1:-13]
            cr = requests.get(url,headers=headers)
            df = pd.concat([df,pd.read_csv(url,delimiter="\t")])
        except:
            df.to_csv(proj_path / "data" / "RawSequences" / f"{ec}.csv")
    df.to_csv(proj_path / "data" / "RawSequences" / f"{ec}.csv")
