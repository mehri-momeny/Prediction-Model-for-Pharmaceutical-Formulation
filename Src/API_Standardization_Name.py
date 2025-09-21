import pandas as pd
import requests
from time import sleep
from tqdm import tqdm
import pubchempy as pcp

# Load your CSV file
df = pd.read_csv("../Data/API_List.csv", encoding='ISO-8859-1')
api_names = df['API'].unique()

# Properties from PubChem
properties = ["MolecularFormula", "MolecularWeight", "CanonicalSMILES", "IsomericSMILES",
"InChI", "InChIKey", "IUPACName", "XLogP", "TPSA", "Complexity", "HBondDonorCount", "HBondAcceptorCount",
"RotatableBondCount", "HeavyAtomCount", "IsotopeAtomCount"]


# Collect results
results = []

for compound in tqdm(api_names):
    try:
        props = pcp.get_properties(properties, compound, 'name')
        if props:
            props[0]["InputName"] = compound  # keep original input name
            results.append(props[0])
        else:
            results.append({"InputName": compound, "Error": "Not found"})
    except Exception as e:
        results.append({"InputName": compound, "Error": str(e)})

# Convert to DataFrame
df = pd.DataFrame(results)

# Show dataframe
print(df.head())


# Save to CSV
df.to_csv("pubchem_properties.csv", index=False)

'''
def get_pubchem_info(compound_name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/property/IUPACName,CanonicalSMILES,InChIKey/JSON"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            props = data['PropertyTable']['Properties'][0]
            return {
                'API': compound_name,
                'CID': props.get('CID'),
                'IUPACName': props.get('IUPACName'),
                'MolecularFormula': props.get('MolecularFormula'),
                'CanonicalSMILES': props.get('CanonicalSMILES'),
                'InChIKey': props.get('InChIKey')
            }
    except Exception as e:
        return {'API': compound_name, 'Error': str(e)}
    return {'API': compound_name, 'Error': 'Not Found'}

# Collect results (with delay to avoid rate-limiting)
results = []
for name in api_names:
    results.append(get_pubchem_info(name))
    sleep(0.5)  # Be kind to PubChem servers

# Convert to DataFrame and save
result_df = pd.DataFrame(results)
result_df.to_csv("Data\Mapped_API_to_PubChem.csv", index=False)
print("Mapping complete. File saved as Mapped_API_to_PubChem.csv")
'''