import pubchempy as pcp
from tqdm import tqdm
import pandas as pd

# === CONFIGURATION ===
input_file = "Excipient/excipients.csv"       # Your input Excel file
input_column = "Excipients"           # Column name containing excipient names
output_file = "Excipient/excipient_info.csv"  # Output Excel file name

# === READ EXCEL ===
df = pd.read_csv(input_file)
print(df)


# Prepare results list
results = []

for name in tqdm(df[input_column], desc="Get information:"):
    try:
        compounds = pcp.get_compounds(name,'name')
        print(f" the information for {name} is {compounds}")
        if compounds:
            comp = compounds[0]
            results.append({
                "Input Name": name,
                "CID": comp.cid,
                "IUPACName": comp.iupac_name,
                "CanonicalSMILES": comp.canonical_smiles,
                "InChIKey": comp.inchikey
            })
        else:
            results.append({
                "Input Name": name,
                "CID": None,
                "IUPACName": None,
                "CanonicalSMILES": None,
                "InChIKey": None
            })
    except Exception as ex:
        results.append({
            "Input Name": name,
            "CID": None,
            "IUPACName": None,
            "CanonicalSMILES": None,
            "InChIKey": None
        })
        print(f"Error fetching {name}: {ex}")


# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Merge with original data if needed
output_df = pd.concat([df, results_df.drop(columns=["Input Name"])], axis=1)

# Save to Excel
output_df.to_csv(output_file, index=False)

print(f"✅ Finished! Results saved to {output_file}")



