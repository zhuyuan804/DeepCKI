import pandas as pd

string = pd.read_table("ppi_entry.txt",sep=',')

string = string[["protein1_Gene","protein2_Gene","combined_score"]]

# cell_cytokine = pd.read_csv("cytokine-cell.csv")
# cytokines = cell_cytokine["cytokine"]
uniprot = pd.read_pickle("features.pkl")

cytokines = uniprot["cytokine"]

cytokine_encoding = {}

for i,cytokine in enumerate(cytokines):
    cytokine_encoding[cytokine] = i

cytokine_string = string[string["protein1_Gene"].isin(cytokines)& string["protein2_Gene"].isin(cytokines)]

cytokine_string["combined_score"] = cytokine_string["combined_score"].apply(lambda x:x*0.001)

cytokine_string["protein1_Gene"] = cytokine_string["protein1_Gene"].apply(lambda x:cytokine_encoding[x])
cytokine_string["protein2_Gene"] = cytokine_string["protein2_Gene"].apply(lambda x:cytokine_encoding[x])


cytokine_string.to_csv("networks/cytokine_ppi.txt",sep='\t')

