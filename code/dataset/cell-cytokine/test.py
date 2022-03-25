import pandas as pd

data = pd.read_csv("cell-cytokine有无相互作用.csv")
cytokine = data["Cytokine.Ontology.Label"]
cells = data["Cell.Ontology.Label"]
Cytokine = []
Cell = []
for i in range(len(cytokine)):
    cyto = cytokine[i]
    cell = cells[i]
    cell = cell.strip().split(";")
    for j in cell:
        Cytokine.append(cyto)
        Cell.append(j)

df = pd.DataFrame.from_dict({"cytokine":Cytokine,"cell":Cell})
df.to_csv("cell-cytokine.csv")
