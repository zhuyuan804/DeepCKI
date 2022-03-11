import pandas as pd


cell_name = pd.read_excel("../cell name.xlsx")

cytokine_cell = pd.read_csv("cytokine-cell-.csv")

Cell_Ontology_ID = cell_name['Cell Ontology ID'].values

Cell_Ontology_Label = cell_name['Cell Ontology Label'].values

name_map = dict(zip(Cell_Ontology_ID,Cell_Ontology_Label))

cytokine_cell['Cell ontology Label'] = cytokine_cell['cell'].apply(lambda x:name_map[x])

cytokine_cell.to_csv("cytokine-cell-.csv")



