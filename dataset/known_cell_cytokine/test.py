import pandas as pd

data = ["cell-cytokine","cell-cytokine+","cytokine-cell+","cytokine-cell-","cytokine-cell"]

df_new = pd.DataFrame()
for i in data:
    df = pd.read_csv(i+".csv")
    list1 = [i]*df.shape[0]
    df["source"] = list1
    df_new = pd.concat([df_new, df], axis=0)

df_new.drop_duplicates(subset=['cytokine','cell'],keep='first',inplace=True)

df_new.to_csv("all_cell_cytokine.csv")



