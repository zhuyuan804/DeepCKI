import pandas as pd

all_data = pd.read_csv("../known_cell_cytokine/all_cell_cytokine.csv")

all_data = all_data[["cytokine","cell"]]

data = ["cell-cytokine","cell-cytokine+","cytokine-cell+","cytokine-cell-","cytokine-cell"]

for i in data:
    df = pd.read_csv(i+"_pred.csv")
    df["cytokine"] = df["cytokines"].apply(lambda x: str(x).split(' ')[0])
    df_ = df[["cytokine","cell"]]
    df_new = pd.concat([all_data, df_], axis=0)
    print("未去重前", df_.shape[0])
    # df_new.drop_duplicates(subset=['cytokine', 'cell'], keep='first', inplace=True)
    df_drop = pd.concat([df_new, all_data, all_data]).drop_duplicates(keep=False)

    print("去重后",df_drop.shape[0])
    df_merge = pd.merge(df,df_drop,how="inner",on=["cytokine","cell"])
    print(df_merge.shape[0])
    df_merge.to_csv("filter/"+i+".csv")


