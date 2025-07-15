import pandas as pd
from sklearn.metrics import cohen_kappa_score

# # ----------------- Stratified Sample of the Dataset for Judgemental Analysis ------------------
# df = pd.concat([
#     pd.read_csv('./data/train.csv').dropna(),
#     pd.read_csv('./data/test.csv').dropna()
# ])
#
# # Define the sampling criteria, e.g., equal representation of class labels
# sample_size = 0.02  # 2% of the dataset
# stratified_sample = df.groupby('Label', group_keys=False).apply(lambda x: x.sample(min(len(x), int(sample_size * len(x)))))
# stratified_sample["Text"] = stratified_sample.apply(lambda row: f"{row['Title']} {row['Content']}", axis=1)
# stratified_sample = stratified_sample[["Text", "Label"]].reset_index(drop=True)
# stratified_sample_khalid = stratified_sample[["Text"]].assign(Label="")
# stratified_sample_jamil = stratified_sample[["Text"]].assign(Label="")
#
#
# # Save the stratified sample for expert review
# stratified_sample.to_csv('./data/judgmental_sample.csv')
# stratified_sample_khalid.to_csv('./data/judgemental_sample_khalid.csv')
# stratified_sample_jamil.to_csv('./data/judgemental_sample_jamil.csv')


# ---------------------- Judgemental Analysis -----------------------------
df_judgmental = pd.read_csv('./data/judgmental_sampling.csv')
df_judgmental["Label"] = df_judgmental.apply(lambda row: 1 if row["Label"] == "bipolar" else 0, axis=1)

df_judgmental_khalid = pd.read_csv('./data/judgmental_sampling_khalid.csv')
df = pd.merge(df_judgmental, df_judgmental_khalid, on="Id", how="inner")
print(cohen_kappa_score(df["Label_x"], df["Label_y"]))

df_judgmental_jamil = pd.read_csv('./data/judgmental_sampling_jamil.csv')
df = df_judgmental.merge(df_judgmental_jamil, on="Id", how="inner")
print(cohen_kappa_score(df["Label_x"], df["Label_y"]))
