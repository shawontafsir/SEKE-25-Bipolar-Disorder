import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformer_train_base import ModelTrain

# Encode target labels using LabelEncoder
label_encoder = LabelEncoder()

df_train = pd.read_csv("./data/train.csv")
df_train["Text"] = df_train.apply(lambda row: row['Title'] + ". " + row['Content'], axis=1)
df_train['label_encoded'] = label_encoder.fit_transform(df_train['Label'])
X_train, y_train = df_train['Text'], df_train['label_encoded']

df_test = pd.read_csv("./data/test.csv")
df_test["Text"] = df_test.apply(lambda row: row['Title'] + ". " + row['Content'], axis=1)
df_test['label_encoded'] = label_encoder.fit_transform(df_test['Label'])
X_test, y_test = df_test['Text'], df_test['label_encoded']

# Split train documents into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, stratify=y_train, test_size=0.20, random_state=42
)

k_folds = 5

print("Roberta:")
tic = time.perf_counter()
model_roberta = ModelTrain('roberta-base', label_encoder, X_train, X_test, X_valid, y_train, y_test, y_valid, k_folds=k_folds)
toc = time.perf_counter()
print(f"Trained Roberta model in {toc - tic:0.4f} seconds")

print("Distilbert:")
tic = time.perf_counter()
model_distilbert = ModelTrain('distilbert-base-uncased', label_encoder, X_train, X_test, X_valid, y_train, y_test, y_valid, k_folds=k_folds)
toc = time.perf_counter()
print(f"Trained Distilbert model in {toc - tic:0.4f} seconds")

print("Albert:")
tic = time.perf_counter()
model_albert = ModelTrain('albert-base-v2', label_encoder, X_train, X_test, X_valid, y_train, y_test, y_valid, k_folds=k_folds)
toc = time.perf_counter()
print(f"Trained Albert model in {toc - tic:0.4f} seconds")

print("Bert:")
tic = time.perf_counter()
model_bert = ModelTrain('bert-base-uncased', label_encoder, X_train, X_test, X_valid, y_train, y_test, y_valid, k_folds=k_folds)
toc = time.perf_counter()
print(f"Trained Bert model in {toc - tic:0.4f} seconds")

print("Electra:")
tic = time.perf_counter()
model_electra = ModelTrain('google/electra-base-discriminator', label_encoder, X_train, X_test, X_valid, y_train, y_test, y_valid, k_folds=k_folds)
toc = time.perf_counter()
print(f"Trained Electra model in {toc - tic:0.4f} seconds")
