import os
import re

import pandas as pd

# posts = pd.read_csv('./suicidewatch_posts_pushshiftapi.csv')
#
# posts = posts.sort_values('Created_UTC', ascending=False)
#
# posts[:300].to_csv('./latest_suicidewatch_posts.csv', index=False)
#
# df_list = [pd.read_csv(f'ACMSE-25/data/{file}') for file in os.listdir("ACMSE-25/data/")]

# df = pd.read_csv('./politics_posts.csv')
# df = df[(df['Created_UTC'] >= 1577836800) & (df['Created_UTC'] <= 1672531199)]
# df = df.dropna().sample(frac=1)
#
# df.to_csv('./ACMSE-25/data/politics_posts_dated.csv', index=False)

non_bipolar_files = [
    "anxiety_posts_dated.csv", "bicycletouring_posts_dated.csv", "confidence_posts_dated.csv",
    "depression_posts_dated.csv", "geopolitics_posts_dated.csv",
    "politics_posts_dated.csv", "sports_posts_dated.csv", "travel_posts_dated.csv"
]
bipolar_files = ["bipolar_posts_dated.csv"]

non_bipolar_df = pd.concat([pd.read_csv(f'./data/{file}') for file in non_bipolar_files])
bipolar_df = pd.read_csv(f'./data/{bipolar_files[0]}')

non_bipolar_df["Label"] = "non-bipolar"
bipolar_df["Label"] = "bipolar"

# Remove common authors
common_author = set(bipolar_df.merge(non_bipolar_df, on='Author', how='inner')['Author'])
bipolar_df = bipolar_df[bipolar_df['Author'].isin(common_author) == False]
non_bipolar_df = non_bipolar_df[non_bipolar_df['Author'].isin(common_author) == False]

# Comprehensive regex for bipolar disorder self-identification
bipolar_regex = r'\b(bipolar disorder|bipolar|manic depressive illness|bipolar I|bipolar II|cyclothymia|diagnosed with bipolar|I have bipolar|living with bipolar disorder|diagnosed with manic depression|bipolar diagnosis|I was diagnosed with bipolar|mood swings|manic episode|depressive episode|manic depression|rapid cycling|highs and lows|elevated mood|euphoric|mania|manic|hypomanic|hypomania|depressed|severe depression|feeling invincible|can\'t sleep during manic episode|racing thoughts|talking fast|reckless behavior|impulsive decisions|bipol\b|manic-depressive|I feel bipolar|I\'m bipolar|been told I\'m bipolar|I think I\'m bipolar|I\'ve been diagnosed as bipolar|people say I\'m bipolar|acting bipolar|extreme highs|extreme lows|feeling euphoric|feeling invincible|over-the-top happiness|over-the-top energy|feeling empty|feeling worthless|deep depression|suicidal thoughts during lows|feeling on top of the world|feeling down in the dumps|lithium|valproate|divalproex|lamotrigine|abilify|seroquel|quetiapine|mood stabilizer|antipsychotics for bipolar|bipolar meds|bipolar medication|been living with bipolar for years|I\'ve struggled with bipolar my whole life|my bipolar diagnosis|my manic depression diagnosis|I\'ve always been bipolar|I\'ve known I\'m bipolar|my therapist says I\'m bipolar|psychiatrist diagnosed me with bipolar|mental health issues|mental illness|I suffer from mental illness|mental health condition|struggling with mental health|my mental health|feeling mentally unstable|unbalanced mentally)\b'
mask = bipolar_df.apply(lambda row: bool(re.search(bipolar_regex, row["Content"])), axis=1)
bipolar_df = bipolar_df[mask]

df = pd.concat([non_bipolar_df, bipolar_df])
df = df.sample(frac=1).reset_index(drop=True)

# ----------------- Stratified Sample of the Dataset for training and testing ------------------
# Calculate stratified sample (20% of each class)
sample_size = 0.2
a = df.groupby('Label', group_keys=False)
stratified_sample = df.groupby('Label', group_keys=False).apply(
    lambda x: x.sample(frac=sample_size, random_state=42)
)

# Ensure remaining data excludes sampled indices
stratified_indices = stratified_sample.index
remaining_data = df.loc[~df.index.isin(stratified_indices)]

# Debugging outputs
print(f"Total dataset size: {len(df)}")
print(f"Bipolar df size: {len(df[df['Label'] == 'bipolar'])}")
print(f"Stratified sample size: {len(stratified_sample)}")
print(f"Remaining data size: {len(remaining_data)}")


# Save the stratified sample for expert review
stratified_sample.to_csv('./data/test.csv', index=False)
remaining_data.to_csv('./data/train.csv', index=False)

