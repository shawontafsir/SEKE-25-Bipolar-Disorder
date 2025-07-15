import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import mannwhitneyu

# Download necessary resources
nltk.download('punkt')
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()


# Merge both train and test set for analysis
df_train = pd.read_csv('./data/train.csv').dropna()
df_test = pd.read_csv('./data/test.csv').dropna()
df = pd.concat([df_train, df_test])

# Compute sentiment variance for each post
def compute_sentiment_variance(text):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return 0.0
    scores = [sia.polarity_scores(s)['compound'] for s in sentences]
    return np.var(scores)

df["Text"] = df.apply(lambda row: row['Title'] + ". " + row['Content'], axis=1)
df['sentiment_variance'] = df['Text'].apply(compute_sentiment_variance)

# Visualize distribution
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Label', y='sentiment_variance')
plt.title('Intra-Post Sentiment Variance by Label')
plt.xlabel('Label')
plt.ylabel('Sentiment Variance')
plt.grid(True)
plt.tight_layout()
plt.show()


# Run the Mann–Whitney U test
bipolar_var = df[df['Label'] == 'bipolar']['sentiment_variance']
non_bipolar_var = df[df['Label'] == 'non-bipolar']['sentiment_variance']
stat, p = mannwhitneyu(bipolar_var, non_bipolar_var, alternative='two-sided')

print("Mann–Whitney U statistic:", stat)
print("p-value:", p)
