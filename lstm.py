import gensim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_outputs):
        # Compute attention scores
        attention_scores = self.attention(lstm_outputs).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Apply attention weights
        weighted_output = torch.sum(lstm_outputs * attention_weights.unsqueeze(-1), dim=1)
        return weighted_output


class LSTM(nn.Module):
    def __init__(
            self, hidden_size, num_classes, is_bidirectional=True, has_attention=True, embedding_type='bert',
            vocab=None, embedding_dim=300, pretrained_embedding_path=None
    ):
        super(LSTM, self).__init__()

        if is_bidirectional and has_attention:
            self.class_name = f"BiLSTMWithAttentionWith{embedding_type}Embeddings"
        elif is_bidirectional:
            self.class_name = f"BiLSTMWith{embedding_type}Embeddings"
        elif has_attention:
            self.class_name = f"LSTMWithAttentionWith{embedding_type}Embeddings"
        else:
            self.class_name = f"LSTMWith{embedding_type}Embeddings"
        self.has_attention = has_attention
        self.embedding_type = embedding_type

        if self.embedding_type == 'bert':
            # Load pre-trained BERT model
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            input_size = self.bert_model.config.hidden_size
        else:
            # Initialize embedding layer with pre-trained GloVe or Word2Vec embeddings
            self.embedding = nn.Embedding(len(vocab), embedding_dim)

            if embedding_type == "glove":
                self.embedding.weight = nn.Parameter(
                    self.load_glove_embeddings(pretrained_embedding_path, vocab, embedding_dim)
                )
            elif embedding_type == "word2vec":
                self.embedding_type = nn.Parameter(
                    self.load_word2vec_embeddings(pretrained_embedding_path, vocab, embedding_dim)
                )
            else:
                raise ValueError("Unsupported embedding type. Only GloVe and Word2Vec are supported.")

            input_size = embedding_dim

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=1, dropout=0.2,
            batch_first=True, bidirectional=is_bidirectional
        )

        # Define Attention mechanism
        if self.has_attention:
            # hidden_size * 2 if BiLSTM
            self.attention = Attention(hidden_size * (2 if is_bidirectional else 1))

        # Define Dropout layers
        if self.has_attention:
            self.attention_dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.2)

        # Define Fully Connected layers
        self.fc1 = nn.Linear(hidden_size * (2 if is_bidirectional else 1), 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask=None):
        if self.embedding_type == 'bert':
            # Generate BERT embeddings
            bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = bert_outputs.last_hidden_state  # Get embeddings from last hidden state
        else:
            # Generate embeddings from GloVe or Word2Vec
            embeddings = self.embedding(input_ids)

        # Pass BERT embeddings through LSTM layer
        lstm_outputs, _ = self.lstm(embeddings)

        # Apply attention mechanism
        if self.has_attention:
            weighted_output = self.attention(lstm_outputs)
            fc1_input = self.attention_dropout(weighted_output)
        else:
            # Take the output of the last time step
            fc1_input = lstm_outputs[:, -1, :]

        # Feed the output of "attention" or "last-time-step" through fully connected layers with dropout
        fc1_output = F.relu(self.fc1(fc1_input))
        fc1_output = self.dropout(fc1_output)
        output = self.fc2(fc1_output)

        return output

    @staticmethod
    def load_glove_embeddings(path, vocab, embedding_dim):
        embeddings = np.zeros((len(vocab), embedding_dim))
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                if word in vocab:
                    idx = vocab[word]
                    embeddings[idx] = vector
        return torch.tensor(embeddings, dtype=torch.float32)

    @staticmethod
    def load_word2vec_embeddings(path, vocab, embedding_dim):
        model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
        embeddings = np.zeros((len(vocab), embedding_dim))
        for word, idx in vocab.items():
            if word in model:
                embeddings[idx] = model[word]
        return torch.tensor(embeddings, dtype=torch.float32)


