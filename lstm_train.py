import random
import time
from collections import Counter
from itertools import chain

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer

from lstm import LSTM

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet


# Define documents augmentation functions
def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_words = words.copy()
    for _ in range(n):
        random_word = random.choice(words)
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = random.choice(synonyms).lemma_names()[0]
            new_words = [synonym if word == random_word else word for word in new_words]
    return ' '.join(new_words)


def create_train_loader(_tokenizer, _x_train, _y_train, _batch_size=16):
    _x_train_tokens = _tokenizer(_x_train.tolist(), padding=True, truncation=True, return_tensors='pt')
    _x_train_input_ids = _x_train_tokens.input_ids
    _x_train_attention_mask = _x_train_tokens.attention_mask
    _y_train_tensor = torch.tensor(_y_train.values, dtype=torch.long)

    # Define train dataset and documents loader
    _train_dataset = TensorDataset(_x_train_input_ids, _x_train_attention_mask, _y_train_tensor)
    _train_loader = DataLoader(_train_dataset, batch_size=_batch_size, shuffle=True)

    return _train_loader


def create_validation_loader(_tokenizer, _x_validation, _y_validation, _batch_size=16):
    _x_validation_tokens = _tokenizer(_x_validation.tolist(), padding=True, truncation=True, return_tensors='pt')
    _x_validation_input_ids = _x_validation_tokens.input_ids
    _x_validation_attention_mask = _x_validation_tokens.attention_mask
    _y_validation_tensor = torch.tensor(_y_validation.values, dtype=torch.long)

    # Define train dataset and documents loader
    _validation_dataset = TensorDataset(_x_validation_input_ids, _x_validation_attention_mask, _y_validation_tensor)
    _validation_loader = DataLoader(_validation_dataset, batch_size=_batch_size, shuffle=False)

    return _validation_loader, _y_validation_tensor


def create_test_loader(_tokenizer, _x_test, _y_test, _batch_size=16):
    _x_test_tokens = _tokenizer(_x_test.tolist(), padding=True, truncation=True, return_tensors='pt')
    _x_test_input_ids = _x_test_tokens.input_ids
    _x_test_attention_mask = _x_test_tokens.attention_mask
    _y_test_tensor = torch.tensor(_y_test.values, dtype=torch.long)

    # Define test dataset and documents loader
    _test_dataset = TensorDataset(_x_test_input_ids, _x_test_attention_mask)
    _test_loader = DataLoader(_test_dataset, batch_size=_batch_size, shuffle=False)

    return _test_loader, _y_test_tensor


def train_with_cross_validation(_tokenizer, _model_params, _config, features, labels, _k_folds):
    # Define k-fold cross-validation
    skf = StratifiedKFold(n_splits=_k_folds, shuffle=True, random_state=42)

    # Perform k-fold cross-validation
    for fold, (train_idx, valid_idx) in enumerate(skf.split(features, labels), 1):
        print(f'Fold {fold}/{_k_folds}')

        # Split documents into train and validation sets
        _X_train, _X_validation = features.iloc[train_idx], features.iloc[valid_idx]
        _y_train, _y_validation = labels.iloc[train_idx], labels.iloc[valid_idx]

        train_loader = create_train_loader(_tokenizer, _X_train, _y_train, _batch_size=16)
        validation_loader, y_validation_tensor = create_validation_loader(_tokenizer, _X_validation, _y_validation)

        model = LSTM(**_model_params)
        train(model, _config, train_loader, validation_loader, y_validation_tensor, fold)


def train(_model: LSTM, _config, _train_loader, _validation_loader, _y_validation_tensor, fold=None):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lr = _config["lr"]
    weight_decay = _config["weight_decay"]
    epochs = _config["epochs"]

    # Define optimizer
    optimizer = torch.optim.AdamW(_model.parameters(), lr=lr, weight_decay=weight_decay)
    _scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1, verbose=True
    )
    criterion = nn.CrossEntropyLoss()
    _model = _model.to(device)

    # LSTM with BERT Training loop
    scaler = torch.cuda.amp.GradScaler()
    train_losses, valid_losses, min_valid_loss = list(), list(), float("infinity")

    # Set patience for early stopping
    patience = 3
    patience_counter = 0

    for epoch in range(epochs):
        tqdm_train_loader = tqdm(
            _train_loader, desc=f"{f'Fold {fold}: ' if fold else ''}Epoch {epoch + 1}/{epochs}"
        )
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = 0

        # Training
        _model.train()
        for batch in tqdm_train_loader:
            torch.cuda.empty_cache()
            optimizer.zero_grad()

            # # For Bert embeddings
            # input_ids, attention_mask, labels = batch[0].to(device), None, batch[2].to(device)

            # For glove and word2vec embeddings
            input_ids, attention_mask, labels = batch[0].to(device), None, batch[1].to(device)

            outputs = _model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            # backward propagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            tqdm_train_loader.set_postfix(train_loss=loss.item(), lr=current_lr)

        train_losses.append(train_loss / len(tqdm_train_loader))

        # Validation
        tqdm_valid_loader = tqdm(
            _validation_loader,
            desc=f"Evaluating valid dataset of {len(_validation_loader.dataset)} instances"
        )
        predicted_labels = torch.IntTensor([]).to(device)
        valid_loss = 0

        _model.eval()
        with torch.no_grad():
            for batch in tqdm_valid_loader:
                # # For Bert embeddings
                # input_ids, attention_mask, labels = batch[0].to(device), None, batch[2].to(device)

                # For Glove and Word2Vec
                input_ids, attention_mask, labels = batch[0].to(device), None, batch[1].to(device)

                logits = _model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                valid_loss += loss.item()
                predicted_labels = torch.cat((predicted_labels, torch.argmax(logits, dim=1)), 0)

            valid_losses.append(valid_loss / len(tqdm_valid_loader))
            _scheduler.step(valid_losses[-1])

        # Calculate accuracy
        accuracy = accuracy_score(_y_validation_tensor.cpu().numpy(), predicted_labels.cpu().numpy())
        print(f"Accuracy: {accuracy}")

        f1 = f1_score(_y_validation_tensor.cpu().numpy(), predicted_labels.cpu().numpy())
        print(f"F1 score: {f1}")

        precision = precision_score(_y_validation_tensor.cpu().numpy(), predicted_labels.cpu().numpy())
        print(f"Precision score: {precision}")

        recall = recall_score(_y_validation_tensor.cpu().numpy(), predicted_labels.cpu().numpy())
        print(f"Recall score: {recall}")
        print(f"Train and validation losses: {train_losses[-1]}, {valid_losses[-1]}")

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            patience_counter = 0

            checkpoint = {
                "state_dict": _model.state_dict(),
                "optimizer": optimizer.state_dict()
            }

            save_checkpoint(
                checkpoint,
                filename=f"{_model.class_name}_{f'{fold}_' if fold else ''}checkpoint.tar"
            )
        else:
            patience_counter += 1

        # Early stopping condition
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"{f'Fold {fold}: ' if fold else ''}Train losses per epoch: {train_losses}")
    print(f"{f'Fold {fold}: ' if fold else ''}Valid losses per epoch: {valid_losses}")


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(_model, filepath):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filepath)
    _model.load_state_dict(checkpoint["state_dict"])


def evaluate(_model: LSTM, _test_loader, _y_test_tensor, fold=None):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _model = _model.to(device)
    load_checkpoint(_model, f"{_model.class_name}_{f'{fold}_' if fold else ''}checkpoint.tar")

    # Evaluation
    tqdm_test_loader = tqdm(
        _test_loader,
        desc=f"{f'Fold {fold}: ' if fold else ''}Evaluating test dataset of {len(_test_loader.dataset)} instances"
    )
    predicted_labels = torch.IntTensor([]).to(device)

    _model.eval()
    with torch.no_grad():
        for batch in tqdm_test_loader:
            torch.cuda.empty_cache()
            input_ids, attention_mask = batch[0].to(device), None
            logits = _model(input_ids, attention_mask=attention_mask)
            predicted_labels = torch.cat((predicted_labels, torch.argmax(logits, dim=1)), 0)

    # Calculate accuracy
    # "accuracy_score" function from scikit-learn operates on CPU-based documents
    accuracy = accuracy_score(_y_test_tensor.cpu().numpy(), predicted_labels.cpu().numpy())
    print(f"{f'Fold {fold}: ' if fold else ''}Accuracy: {accuracy}")

    f1 = f1_score(_y_test_tensor.cpu().numpy(), predicted_labels.cpu().numpy())
    print(f"{f'Fold {fold}: ' if fold else ''}F1 score: {f1}")

    precision = precision_score(_y_test_tensor.cpu().numpy(), predicted_labels.cpu().numpy())
    print(f"{f'Fold {fold}: ' if fold else ''}Precision score: {precision}")

    recall = recall_score(_y_test_tensor.cpu().numpy(), predicted_labels.cpu().numpy())
    print(f"{f'Fold {fold}: ' if fold else ''}Recall score: {recall}")

    # # Inverse transform labels to original classes
    # predicted_labels_original = label_encoder.inverse_transform(predicted_labels.cpu().numpy())
    # print("Predicted labels:", predicted_labels_original)

    return predicted_labels.tolist()


def evaluate_with_cross_validation(_tokenizer, _model_params, _x_test, _y_test, _k_folds):
    _test_loader, _y_test_tensor = create_test_loader(_tokenizer, _x_test, _y_test)
    model = LSTM(**_model_params)

    predictions = []
    for fold in range(1, _k_folds + 1):
        predictions.append(evaluate(model, _test_loader, _y_test_tensor, fold))

    # Take majority vote for classification
    weighted_predicted_labels = []
    for j in range(len(predictions[0])):
        # A list containing predicted labels of each model for a particular instance indexed at j
        labels = [predictions[i][j] for i in range(len(predictions))]

        # Take the label having maximum occurrence in each model fold evaluation
        weighted_predicted_labels.append(max(labels, key=labels.count))

    # Calculate accuracy
    accuracy = accuracy_score(_y_test_tensor.cpu().tolist(), weighted_predicted_labels)
    print(f"Cross Validation Accuracy: {accuracy}")

    # Calculate F1-score
    f1 = f1_score(_y_test_tensor.cpu().tolist(), weighted_predicted_labels)
    print(f"Cross Validation F1-score: {f1}")

    precision = precision_score(_y_test_tensor.cpu().tolist(), weighted_predicted_labels)
    print(f"Cross Validation Precision: {precision}")

    recall = recall_score(_y_test_tensor.cpu().tolist(), weighted_predicted_labels)
    print(f"Cross Validation Recall: {recall}")

    return weighted_predicted_labels


def build_vocab(sentences, min_freq=1):
    """
    Build a vocabulary from a list of sentences.

    Args:
        sentences (list of str): List of sentences from which to build the vocabulary.
        min_freq (int): Minimum frequency of words to be included in the vocabulary.

    Returns:
        dict: A dictionary mapping words to indices.
    """
    # Tokenize the sentences and count the frequency of each word
    word_counts = Counter(chain(*[sentence.split() for sentence in sentences]))

    # Filter words by minimum frequency and create a mapping from words to indices
    _vocab = {word: idx for idx, (word, count) in enumerate(word_counts.items()) if count >= min_freq}
    _vocab['<PAD>'] = len(_vocab)

    return _vocab


def encode_sentences(_sentences, _vocab):
    _encoded_sentences = []
    for sentence in _sentences:
        _encoded_sentence = [_vocab.get(word, _vocab['<PAD>']) for word in sentence.split()]
        _encoded_sentences.append(_encoded_sentence)
    return _encoded_sentences


if __name__ == "__main__":
    label_encoder = LabelEncoder()
    df_train = pd.read_csv("./data/train.csv")
    df_train["Text"] = df_train.apply(lambda row: row['Title'] + ". " + row['Content'], axis=1)
    df_train['label_encoded'] = label_encoder.fit_transform(df_train['Label'])
    X_train, y_train = df_train['Text'], df_train['label_encoded']

    df_test = pd.read_csv("./data/test.csv")
    df_test["Text"] = df_test.apply(lambda row: row['Title'] + ". " + row['Content'], axis=1)
    df_test['label_encoded'] = label_encoder.transform(df_test['Label'])
    X_test, y_test = df_test['Text'], df_test['label_encoded']

    # Split train documents into train and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, stratify=y_train, test_size=0.20, random_state=42
    )

    model_params = {
        'hidden_size': 128,  # Size of LSTM hidden state
        'num_classes': len(label_encoder.classes_)  # Number of output classes
    }
    config = {
        "lr": 1e-6,
        "weight_decay": 1e-2,
        "epochs": 300
    }
    k_folds = 5


    # ----------------- Train LSTM with BERT Embedding -----------------------
    # Tokenize text using BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # train_loader = create_train_loader(tokenizer, X_train, y_train, _batch_size=16)
    # validation_loader, y_validation_tensor = create_validation_loader(tokenizer, X_valid, y_valid)
    # test_loader, y_test_tensor = create_test_loader(tokenizer, X_test, y_test)

    print("BiLSTM with Attention:")
    tic = time.perf_counter()
    model_params.update(dict(is_bidirectional=True, has_attention=True))
    # train_with_cross_validation(
    #     tokenizer, model_params, config, pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]), _k_folds=k_folds
    # )
    # evaluate_with_cross_validation(tokenizer, model_params, X_test, y_test, k_folds)
    # model = LSTM(**model_params)
    # train(model, config, train_loader, validation_loader, y_validation_tensor)
    # print(evaluate(model, test_loader, y_test_tensor))
    toc = time.perf_counter()
    print(f"Trained BiLSTM with Attention model in {toc - tic:0.4f} seconds")

    print("BiLSTM:")
    tic = time.perf_counter()
    model_params.update(dict(is_bidirectional=True, has_attention=False))
    # train_with_cross_validation(
    #     tokenizer, model_params, config, pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]), _k_folds=k_folds
    # )
    # evaluate_with_cross_validation(tokenizer, model_params, X_test, y_test, k_folds)
    # model = LSTM(**model_params)
    # train(model, config, train_loader, validation_loader, y_validation_tensor)
    # print(evaluate(model, test_loader, y_test_tensor))
    toc = time.perf_counter()
    print(f"Trained BiLSTM model in {toc - tic:0.4f} seconds")

    print("LSTM with Attention:")
    tic = time.perf_counter()
    model_params.update(dict(is_bidirectional=False, has_attention=True))
    # train_with_cross_validation(
    #     tokenizer, model_params, config, pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]), _k_folds=k_folds
    # )
    # evaluate_with_cross_validation(tokenizer, model_params, X_test, y_test, k_folds)
    # model = LSTM(**model_params)
    # train(model, config, train_loader, validation_loader, y_validation_tensor)
    # print(evaluate(model, test_loader, y_test_tensor))
    toc = time.perf_counter()
    print(f"Trained LSTM with Attention model in {toc - tic:0.4f} seconds")

    print("LSTM:")
    tic = time.perf_counter()
    model_params.update(dict(is_bidirectional=False, has_attention=False))
    # train_with_cross_validation(
    #     tokenizer, model_params, config, pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]), _k_folds=k_folds
    # )
    # evaluate_with_cross_validation(tokenizer, model_params, X_test, y_test, k_folds)
    # model = LSTM(**model_params)
    # train(model, config, train_loader, validation_loader, y_validation_tensor)
    # print(evaluate(model, test_loader, y_test_tensor))
    toc = time.perf_counter()
    print(f"Trained LSTM model in {toc - tic:0.4f} seconds")


    # ------------------ Preparing Vocabulary and Tokenized Posts for Glove and Word2Vec Embeddings ---------------
    vocab = build_vocab(pd.concat([X_train, X_valid, X_test])[0])

    encoded_sentences = encode_sentences(X_train, vocab)
    max_len = max(len(sentence) for sentence in encoded_sentences)
    padded_sentences = [sentence + [vocab['<PAD>']] * (max_len - len(sentence)) for sentence in encoded_sentences]
    sentences_tensor = torch.tensor(padded_sentences, dtype=torch.long)
    labels_tensor = torch.tensor(y_train.values, dtype=torch.long)

    train_dataset = TensorDataset(sentences_tensor, labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    encoded_sentences = encode_sentences(X_valid, vocab)
    max_len = max(len(sentence) for sentence in encoded_sentences)
    padded_sentences = [sentence + [vocab['<PAD>']] * (max_len - len(sentence)) for sentence in encoded_sentences]
    sentences_tensor = torch.tensor(padded_sentences, dtype=torch.long)
    labels_tensor = torch.tensor(y_valid.values, dtype=torch.long)

    validation_dataset = TensorDataset(sentences_tensor, labels_tensor)
    validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False)
    y_validation_tensor = torch.tensor(y_valid.values, dtype=torch.long)

    encoded_sentences = encode_sentences(X_test, vocab)
    max_len = max(len(sentence) for sentence in encoded_sentences)
    padded_sentences = [sentence + [vocab['<PAD>']] * (max_len - len(sentence)) for sentence in encoded_sentences]
    sentences_tensor = torch.tensor(padded_sentences, dtype=torch.long)
    labels_tensor = torch.tensor(y_test.values, dtype=torch.long)

    test_dataset = TensorDataset(sentences_tensor, labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)


    # ------------------- Train LSTM with Glove Embedding ---------------------------------
    # glove_path = "embeddings/glove.6B.300d.txt"
    #
    # print("BiLSTMWithAttentionWithGlove:")
    # model_params.update(
    #     dict(
    #         embedding_type='glove', pretrained_embedding_path=glove_path, vocab=vocab, embedding_dim=300,
    #         is_bidirectional=True, has_attention=True
    #     )
    # )
    # model = LSTM(**model_params)
    # tic = time.perf_counter()
    # train(model, config, train_loader, validation_loader, y_validation_tensor)
    # toc = time.perf_counter()
    # print(f"Trained {model.class_name} model in {toc - tic:0.4f} seconds")
    # evaluate(model, test_loader, y_test_tensor)
    #
    # print("BiLSTMWithGlove:")
    # model_params.update(
    #     dict(
    #         embedding_type='glove', pretrained_embedding_path=glove_path, vocab=vocab, embedding_dim=300,
    #         is_bidirectional=True, has_attention=False
    #     )
    # )
    # model = LSTM(**model_params)
    # tic = time.perf_counter()
    # train(model, config, train_loader, validation_loader, y_validation_tensor)
    # toc = time.perf_counter()
    # print(f"Trained {model.class_name} model in {toc - tic:0.4f} seconds")
    # evaluate(model, test_loader, y_test_tensor)
    #
    # print("LSTMWithAttentionWithGlove:")
    # model_params.update(
    #     dict(
    #         embedding_type='glove', pretrained_embedding_path=glove_path, vocab=vocab, embedding_dim=300,
    #         is_bidirectional=False, has_attention=True
    #     )
    # )
    # model = LSTM(**model_params)
    # tic = time.perf_counter()
    # train(model, config, train_loader, validation_loader, y_validation_tensor)
    # toc = time.perf_counter()
    # print(f"Trained {model.class_name} model in {toc - tic:0.4f} seconds")
    # evaluate(model, test_loader, y_test_tensor)
    #
    # print("LSTMWithGlove:")
    # model_params.update(
    #     dict(
    #         embedding_type='glove', pretrained_embedding_path=glove_path, vocab=vocab, embedding_dim=300,
    #         is_bidirectional=False, has_attention=False
    #     )
    # )
    # model = LSTM(**model_params)
    # tic = time.perf_counter()
    # train(model, config, train_loader, validation_loader, y_validation_tensor)
    # toc = time.perf_counter()
    # print(f"Trained {model.class_name} model in {toc - tic:0.4f} seconds")
    #
    # evaluate(model, test_loader, y_test_tensor)


    # --------------------- Train LSTM with Word2Vec Embeddings ------------------------
    word2vec_path = "embeddings/GoogleNews-vectors-negative300.bin"

    print("BiLSTMWithAttentionWithWord2Vec:")
    model_params.update(
        dict(
            embedding_type='word2vec', pretrained_embedding_path=word2vec_path, vocab=vocab, embedding_dim=300,
            is_bidirectional=True, has_attention=True
        )
    )
    model = LSTM(**model_params)
    tic = time.perf_counter()
    train(model, config, train_loader, validation_loader, y_validation_tensor)
    toc = time.perf_counter()
    print(f"Trained {model.class_name} model in {toc - tic:0.4f} seconds")
    evaluate(model, test_loader, y_test_tensor)

    print("BiLSTMWithWord2Vec:")
    model_params.update(
        dict(
            embedding_type='word2vec', pretrained_embedding_path=word2vec_path, vocab=vocab, embedding_dim=300,
            is_bidirectional=True, has_attention=False
        )
    )
    model = LSTM(**model_params)
    tic = time.perf_counter()
    train(model, config, train_loader, validation_loader, y_validation_tensor)
    toc = time.perf_counter()
    print(f"Trained {model.class_name} model in {toc - tic:0.4f} seconds")
    evaluate(model, test_loader, y_test_tensor)

    print("LSTMWithAttentionWithWord2Vec:")
    model_params.update(
        dict(
            embedding_type='word2vec', pretrained_embedding_path=word2vec_path, vocab=vocab, embedding_dim=300,
            is_bidirectional=False, has_attention=True
        )
    )
    model = LSTM(**model_params)
    tic = time.perf_counter()
    train(model, config, train_loader, validation_loader, y_validation_tensor)
    toc = time.perf_counter()
    print(f"Trained {model.class_name} model in {toc - tic:0.4f} seconds")
    evaluate(model, test_loader, y_test_tensor)

    print("LSTMWithWord2Vec:")
    model_params.update(
        dict(
            embedding_type='word2vec', pretrained_embedding_path=word2vec_path, vocab=vocab, embedding_dim=300,
            is_bidirectional=False, has_attention=False
        )
    )
    model = LSTM(**model_params)
    tic = time.perf_counter()
    train(model, config, train_loader, validation_loader, y_validation_tensor)
    toc = time.perf_counter()
    print(f"Trained {model.class_name} model in {toc - tic:0.4f} seconds")

    evaluate(model, test_loader, y_test_tensor)
