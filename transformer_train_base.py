import random

import nltk
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download('wordnet')
from nltk.corpus import wordnet


class ModelTrain:

    def __init__(self, pretrained_model_name_or_path, label_encoder, X_train, X_test, X_valid, y_train, y_test, y_valid, k_folds=1, X_test_extra=None, y_test_extra=None):
        self.label_encoder = label_encoder

        self.config = {
            "lr": 1e-6,
            "weight_decay": 1e-2,
            "epochs": 100
        }

        # self.scheduler = tune.schedulers.ASHAScheduler(
        #     metric="f1",
        #     mode="max",
        #     max_t=50,
        #     grace_period=10,
        #     reduction_factor=2
        # )

        # Define new tokenizer and model based on given pretrained model name for each fold
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        if k_folds <= 1:
            train_loader = self.create_train_loader(tokenizer, X_train, y_train, _batch_size=16)
            validation_loader, y_validation_tensor = self.create_validation_loader(tokenizer, X_valid, y_valid)

            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path, num_labels=len(self.label_encoder.classes_)
            )
            self.train(model, self.config, train_loader, validation_loader, y_validation_tensor)

            # analysis = tune.run(
            #     tune.with_parameters(self.train, model=model),
            #     resources_per_trial={"cpu": 2, "gpu": 1},
            #     config=self.config,
            #     num_samples=10,
            #     scheduler=self.scheduler,
            #     local_dir="./ray_tune_results"
            # )
            #
            # print("Best hyperparameters found were: ", analysis.best_config)

            test_loader, y_test_tensor = self.create_test_loader(tokenizer, X_test, y_test)
            self.predicted_labels = self.evaluate(model, test_loader, y_test_tensor)
            if X_test_extra is not None:
                test_loader_extra, y_test_tensor_extra = self.create_test_loader(tokenizer, X_test_extra, y_test_extra)
                self.predicted_labels_extra = self.evaluate(model, test_loader_extra, y_test_tensor_extra)

        else:
            self.train_with_cross_validation(
                tokenizer, pretrained_model_name_or_path, pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]), k_folds
            )
            self.predicted_labels = self.evaluate_with_cross_validation(tokenizer, pretrained_model_name_or_path, X_test, y_test, k_folds)
            if X_test_extra is not None:
                self.predicted_labels_extra = self.evaluate_with_cross_validation(tokenizer, pretrained_model_name_or_path, X_test_extra, y_test_extra, k_folds)

    # Define documents augmentation functions
    @staticmethod
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

    @staticmethod
    def random_swap(sentence, n=5):
        words = sentence.split()

        # Check if the sentence has at least two words
        if len(words) < 2:
            return sentence

        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

    @staticmethod
    def create_train_loader(_tokenizer, _x_train, _y_train, _batch_size=16):
        _x_train_tokens = _tokenizer(_x_train.tolist(), padding=True, truncation=True, return_tensors='pt')
        _x_train_input_ids = _x_train_tokens.input_ids
        _x_train_attention_mask = _x_train_tokens.attention_mask
        _y_train_tensor = torch.tensor(_y_train.values, dtype=torch.long)

        # Define train dataset and documents loader
        _train_dataset = TensorDataset(_x_train_input_ids, _x_train_attention_mask, _y_train_tensor)
        _train_loader = DataLoader(_train_dataset, batch_size=_batch_size, shuffle=True)

        return _train_loader

    @staticmethod
    def create_validation_loader(_tokenizer, _x_validation, _y_validation, _batch_size=16):
        _x_validation_tokens = _tokenizer(_x_validation.tolist(), padding=True, truncation=True, return_tensors='pt')
        _x_validation_input_ids = _x_validation_tokens.input_ids
        _x_validation_attention_mask = _x_validation_tokens.attention_mask
        _y_validation_tensor = torch.tensor(_y_validation.values, dtype=torch.long)

        # Define train dataset and documents loader
        _validation_dataset = TensorDataset(_x_validation_input_ids, _x_validation_attention_mask, _y_validation_tensor)
        _validation_loader = DataLoader(_validation_dataset, batch_size=_batch_size, shuffle=False)

        return _validation_loader, _y_validation_tensor

    @staticmethod
    def create_test_loader(_tokenizer, _x_test, _y_test, _batch_size=16):
        _x_test_tokens = _tokenizer(_x_test.tolist(), padding=True, truncation=True, return_tensors='pt')
        _x_test_input_ids = _x_test_tokens.input_ids
        _x_test_attention_mask = _x_test_tokens.attention_mask
        _y_test_tensor = torch.tensor(_y_test.values, dtype=torch.long)

        # Define test dataset and documents loader
        _test_dataset = TensorDataset(_x_test_input_ids, _x_test_attention_mask)
        _test_loader = DataLoader(_test_dataset, batch_size=_batch_size, shuffle=False)

        return _test_loader, _y_test_tensor

    def train(self, _model, _config, _train_loader, _validation_loader, _y_validation_tensor, fold=None):
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
        _model = _model.to(device)

        # Training loop with tqdm
        scaler = torch.cuda.amp.GradScaler()
        train_losses, valid_losses, min_valid_loss = list(), list(), float("infinity")

        # Set patience for early stopping
        patience, patience_counter = 3, 0

        for epoch in range(epochs):
            tqdm_train_loader = tqdm(
                _train_loader, desc=f"{f'Fold {fold}: ' if fold else ''}Epoch {epoch + 1}/{epochs}"
            )
            current_lr = optimizer.param_groups[0]["lr"]
            train_loss = 0

            _model.train()
            for batch in tqdm_train_loader:
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = _model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

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
                desc=f"Evaluating validation dataset of {len(_validation_loader.dataset)} instances"
            )
            predicted_labels = torch.IntTensor([]).to(device)
            valid_loss = 0

            _model.eval()
            with torch.no_grad():
                for batch in tqdm_valid_loader:
                    input_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    logits = _model(input_ids, attention_mask=attention_mask, labels=labels)
                    # logits = {'loss': ..., 'logits': ..., ...}
                    valid_loss += logits.loss.item()
                    predicted_labels = torch.cat((predicted_labels, torch.argmax(logits.logits, dim=1)), 0)

                valid_losses.append(valid_loss / len(tqdm_valid_loader))
                _scheduler.step(valid_losses[-1])

            # Calculate accuracy
            accuracy = accuracy_score(_y_validation_tensor.cpu().numpy(), predicted_labels.cpu().numpy())
            print(f"Accuracy: {accuracy}")

            # Calculate F1-score
            f1 = f1_score(_y_validation_tensor.cpu().numpy(), predicted_labels.cpu().numpy())
            print(f"F1-score: {f1}")

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

                self.save_checkpoint(
                    checkpoint,
                    filename=f"{_model.__class__.__name__}_{f'{fold}_' if fold else ''}checkpoint.tar"
                )
            else:
                patience_counter += 1

            # Early stopping condition
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            # Report the F1-score to Ray
            # ray_train.report({'f1': f1, 'accuracy': accuracy})

        print(f"{f'Fold {fold}: ' if fold else ''}Train losses per epoch: {train_losses}")
        print(f"{f'Fold {fold}: ' if fold else ''}Valid losses per epoch: {valid_losses}")

    def train_with_cross_validation(self, _tokenizer, _pretrained_model_name_or_path, features, labels, _k_folds):
        # Define k-fold cross-validation
        skf = StratifiedKFold(n_splits=_k_folds, shuffle=True, random_state=42)

        # Perform k-fold cross-validation
        for fold, (train_idx, valid_idx) in enumerate(skf.split(features, labels), 1):
            print(f'Fold {fold}/{_k_folds}')

            # Split documents into train and validation sets
            X_train, X_validation = features.iloc[train_idx], features.iloc[valid_idx]
            y_train, y_validation = labels.iloc[train_idx], labels.iloc[valid_idx]

            train_loader = self.create_train_loader(_tokenizer, X_train, y_train, _batch_size=16)
            validation_loader, y_validation_tensor = self.create_validation_loader(
                _tokenizer, X_validation, y_validation
            )

            # self.create_loader(tokenizer, X_train, X_test, X_valid, y_train, y_test, y_valid)

            # analysis = tune.run(
            #     tune.with_parameters(self.train, model=model, fold=fold),
            #     resources_per_trial={"cpu": 2, "gpu": 1},
            #     config=self.config,
            #     num_samples=10,
            #     scheduler=self.scheduler
            # )
            #
            # best_trial = analysis.get_best_trial("f1", "max", "last")
            # print(f"Best trial config: {best_trial.config}")
            # print(f"Best trial final validation loss: {best_trial.last_result['accuracy']}")

            model = AutoModelForSequenceClassification.from_pretrained(
                _pretrained_model_name_or_path, num_labels=len(self.label_encoder.classes_)
            )
            self.train(model, self.config, train_loader, validation_loader, y_validation_tensor, fold)

    @staticmethod
    def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
        print("=> Saving checkpoint")
        torch.save(state, filename)

    @staticmethod
    def load_checkpoint(model, filepath):
        print("=> Loading checkpoint")
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint["state_dict"])

    def evaluate(self, _model, _test_loader, _y_test_tensor, fold=None):
        # Check if GPU is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _model = _model.to(device)
        self.load_checkpoint(_model, f"{_model.__class__.__name__}_{f'{fold}_' if fold else ''}checkpoint.tar")

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
                input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
                logits = _model(input_ids, attention_mask=attention_mask)
                # logits = {'loss': ..., 'logits': ..., ...}
                predicted_labels = torch.cat((predicted_labels, torch.argmax(logits.logits, dim=1)), 0)

        # Calculate accuracy
        accuracy = accuracy_score(_y_test_tensor.cpu().numpy(), predicted_labels.cpu().numpy())
        print(f"{f'Fold {fold}: ' if fold else ''}Accuracy: {accuracy}")

        # Calculate F1-score
        f1 = f1_score(_y_test_tensor.cpu().numpy(), predicted_labels.cpu().numpy())
        print(f"{f'Fold {fold}: ' if fold else ''}F1-score: {f1}")

        precision = precision_score(_y_test_tensor.cpu().numpy(), predicted_labels.cpu().numpy())
        print(f"Precision: {precision}")

        recall = recall_score(_y_test_tensor.cpu().numpy(), predicted_labels.cpu().numpy())
        print(f"Recall: {recall}")

        # Inverse transform labels to original classes
        # predicted_labels_original = self.label_encoder.inverse_transform(predicted_labels.cpu().numpy())
        # print(f"{f'Fold {fold}: ' if fold else ''}Predicted labels: {predicted_labels_original}")

        return predicted_labels.tolist()

    def evaluate_with_cross_validation(self, _tokenizer, _pretrained_model_name_or_path, _x_test, _y_test, _k_folds):
        _test_loader, _y_test_tensor = self.create_test_loader(_tokenizer, _x_test, _y_test)
        _model = AutoModelForSequenceClassification.from_pretrained(
            _pretrained_model_name_or_path, num_labels=len(self.label_encoder.classes_)
        )

        predictions = []
        for fold in range(1, _k_folds+1):
            predictions.append(self.evaluate(_model, _test_loader, _y_test_tensor, fold))

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
