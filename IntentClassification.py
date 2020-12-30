import json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import torch
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import os
import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report

class IntentClassificationWithBert:

    def __init__(self, device, path_to_json, num_labels_to_sample=20,
                 seed=5, padding_max_length=128, batch_size=16,
                 optimizer='AdamW', lr=2e-5, epochs=4):
        """
        Initialize IntentClassificationWithBert.

        :param device (str): GPU or CPU
        :param path_to_json (str): Path to json file,
        json file must contain train, val and test tags. data should be a list
        with sentence and intent label e.g.
        [how would you say fly in italian, translate]
        :param num_labels_to_sample (int): Number of labels to sample
        :param seed: seed for random
        :param padding_max_length (int): max length to pad sequences for constant
        length BERT input
        :param batch_size (int): batch size
        :param optimizer (str) : optimizer to use with BERT
        :param lr (float): learning rate to use with optimizer
        """

        self.model = \
            BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=num_labels_to_sample)
        self.model.cuda() if torch.cuda.is_available() else self.model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                       do_lower_case=True)
        self.device = device
        self.path_to_json = path_to_json
        self.seed = seed
        self.num_labels_to_sample = num_labels_to_sample
        self.padding_max_length = padding_max_length
        self.batch_size = batch_size
        self.sampled_intents = []
        self.train_dataloader, self.validation_dataloader, \
            self.prediction_dataloader = self._preprocess()
        self.optimizer = self._initialize_optimizer(optimizer, lr)
        self.epochs = epochs

    def _initialize_optimizer(self, optimizer, lr):
        # BERT fine-tuning parameters
        if optimizer == 'AdamW':
            optimizer = AdamW(self.model.parameters(),
                              lr=lr)
        return optimizer

    def _preprocess(self):
        """Return the train, validation and test dataloaders
        to use as input to BERT."""

        # load data using Python JSON module
        with open(self.path_to_json, 'r') as f:
            data = json.loads(f.read())

        df = pd.DataFrame.from_dict(data, orient='index')
        df = df.transpose()

        assert 'train' in df.columns, 'json does not contain train key'
        assert 'val' in df.columns, 'json does not contain val key'
        assert 'test' in df.columns, 'json does not contain test key'

        train_df = self._col2df(df, 'train')
        val_df = self._col2df(df, 'val')
        test_df = self._col2df(df, 'test')

        sampled_intents = self._sample_intents(self.seed, train_df['intent'],
                             self.num_labels_to_sample)

        self.sampled_intents = sampled_intents.copy()

        # select rows corresponding to sampled_intents
        train_df = self._select_rows(train_df, sampled_intents)
        val_df = self._select_rows(val_df, sampled_intents)
        test_df = self._select_rows(test_df, sampled_intents)

        # prepare the input to feed into BERT model

        # tokenize, pad and mask the features to feed into BERT model
        train_inputs, train_masks = self._transform_features(train_df.content)
        validation_inputs, validation_masks = self._transform_features(
            val_df.content)
        # tokenize, pad and mask the features to feed into BERT model
        prediction_inputs, prediction_masks = self._transform_features(
            test_df.content)

        # encode categorical intents to intergers
        train_labels = self._transform_labels(train_df.intent)
        validation_labels = self._transform_labels(val_df.intent)
        prediction_labels = self._transform_labels(test_df.intent)

        train_dataloader = self._create_dataloader(train_inputs, train_masks,
                                                   train_labels, RandomSampler,
                                                   self.batch_size)

        validation_dataloader = self._create_dataloader(validation_inputs,
                                                        validation_masks,
                                                        validation_labels,
                                                        SequentialSampler,
                                                        self.batch_size)

        prediction_dataloader = self._create_dataloader(prediction_inputs,
                                                        prediction_masks,
                                                        prediction_labels,
                                                        SequentialSampler,
                                                        batch_size=32)

        return train_dataloader, validation_dataloader, prediction_dataloader

    @staticmethod
    def _sample_intents(seed, intents, k):
        """Randomly sample k classes from intents."""
        random.seed(seed)
        sampled_intents = random.sample(list(intents.unique()), k=k)
        return sampled_intents

    @staticmethod
    def _select_rows(df, sampled_intents):
        """Choose data from df that corresponds to labels in sampled_intents."""
        return df.loc[df.intent.isin(sampled_intents)]

    @staticmethod
    def _col2df(df, column):
        """Split column into a dataframe with content and intent columns"""

        # check if there are any NaN values
        #print(f'{column} has NaN values: {df[column].isnull().values.any()}')
        # drop the rows with invalid values
        sliced_df = df[column].dropna()
        assert not any(df["train"].str.len() != 2), \
            'data must be in the form [sentence, intent]'
        # split list into content and intent columns
        sliced_df = pd.DataFrame(sliced_df.to_list(),
                                 columns=['content', 'intent'])
        return sliced_df

    @staticmethod
    def _tokenize(content, tokenizer):
        """Tokenize content using tokenizer."""
        # add special tokens for formatting
        sentences = ["[CLS] " + query + " [SEP]" for query in content]
        # tokenize using BERT tokenizer
        tokenized_texts = [tokenizer.tokenize(sentence) for sentence in
                           sentences]
        return tokenized_texts

    @staticmethod
    def _pad(tokenized_texts, tokenizer, seq_len):
        """Return mapped and padded input ids."""
        # Set the maximum sequence length.
        MAX_LEN = seq_len
        # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in
                     tokenized_texts]
        # Pad our input tokens
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                                  truncating="post", padding="post")
        return input_ids

    def _transform_features(self, content):

        # Tokenize with BERT tokenizer
        tokenized_texts = self._tokenize(content, self.tokenizer)

        # Use the BERT tokenizer to convert the tokens to their index numbers
        # in the BERT vocabulary
        input_ids = self._pad(tokenized_texts, self.tokenizer,
                              self.padding_max_length)

        attention_masks = self._mask(input_ids)

        return torch.tensor(input_ids), torch.tensor(attention_masks)

    @staticmethod
    def _mask(input_ids):
        """"Create attention masks of 1s for each
        token followed by 0s for padding."""
        # Create attention masks
        attention_masks = []
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)
        return attention_masks

    @staticmethod
    def _transform_labels(labels):
        """Encode categorical labels to numbers."""
        le = LabelEncoder()
        le.fit(labels)
        label_enc = le.transform(labels)
        return torch.tensor(label_enc)

    @staticmethod
    def _create_dataloader(inputs, masks, labels, sampler_fn, batch_size):
        """Create an iterator of our data with torch DataLoader."""
        data = TensorDataset(inputs, masks, labels)
        sampler = sampler_fn(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

        return dataloader

    @staticmethod
    def _flat_accuracy(preds, labels):
        """Calculate the accuracy of our predictions vs labels"""
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def save(self):

        output_dir = './model_save/'

        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)

        # Save a trained model, configuration and tokenizer using
        # `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = self.model.module if hasattr(self.model,
                                                     'module') else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load(self, output_dir):
        # Load a trained model and vocabulary that you have fine-tuned
        self.model = BertForSequenceClassification.from_pretrained(output_dir)
        self.tokenizer = BertTokenizer.from_pretrained(output_dir)

        # Copy the model to the GPU.
        self.model.cuda() if torch.cuda.is_available() else self.model

    def train(self):

        train_dataloader = self.train_dataloader
        validation_dataloader = self.validation_dataloader
        model = self.model
        device = self.device
        optimizer = self.optimizer
        train_loss_set = []
        # Number of training epochs
        epochs = self.epochs
        min_loss = float('inf')

        # BERT training loop
        for _ in range(epochs):
            print('epoch {}'.format(_ + 1))

            ## TRAINING

            # Set our model to training mode
            model.train()
            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):
                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                # Clear out the gradients (by default they accumulate)
                optimizer.zero_grad()
                # Forward pass
                output = model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask, labels=b_labels)
                loss = output[0]
                train_loss_set.append(loss.item())
                # print(len(train_loss_set))
                # Backward pass
                loss.backward()
                # Update parameters and take a step using the computed gradient
                optimizer.step()
                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
            print("Train loss: {}".format(tr_loss / nb_tr_steps))

            ## VALIDATION

            # Put model in evaluation mode
            model.eval()
            # Tracking variables
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            # Evaluate data for one epoch
            batch_loss = 0
            for batch in validation_dataloader:
                batch_loss = 0
                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    output = model(b_input_ids, token_type_ids=None,
                                   attention_mask=b_input_mask, labels=b_labels)
                    loss = output[0]
                    logits = output[1]

                batch_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                tmp_eval_accuracy = self._flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

            print(
                "Validation loss: {}".format(batch_loss / nb_eval_steps))
            if batch_loss < min_loss:
                print('Improvement-Detected, save-model')
                self.save()
                min_loss = batch_loss

            print(
                "Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
            print("==================================================")

        # plot training performance
        plt.figure(figsize=(15, 8))
        plt.title("Training loss")
        plt.xlabel("Total number of batches")
        plt.ylabel("Loss")
        plt.plot(train_loss_set)
        plt.show()

    def evaluate(self):
        """Return true and predicted labels."""
        model = self.model
        device = self.device
        prediction_dataloader = self.prediction_dataloader

        # Put model in evaluation mode
        model.eval()
        # Tracking variables
        predictions, true_labels = [], []
        # Predict
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory
            # and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                output = model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)
                logits = output[0]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

            # Flatten the predictions and true values for aggregate  evaluation
            # on the whole dataset
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        flat_true_labels = [item for sublist in true_labels for item in sublist]

        accuracy = np.sum(flat_predictions == flat_true_labels) / len(
            flat_true_labels)

        print('Classification accuracy using BERT Fine Tuning: {0:0.2%}'.format(
            accuracy))

        return flat_true_labels, flat_predictions

    def plot_confusion_matrix(self, flat_true_labels, flat_predictions):
        intents = self.sampled_intents
        cm = confusion_matrix(flat_true_labels,
                              flat_predictions)
        array = cm
        df_cm = pd.DataFrame(array, index=[i for i in intents],
                             columns=[i for i in intents])
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)

    def get_classification_report(self, flat_true_labels, flat_predictions):
        target_names = self.sampled_intents
        return classification_report(flat_true_labels, flat_predictions,
                                    target_names=target_names)

