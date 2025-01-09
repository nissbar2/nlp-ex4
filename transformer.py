import operator
import os
import pickle

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import data_loader

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300
HIDDEN_DIM = 100

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------


def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------


def load_word2vec():
    """Load Word2Vec Vectors
    Return:
        wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api

    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print(wv_from_bin.key_to_index[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=True):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim=W2V_EMBEDDING_DIM):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    text = sent.text
    vectors = []
    for word in text:
        if word in word_to_vec:
            vectors.append(word_to_vec[word])
    if len(vectors) == 0:
        # raise ValueError(f"shit: {text}")
        return np.zeros(embedding_dim, dtype=np.float32)
    result = np.mean(vectors, axis=0).astype(np.float32)
    return result


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    onehot = np.zeros(size, dtype=np.float32)
    onehot[ind] = 1
    return onehot


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    text = sent.text
    vectors = []
    for word in text:
        vectors.append(get_one_hot(len(word_to_ind), word_to_ind[word]))
    return np.mean(vectors, axis=0)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    words_list = sorted(words_list)
    return {word: words_list.index(word) for word in words_list}


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    vectors = np.zeros((seq_len, embedding_dim), dtype=np.float32)
    tokens = sent.text
    for i, token in enumerate(tokens[:seq_len]):
        if token in word_to_vec:
            vectors[i] = word_to_vec[token]
    return vectors


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager:
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(
        self,
        data_type=ONEHOT_AVERAGE,
        use_sub_phrases=False,
        dataset_path="stanfordSentimentTreebank",
        batch_size=50,
        embedding_dim=None,
    ):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(
            dataset_path, split_words=True
        )
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {
                "seq_len": SEQ_LEN,
                "word_to_vec": create_or_load_slim_w2v(words_list),
                "embedding_dim": embedding_dim,
            }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {
                "word_to_vec": create_or_load_slim_w2v(words_list),
                "embedding_dim": embedding_dim,
            }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {
            k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs)
            for k, sentences in self.sentences.items()
        }
        self.torch_iterators = {
            k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
            for k, dataset in self.torch_datasets.items()
        }

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_sent_words(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return [" ".join(sent.text) for sent in self.sentences[data_subset]]

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    return torch.sum(preds == y) / preds.shape[0]


def transformer_classification(portion=1.0):
    class Dataset(torch.utils.data.Dataset):
        """
        Dataset for loading data
        """

        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    def train_epoch(model, data_loader, optimizer, dev="cpu"):
        """
        Perform an epoch of training of the model with the optimizer
        :param model:
        :param data_loader:
        :param optimizer:
        :param dev:
        :return: Average loss over the epoch
        """
        model.train()
        total_loss = 0.0
        accuracy = 0.0
        # iterate over batches
        for batch in tqdm(data_loader):
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            labels = batch["labels"].to(dev)
            # print(type(input_ids))
            ########### add your code here ###########
            model.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(outputs.logits, dim=1)
                accuracy += binary_accuracy(pred, batch["labels"])

        epoch_mean_loss = total_loss / len(data_loader)
        epoch_mean_accuracy = accuracy / len(data_loader)
        return epoch_mean_loss, epoch_mean_loss

    def evaluate_model(model, data_loader, dev="cpu", metric=None):
        model.eval()

        epoch_validation_accuracy = 0.0
        epoch_validation_loss = 0.0
        predictions = []
        references = []
        for batch in tqdm(data_loader):
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            labels = batch["labels"].to(dev)
            ########### add your code here ###########
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1)
                ##predictions.append(pred.detach().cpu().numpy())
                ##references.append(labels.detach().cpu().numpy())
                epoch_validation_accuracy += accuracy_score(
                    labels.cpu().numpy(), pred.cpu().numpy()
                )
                epoch_validation_loss += outputs.loss.item()

        epoch_mean_validation_accuracy = epoch_validation_accuracy / len(data_loader)
        epoch_mean_validation_loss = epoch_mean_loss / len(data_loader)
        return epoch_mean_validation_loss, epoch_mean_validation_accuracy

    data_manager = DataManager(data_type=W2V_SEQUENCE, batch_size=64)
    # x_train, y_train, x_test, y_test = get_data(
    #     categories=category_dict.keys(), portion=portion
    # )
    x_train = data_manager.get_sent_words(TRAIN)
    y_train = data_manager.get_labels(TRAIN)
    x_test = data_manager.get_sent_words(TEST)
    y_test = data_manager.get_labels(TEST)

    # Parameters
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = 2
    epochs = 2
    batch_size = 64
    learning_rate = 1e-5

    # Model, tokenizer, and metric
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilroberta-base", num_labels=num_labels, cache_dir="./transformer_cache"
    ).to(dev)
    tokenizer = AutoTokenizer.from_pretrained(
        "distilroberta-base", cache_dir="./tokenizer_cache"
    )
    metric = evaluate.load("accuracy")

    # Datasets and DataLoaders
    train_dataset = Dataset(tokenizer(x_train, truncation=True, padding=True), y_train)
    val_dataset = Dataset(tokenizer(x_test, truncation=True, padding=True), y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    ########### add your code here ###########
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, dev)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_loss, val_acc = evaluate_model(model, val_loader, dev, metric)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    return train_loss, train_accuracies, val_losses, val_accuracies, model


if __name__ == "__main__":
    EPOCHS = 2
    (
        train_loss,
        train_accuracy,
        val_loss,
        val_accuracy,
        model,
    ) = transformer_classification()

    x = [str(a + 1) for a in range(EPOCHS)]
    plt.plot(x, train_loss, label="train loss", c="blue")
    plt.plot(x, val_loss, label="validation loss", c="orange")
    plt.legend()
    plt.title("Loss as function of epochs")
    plt.xlabel("#epochs")
    plt.ylabel("Mean Loss")
    plt.show()

    plt.plot(x, train_accuracy, label="train accuracy", c="blue")
    plt.plot(x, val_accuracy, label="validation accuracy", c="orange")
    plt.legend()
    plt.title("Accuracy as function of epochs")
    plt.xlabel("#epochs")
    plt.ylabel("Mean Accuracy")
    plt.show()

    criterion = nn.BCEWithLogitsLoss()
    mean_test_loss, mean_test_accuracy = evaluate_model(
        model, data_manager.torch_iterators[TEST], criterion
    )
    print("Test loss ", mean_test_loss)
    print("Test accuracy ", mean_test_accuracy.item())
    indices = data_loader.get_negated_polarity_examples(data_manager.sentences[TEST])
    subset = Subset(data_manager.torch_datasets[TEST], indices)
    dataloader = DataLoader(subset, batch_size=64)
    predictions, y = get_predictions_for_data(model, dataloader)
    success_rate = binary_accuracy(predictions, y)
    print("Test Negated Polarity Accuracy ", success_rate.item())

    indices = data_loader.get_rare_words_examples(
        data_manager.sentences[TEST], data_manager.sentiment_dataset
    )
    subset = Subset(data_manager.torch_datasets[TEST], indices)
    dataloader = DataLoader(subset, batch_size=64)
    predictions, y = get_predictions_for_data(model, dataloader)
    success_rate = binary_accuracy(predictions, y)
    print("Test Rare Words Accuracy ", success_rate.item())
