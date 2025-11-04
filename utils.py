import matplotlib.pyplot as plt
import torch
import re
import os
import pandas
from typing import Any, Tuple, List
from huggingface_hub import login
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef, cohen_kappa_score, ConfusionMatrixDisplay

def get_lengths(df: pandas.DataFrame, tokenizer: Any, dataset_type) -> Tuple[List[int], List[int]]:
    """ 
    Tokenizes the prompt and label, and returns two lists containing the lengths of each tokenized sequence.

    Parameters
    ----------
    df: pandas.DataFrame
        A DataFrame containing sentence1, sentence2 and label
    tokenizer: Any
        A tokenizer instance via AutoTokenizer.from_pretrained().
    type: str
        The type of dataset.
            - "mnli"
            - "qnli"
            - "scitail"

    Returns
    -------
    Tuple[List[int], List[int]]
        - prompt_token_lengths (List[int]): Lengths of prompts
        - label_token_lengths (List[int]): Lengths of labels
    """

    prompt_token_lengths = []
    label_token_lengths = []

    if dataset_type == "mnli":
        sentence1 = "Hypothesis"
        sentence2 = "Premise"
        labels = "'contradiction', 'neutral' or 'entailment'"
    elif dataset_type == "qnli":
        sentence1 = "Sentence"
        sentence2 = "Question"
        labels = "'not_entailment' or 'entailment'"
    elif dataset_type == "scitail":
        sentence1 = "Hypothesis"
        sentence2 = "Premise"
        labels = "'neutral' or 'entails'"
    else:
        raise ValueError(f"Invalid type: {dataset_type}. Choose one of 'mnli', 'qnli' or 'scitail'.")
    
    for _, row in df.iterrows():
        # Create the prompt
        prompt = f"Does the {sentence1} entail the {sentence2}? \
            Answer exactly one word: {labels}. \n{sentence2}: {row[sentence2.lower()]} \n{sentence1}: {row[sentence1.lower()]} \nAnswer:"
        
        # Tokenize the prompt
        prompt_token = tokenizer(prompt)
        prompt_token_lengths.append(len(prompt_token["input_ids"]))

        # Tokenize label
        label_token = tokenizer(row["label"])
        label_token_lengths.append(len(label_token["input_ids"]))
    return prompt_token_lengths, label_token_lengths


def find_max_length(df, tokenizer, dataset_type) -> None:
    """
    Plots a histogram of the prompt lengths and prints the max size.

    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame containing sentence1, sentence2 and label
    tokenizer: PreTrainedTokenizerBase
        A tokenizer instance via AutoTokenizer.from_pretrained().
    type: str
        The type of dataset.
            - "mnli"
            - "qnli"
            - "scitail"

    Returns
    -------
    """
    prompt_lengths, label_lengths = get_lengths(df, tokenizer, dataset_type)
    plt.hist(prompt_lengths, bins=50)
    plt.show()
        
    print("Max prompt length:", max(prompt_lengths))
    print("Max label length:", max(label_lengths))


def test_run(model: Any, dataloader: Any, tokenizer: Any, dataset_type: str) -> Tuple[List[str], List[str]]:
    """
    Generates predictions using a single batch for testing.

    Parameters
    ----------
    model: Any
        An instance of a model.
    dataloader: Any
        An instance of a dataloader.
    tokenizer:
        An instance of a tokenizer.
    dataset_type: str
        The type of the dataset.

    Returns
    -------
    Tuple[List[str], List[str]]
        A tuple containing two lists:
            - predictions: A list of strings containing the predictions of the model.
            - gold_labels: A list of strings containing the equivalent gold labels.
    """
    labels = get_labels(dataset_type=dataset_type)
    batch = next(iter(dataloader))

    input_ids_batch = batch["input_ids"].to(model.device) # Move to GPU
    attention_mask_batch = batch["attention_mask"].to(model.device) # Move to GPU
    gold_labels_batch = batch["labels"] # Keep to CPU

    batch_probs = get_model_probs(batch_input_ids=input_ids_batch,
                                  batch_attention_mask=attention_mask_batch,
                                  dataset_type='scitail',
                                  model=model,
                                  tokenizer=tokenizer)
    
    batch_pred_indices = torch.argmax(batch_probs)
    batch_pred_labels = [labels[i] for i in batch_pred_indices]
    
    return batch_pred_labels, gold_labels_batch, batch_probs


def load_checkpoint(checkpoint_path):
    """
    Loads the file with checkpoint using torch.

    Parameters
    ----------
    checkpoint_path: str
        The path to the file.

    Returns
    -------
    predicted_labels: list
        A list with the predictions of the model.
    gold_labels: list
        A list with the gold labels.
    no_answer: int
        The number of times the model was unable to give an answer.
    start_batch: int
        The batch where the evaluation will continue from.
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        predicted_labels = checkpoint["predicted_labels"]
        gold_labels = checkpoint["gold_labels"]
        start_batch = checkpoint['batch_no']
        print(f"Checkpoint found.")
    else:
        gold_labels, predicted_labels = [], []
        start_batch = 0
        print("No checkpoint found.")
    return predicted_labels, gold_labels, start_batch


def hf_login(token_name: str = "HF_TOKEN") -> None:
    """
    Detects environment and logins to HuggingFace. Can handle local from a .env file as well as secrets from Google Colab and Kaggle.

    Parameters
    ----------
    token_name: str
        The name of the token. 'HF_TOKEN' by default

    Returns
    -------
    None
    """
    from dotenv import load_dotenv
    load_dotenv()
    token = os.getenv(token_name)

    if token is None:
        try:
            from google.colab import userdata
            token = userdata.get(token_name)
        except:
            pass

        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            token = user_secrets.get_secret(token_name)
        except:
            pass

    login(token=token)
    

def evaluate_metrics(gold_labels: list, predicted_labels: list, dataset_type: str) -> None:
    """
    Evaluates and displays the following metrics: Accuracy, F1-Score, Matthew;s Correlation Coefficient, Cohen's Kappa.
    Also plots the confusion matrix.

    Parameters
    ----------
    gold_labels: list
        A list of the gold labels
    predicted_labels: list
        A list with the labels that were predicted
    dataset_type: str
        The type of the dataset. One of 'qnli', 'mnli' and 'scitail'.

    Returns
    -------
    None
    """
    acc = accuracy_score(y_true=gold_labels, y_pred=predicted_labels)
    f1 = f1_score(y_true=gold_labels, y_pred=predicted_labels, average='macro')
    mcc = matthews_corrcoef(y_true=gold_labels, y_pred=predicted_labels)
    kappa = cohen_kappa_score(y1=gold_labels, y2=predicted_labels)
    
    display_labels = get_labels(dataset_type=dataset_type)
    
    cm = confusion_matrix(y_true=gold_labels, y_pred=predicted_labels, labels=display_labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    cm_display.plot(cmap="Blues")
    print(f"Accuracy: {acc:.4f}.\n",
          f"F1 Score: {f1:.4f}.\n",
          f"Matthew's Correlation Coefficient: {mcc:.4f}.\n",
          f"Cohen's Kappa Score: {kappa:.4f}.")


class MyDataset(Dataset):

    def __init__(self, dataframe, tokenizer, dataset_type, prompt_max_length, label_max_length, training=False):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.prompt_max_length = prompt_max_length
        self.label_max_length = label_max_length
        self.training = training
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        gold_label = item['label']

        # Format the prompt depending on dataset_type
        if self.dataset_type == "mnli":
            sentence1 = "Hypothesis"
            sentence2 = "Premise"
            labels = "'contradiction', 'neutral' or 'entailment'"
        elif self.dataset_type == "qnli":
            sentence1 = "Sentence"
            sentence2 = "Question"
            labels = "'not entailment' or 'entailment'"
        elif self.dataset_type == "scitail":
            sentence1 = "Hypothesis"
            sentence2 = "Premise"
            labels = "'neutral' or 'entails'"
        else:
            raise ValueError(f"Invalid type: {self.dataset_type}. Choose one of 'mnli', 'qnli' or 'scitail'.")
        
        prompt = (f"Does the {sentence1} entail the {sentence2}? "
                  f"Answer exactly one word in lowercase: {labels}. \n{sentence2}: {item[sentence2.lower()]} \n{sentence1}: {item[sentence1.lower()]} \nAnswer:")
        
        # Tokenise prompt
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            add_special_tokens=False,
            max_length=self.prompt_max_length,
            return_tensors="pt"
        ) 

        # Tokenise gold label for training
        
        gold_label_ids = self.tokenizer(
        gold_label,
        truncation=True,
        padding="max_length",
        max_length=self.label_max_length, # We found 6 to be the max_length of the labels
        return_tensors="pt"
        )

        return {"input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels_ids": gold_label_ids["input_ids"].squeeze(),
            "labels": gold_label,
            "prompt": prompt}
    

def detect_env() -> str:
    """
    Detects the environment. Works for Google Colab, Kaggle and local environments.

    Returns
    -------
    str
        one of 'colab', 'kaggle', 'local'
    """
    try:
        import google.colab
        return 'colab'
    except ImportError:
        pass

    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
    
    return 'local'


def create_checkpoint_path(model_id: str, name: str) -> str:
    """
    Creates a path string to a checkpoint file.

    If run in Google colab, it creates a folder named 'eval_checkpoints' to save the file there.
    If run in Kaggle, file is saved in '/kaggle/working'.
    If run locally, file is saved in the current worrking folder.

    Parameters
    ----------
    model_id: str
        The name of the model being used.
    name: str
        The type of evaluation being done. Example: 'scitail_zero_shot'

    Returns
    -------
    str:
        The path to the file.
    """

    env = detect_env()
    filename = f"checkpoint_{name}_{model_id.split('/')[1]}.pt".replace('-', '_')

    if env == 'colab':
        checkpoint_dir = "/content/drive/MyDrive/eval_checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, filename)
    elif env == 'kaggle':
        checkpoint_dir = "/kaggle/working"
        checkpoint_path = os.path.join(checkpoint_dir, filename)
    elif env == 'local':
        checkpoint_dir = os.getcwd()
        checkpoint_path = os.path.join(checkpoint_dir, filename)

    print('Saving to:', checkpoint_path)
    return checkpoint_path


def get_labels(dataset_type: str) -> List[str]:
    """
    Return a list of the class labels given a dataset_type.

    Parameters
    ----------
    dataset_type: str
        The type of the dataset. One of 'mnli', 'qnli', 'scitail'.

    Returns
    -------
    labels: List[str]
        The labels associated with the specified dataset.
    """
    if dataset_type == "mnli":
        labels = ['contradiction', 'neutral', 'entailment']
    elif dataset_type == "qnli":
        labels = ['entailment', 'not entailment']
    elif dataset_type == 'scitail':
        labels = ['entails', 'neutral']
    else:
        raise ValueError(f"Invalid type: {dataset_type}. Choose one of 'mnli', 'qnli' or 'scitail'.")
    return labels


def get_model_probs(batch_input_ids: List, batch_attention_mask: List, model: Any, tokenizer: Any, dataset_type: str) -> torch.Tensor:
    """
    Gets as input a batch and gives as output the probabilities of each label. The size of the output depends on the dataset_type specified.
    """
    labels = get_labels(dataset_type=dataset_type)
    batch_size = batch_input_ids.size(0)

    # Tokenize target labels
    label_ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    
    probs = torch.zeros(batch_size, len(labels))
    # Loop over each example
    for i in range(batch_size):
        input_ids = batch_input_ids[i].unsqueeze(0)
        attention_mask = batch_attention_mask[i].unsqueeze(0)

        # Loop over each label
        for j, label_tokens in enumerate(label_ids):
            p = 1.0
            generated_ids = input_ids.clone()
            generated_mask = attention_mask.clone()

            # Loop over each token
            for tid in label_tokens:
                with torch.no_grad():
                      outputs = model(input_ids=generated_ids, attention_mask=generated_mask)
                      next_token_logits = outputs.logits[:, -1, :]
                      next_token_probs = F.softmax(next_token_logits, dim=-1)

                      # Get the probability
                      p *= next_token_probs[0, tid].item()
                
                # Feed the chosen token as next input to get next token prob
                generated_ids = torch.cat([generated_ids, torch.tensor([[tid]], device=input_ids.device)], dim=-1)
                generated_mask = torch.cat([generated_mask, torch.ones(1, len(label_tokens), device=attention_mask.device)], dim=-1)
                
                
                
            # Update the probs tensor of ith example and jth label
            probs[i,j] = p
    return probs