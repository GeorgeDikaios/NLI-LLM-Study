import matplotlib.pyplot as plt
import torch
import re
import os
from huggingface_hub import login
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef, cohen_kappa_score, ConfusionMatrixDisplay

def get_lengths(df, tokenizer, dataset_type):
    """ Tokenizes the prompt and label, and returns two lists containing the lengths of each tokenized sequence.

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
    tuple of lists
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
        labels = "'neutral' or 'entailment'"
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


def get_predictions(outputs_decoded, dataset_type):
    predicted_labels = []

    for text in outputs_decoded:
        if dataset_type == "qnli":
            x = re.findall('[e|E]ntailment|[n|N]ot_entailment', text)
            try:
                predicted_labels.append(x[2].lower())
            except IndexError:
                no_answer += 1
                predicted_labels.append("no_answer")
        elif dataset_type == "mnli":
            x = re.findall('[e|E]ntailment|[c|C]ontradiction|[n|N]eutral', text)
            try:
                predicted_labels.append(x[3].lower())
            except IndexError:
                no_answer += 1
                predicted_labels.append("no_answer")
        elif dataset_type == "scitail":
            x = re.findall('[e|E]ntailment|[n|N]eutral', text)
            try:
                predicted_labels.append(x[2].lower())
            except IndexError:
                no_answer += 1
                predicted_labels.append("no_answer")
        else:
            raise ValueError(f"Invalid type: {dataset_type}. Choose one of 'mnli', 'qnli' or 'scitail'.")
    return predicted_labels


def test_run(model, dataloader, tokenizer, dataset_type):
    predictions = []
    batch_sample = next(iter(dataloader))
    input_ids = {k: v for k, v in batch_sample.items() if k != "labels" and k != "prompt"}
    gold_labels = batch_sample['labels']

    with torch.no_grad():
        outputs = model.generate(**input_ids, max_new_tokens=20)

    outputs_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch = get_predictions(outputs_decoded, dataset_type=dataset_type)
    predictions.extend(batch)
    
    return predictions, gold_labels


def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        predicted_labels = checkpoint["predicted_labels"]
        gold_labels = checkpoint["gold_labels"]
        no_answer = checkpoint["no_answer"]
        start_batch = checkpoint['batch_no']
        print(f"Checkpoint found. Continuing from batch #{start_batch}.")
    else:
        gold_labels, predicted_labels = [], []
        start_batch = 0
        no_answer = 0
        print("No checkpoint found.")
    return predicted_labels, gold_labels, no_answer, start_batch


def hf_login(token_name="HF_TOKEN"):
    """
    Detects environment and logins to HuggingFace. Can handle local from a .env file as well as secrets from Google Colab and Kaggle.

    Parameters
    ----------
    token_name: str
        The name of the token.

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
    

def evaluate_metrics(gold_labels, predicted_labels):
    cm = confusion_matrix(y_true=gold_labels, y_pred=predicted_labels)
    acc = accuracy_score(y_true=gold_labels, y_pred=predicted_labels)
    f1 = f1_score(y_true=gold_labels, y_pred=predicted_labels, average='macro')
    mcc = matthews_corrcoef(y_true=gold_labels, y_pred=predicted_labels)
    kappa = cohen_kappa_score(y1=gold_labels, y2=predicted_labels)
    
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=set(gold_labels))
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
            labels = "'not_entailment' or 'entailment'"
        elif self.dataset_type == "scitail":
            sentence1 = "Hypothesis"
            sentence2 = "Premise"
            labels = "'neutral' or 'entailment'"
        else:
            raise ValueError(f"Invalid type: {self.dataset_type}. Choose one of 'mnli', 'qnli' or 'scitail'.")
        
        prompt = (f"Does the {sentence1} entail the {sentence2}?" 
                  f"Answer exactly one word: {labels}. \n{sentence2}: {item[sentence2.lower()]} \n{sentence1}: {item[sentence1.lower()]} \nAnswer:")
        
        # Tokenise prompt
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.prompt_max_length,
            return_tensors="pt"
        ) 

        # Tokenise gold label for training
        if self.training:
            gold_label = self.tokenizer(
            gold_label,
            truncation=True,
            padding="max_length",
            max_length=self.label_max_length, # We found 6 to be the max_length of the labels
            return_tensors="pt"
            )

            return {"input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": gold_label["input_ids"].squeeze()}

        return {"input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": gold_label,
                "prompt": prompt}