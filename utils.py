import matplotlib.pyplot as plt
import torch
import os
import numpy
import pandas
from typing import Any, Tuple, List
from huggingface_hub import login
from torch.utils.data import Dataset
import torch.nn.functional as F
# from captum.attr import visualization as viz
# from captum.attr import InterpretableEmbeddingBase
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef, cohen_kappa_score, ConfusionMatrixDisplay, recall_score, precision_score

def make_prompt(row, dataset_type, kind, model_type, examples=None):
    dataset_type = dataset_type.split('_')[0]
    system_content, user_content = None, None

    # Format the prompt depending on dataset_type
    if dataset_type == "mnli":
        sentence1 = "premise"
        sentence2 = "hypothesis"
        labels = "'contradiction', 'neutral' or 'entailment'"
    elif dataset_type == "qnli":
        sentence1 = "Sentence"
        sentence2 = "Question"
        labels = "'not entailment' or 'entailment'"
    elif dataset_type == "scitail":
        sentence1 = "premise"
        sentence2 = "hypothesis"
        labels = "'neutral' or 'entails'"
    else:
        raise ValueError(f"Invalid type: {dataset_type}. Choose one of 'mnli', 'qnli' or 'scitail'.")

    if kind == 'zero_shot':
        system_content = f"You are an NLI classifier. Does the {sentence1} entail the {sentence2}? Output only one word in lowercase: {labels}."
        if 'llama' in model_type.lower():
            user_content = (f"{sentence1}: {row[sentence1.lower()]} \n,"
                            f"{sentence2}: {row[sentence2.lower()]} \n,"
                            f"Answer:")
        else:
            user_content = (f"Does the {sentence1} entail the {sentence2}?\n"
                            f"{sentence1}: {row[sentence1.lower()]}\n"
                            f"{sentence2}: {row[sentence2.lower()]}\n"
                            f"Answer:")
    elif kind == "few_shot":
        if 'llama' in model_type.lower():
            system_content = f"You are an NLI classifier. Given the following examples, does the {sentence1} entail the {sentence2} in the next case? Answer only one word in lowercase as in the examples: {labels}."
            user_content = (f"Examples:\n{examples}\n",
                            f"{sentence1}: {row[sentence1.lower()]} \n",
                            f"{sentence2}: {row[sentence2.lower()]} \n",
                            f"Answer:")
        else:
            user_content = (f"Examples:\n{examples} \nGiven the above examples as reference does the {sentence1} entail the {sentence2} in the following case? "
                            f"Answer exactly one word in lowercase as in the examples: {labels}. \n{sentence1}: {row[sentence1.lower()]} \n{sentence2}: {row[sentence2.lower()]} \nAnswer:")
    elif kind == "with_def":
        if 'llama' in model_type.lower():
            system_content = (f"You are an NLI classifier. Given the definitions for each label:",
                              f"does the {sentence1} entail the {sentence2}? Output only one word in lowercase: {labels}.")
            user_content = f"{sentence1}: {row[sentence1.lower()]} \n{sentence2}: {row[sentence2.lower()]} \nAnswer:"
        else:
            user_content = (f"Given the definitions for each label:\n"
                            f"Entailment: The hypothesis must be true if the premise is true.\n"
                            f"Neutral: The truth of the hypothesis cannot be determined from the premise.\n"
                            f"Does the {sentence1} entail the {sentence2}?\nAnswer exactly one word in lowercase: {labels}.\n"
                            f"{sentence1}: {row[sentence1.lower()]} \n{sentence2}: {row[sentence2.lower()]} \nAnswer:")
    else:
        raise ValueError(f"Unknown kind: {kind}")
    if "llama" in model_type.lower():
        return [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
    else:
        return [{"role": "user", "content": user_content}]
        
    
            

def get_lengths(df: pandas.DataFrame, tokenizer: Any, dataset_type: str, chat_template: bool, examples: str = None, kind: str = 'zero_shot', model_type: str = 'llama') -> Tuple[List[int], List[int]]:
    """
    Tokenizes the prompts and labels, returns lists of token lengths.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing sentence1, sentence2, and label.
    tokenizer : Any
        Tokenizer instance (from AutoTokenizer).
    dataset_type : str
        One of 'mnli', 'qnli', or 'scitail'.
    chat_template : bool
        Whether to use apply_chat_template or normal tokenization.
    examples : str
        Few-shot examples to include if kind='few_shot'.
    kind : str
        'zero_shot', 'few_shot', 'CoT', or 'MP'.

    Returns
    -------
    Tuple[List[int], List[int]]
        - prompt_token_lengths: List of token lengths for each prompt
        - label_token_lengths: List of token lengths for each label
    """
    # Build all prompts in a list
    prompts = []
    for _, row in df.iterrows():
        system_content, user_content = make_prompt(row, dataset_type, kind, model_type, examples)
        prompts.append(user_content)

    if chat_template:
        # Apply chat template to all prompts
        prompt_token_lengths = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
            tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            prompt_token_lengths.append(tokens["input_ids"].size(1))  # real sequence length
    else:
        # Batch tokenize directly
        tokenized = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors=None  # returns list of lists
        )
        prompt_token_lengths = [len(x) for x in tokenized["input_ids"]]

    labels = get_labels(dataset_type=dataset_type)
    tokenized_labels = tokenizer(
        labels,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_tensors=None
    )
    label_token_lengths = [len(x) for x in tokenized_labels["input_ids"]]

    return prompt_token_lengths, label_token_lengths


def find_max_length(df: Any, tokenizer: Any, dataset_type: str, model_type: str, chat_template: bool, examples: str = None,
                     kind: str = 'zero_shot', fraction: float = 0.002) -> None:
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
    prompt_lengths, label_lengths = get_lengths(df, tokenizer, dataset_type, chat_template, examples, kind, model_type)
    
    plt.hist(prompt_lengths, bins=50)
    plt.show()
        
    print("Max prompt length:", max(prompt_lengths))
    print("Max label length:", max(label_lengths))
    compute_safe_max_length(df, tokenizer, dataset_type, chat_template, examples=examples, kind=kind, fraction=fraction)

def compute_safe_max_length(df: Any, tokenizer: Any, dataset_type: str, chat_template: bool,
                            examples=None, kind: str = 'zero_shot', fraction: float = 0.002) -> int:
    """
    Computes a prompt max length that would truncate at most a given fraction of examples.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing sentence1, sentence2, and label.
    tokenizer : Any
        Tokenizer instance (from AutoTokenizer).
    dataset_type : str
        One of 'mnli', 'qnli', or 'scitail'.
    chat_template : bool
        Whether prompts use chat template tokenization.
    kind : str
        Prompt type: 'zero_shot', 'few_shot', 'CoT', or 'MP'.
    fraction : float
        Fraction of examples allowed to be truncated (default 0.002 = 0.2%).

    Returns
    -------
    max_length : int
        Suggested max token length for prompts.
    """
    # Get token lengths without truncation
    prompt_lengths, _ = get_lengths(df, tokenizer, dataset_type, chat_template, examples=examples, kind=kind)

    lengths = numpy.array(prompt_lengths)
    # Compute the percentile corresponding to 1 - fraction
    max_length = int(numpy.percentile(lengths, 100 * (1 - fraction)))

    n_truncated = numpy.sum(lengths > max_length)
    percent_truncated = n_truncated / len(lengths) * 100

    print(f"Safe prompt_max_length: {max_length} tokens")
    print(f"This would truncate {n_truncated} examples ({percent_truncated:.3f}%) out of {len(lengths)}")

    return max_length

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
    dataset_type=dataset_type.split('_')[0]
    labels = get_labels(dataset_type=dataset_type)
    batch = next(iter(dataloader))

    input_ids_batch = batch["input_ids"].to(model.device) # Move to GPU
    attention_mask_batch = batch["attention_mask"].to(model.device) # Move to GPU
    gold_labels_batch = batch["labels"] # Keep to CPU

    batch_probs = get_model_probs(batch_input_ids=input_ids_batch,
                                  batch_attention_mask=attention_mask_batch,
                                  dataset_type=dataset_type,
                                  model=model,
                                  tokenizer=tokenizer)
    
    batch_pred_indices = torch.argmax(batch_probs, dim=1)
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
    

def evaluate_metrics(gold_labels: list, predicted_labels: list, params: dict) -> None:
    """
    Evaluates and displays the following metrics: Accuracy, F1-Score, Matthew;s Correlation Coefficient, Cohen's Kappa.
    Also plots the confusion matrix.

    Parameters
    ----------
    gold_labels: list
        A list of the gold labels
    predicted_labels: list
        A list with the labels that were predicted
    params: dict
        A dictionary containing information regarding 'dataset_type', 'model_id', 'quantization' and 'training_mode'

    Returns
    -------
    None
    """
    acc = accuracy_score(y_true=gold_labels, y_pred=predicted_labels)
    precision = precision_score(y_true=gold_labels, y_pred=predicted_labels, average='macro')
    recall = recall_score(y_true=gold_labels, y_pred=predicted_labels, average='macro')
    f1 = f1_score(y_true=gold_labels, y_pred=predicted_labels, average='macro')
    mcc = matthews_corrcoef(y_true=gold_labels, y_pred=predicted_labels)
    kappa = cohen_kappa_score(y1=gold_labels, y2=predicted_labels)
    
    display_labels = get_labels(dataset_type=params['dataset_type'].split('_')[0])
    
    print(f"Accuracy: {acc:.4f}.\n",
          f"Precision: {precision:.4f}.\n"
          f"Recall: {recall:.4f}.\n"
          f"F1 Score: {f1:.4f}.\n",
          f"Matthew's Correlation Coefficient: {mcc:.4f}.\n",
          f"Cohen's Kappa Score: {kappa:.4f}.")
    
    cm = confusion_matrix(y_true=gold_labels, y_pred=predicted_labels, labels=display_labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    cm_display.plot(cmap="Blues")
    plt.title(f"model: {params['model_id']} \n quantized: {params['quantization']} \n training mode: {params['training_mode']}")

def get_metrics(gold_labels: list, predicted_labels: list, params: dict, ax) -> None:
    """
    Evaluates and displays the following metrics: Accuracy, F1-Score, Matthew;s Correlation Coefficient, Cohen's Kappa.
    Also plots the confusion matrix.

    Parameters
    ----------
    gold_labels: list
        A list of the gold labels
    predicted_labels: list
        A list with the labels that were predicted
    params: dict
        A dictionary containing information regarding 'dataset_type', 'model_id', 'quantization' and 'training_mode'

    Returns
    -------
    None
    """
    acc = accuracy_score(y_true=gold_labels, y_pred=predicted_labels)
    precision = precision_score(y_true=gold_labels, y_pred=predicted_labels, average='macro')
    recall = recall_score(y_true=gold_labels, y_pred=predicted_labels, average='macro')
    f1 = f1_score(y_true=gold_labels, y_pred=predicted_labels, average='macro')
    mcc = matthews_corrcoef(y_true=gold_labels, y_pred=predicted_labels)
    kappa = cohen_kappa_score(y1=gold_labels, y2=predicted_labels)
    
    display_labels = get_labels(dataset_type=params['dataset_type'].split('_')[0])

    cm = confusion_matrix(
        y_true=gold_labels,
        y_pred=predicted_labels,
        labels=display_labels
    )

    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    cm_display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{params['training_mode']}", fontsize=10)

    return acc, precision, recall, f1, mcc, kappa


class MyDataset(Dataset):

    def __init__(self, dataframe, tokenizer, dataset_type, model_type, prompt_max_length, label_max_length, chat_template, kind, examples, training=False):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type.split('_')[0]
        self.model_type = model_type
        self.prompt_max_length = prompt_max_length
        self.label_max_length = label_max_length
        self.training = training
        self.chat_template = chat_template
        self.kind = kind
        self.examples = examples
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        gold_label = item['label']
        system_content, user_content = make_prompt(row=item, dataset_type=self.dataset_type, kind=self.kind, examples=self.examples, model_type=self.model_type)

        # Tokenise prompt
        if self.chat_template:
            if 'llama' in self.model_type:
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
            else:
                messages = [
                    {"role": "user", "content": user_content}
                ]

            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            encoding = self.tokenizer(
                formatted_prompt,
                truncation=True,
                padding='max_length',
                max_length=self.prompt_max_length,
                return_tensors="pt"
            )
        else:
            encoding = self.tokenizer(
                user_content,
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
        max_length=self.label_max_length,
        add_special_tokens=False,
        return_tensors="pt"
        )
        
        return {"input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels_ids": gold_label_ids["input_ids"].squeeze(),
            "labels": gold_label,
            "prompt": user_content,
            "formatted_prompt": formatted_prompt if self.chat_template else None}
    

def detect_env() -> str:
    """
    Detects the environment. Works for Google Colab, Kaggle and local environments.

    Returns
    -------
    str
        one of 'colab', 'kaggle', 'local'
    """
    
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
    
    try:
        import google.colab
        return 'colab'
    except ImportError:
        pass
    return 'local'


def create_checkpoint_path(params: dict) -> str:
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
    model_id = params['model_id']
    dataset_type = params['dataset_type']
    quantization = params['quantization']
    training_mode = params['training_mode'].replace(' ', '_')

    filename = f"checkpoint_{dataset_type}_{model_id.split('/')[1]}_{quantization}_{training_mode}.pt".replace('-', '_')

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
    dataset_type=dataset_type.split('_')[0]
    if dataset_type == "mnli":
        labels = ['contradiction', 'neutral', 'entailment']
    elif dataset_type == "qnli":
        labels = ['entailment', 'not entailment']
    elif dataset_type == 'scitail':
        labels = ['entails', 'neutral']
    else:
        raise ValueError(f"Invalid type: {dataset_type}. Choose one of 'mnli', 'qnli' or 'scitail'.")
    return labels


def get_model_probs(batch_input_ids: torch.Tensor, batch_attention_mask: torch.Tensor,
                    model: Any, tokenizer: Any, dataset_type: str) -> torch.Tensor:
    """
    Gets as input a batch and gives as output the probabilities of each label.
    This version uses past_key_values to avoid recomputing the prompt for every label token.
    """
    dataset_type = dataset_type.split('_')[0]
    labels = get_labels(dataset_type=dataset_type)
    batch_size = batch_input_ids.size(0)

    # Tokenize target labels
    label_ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]

    log_p_list = []

    # Loop over each label
    for label_tokens in label_ids:
        log_p = torch.zeros(batch_size, device=batch_input_ids.device, dtype=torch.float32)

        with torch.no_grad():
            # Run the prompt once
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                use_cache=True
            )

            past = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            # Loop over each token in the label
            for tid in label_tokens:
                log_probs = F.log_softmax(next_token_logits.float(), dim=-1)
                log_p += log_probs[:, tid]

                # feed only the current token
                tid_batch = torch.full(
                    (batch_size, 1),
                    tid,
                    device=batch_input_ids.device,
                    dtype=batch_input_ids.dtype
                )

                outputs = model(
                    input_ids=tid_batch,
                    past_key_values=past,
                    use_cache=True
                )

                past = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]

        log_p_list.append(log_p)

    # Normalize across labels
    log_p_tensor = torch.stack(log_p_list).float()
    log_p_tensor -= torch.logsumexp(log_p_tensor, dim=0, keepdim=True)
    probs = torch.exp(log_p_tensor.T)
    return probs



def get_flant5_probs(batch_input_ids: torch.Tensor, batch_attention_mask: torch.Tensor,
                    model: Any, tokenizer: Any, dataset_type: str) -> torch.Tensor:
    """
    Gets as input a batch and gives as output the probabilities of each label.
    This version uses past_key_values to avoid recomputing the prompt for every label token.
    """
    dataset_type = dataset_type.split('_')[0]
    labels = get_labels(dataset_type=dataset_type)
    batch_size = batch_input_ids.size(0)

    # Tokenize target labels
    label_ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]

    log_p_list = []

    # Precompute encoder outputs once for the batch
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask
        )

    # Loop over each label
    for label_tokens in label_ids:
        # Initialize the decoder input ids
        decoder_input_ids = torch.full((batch_size, 1),
                                        model.config.decoder_start_token_id,
                                        device=batch_input_ids.device,
                                        dtype=torch.long)
        
        log_p = torch.zeros(batch_size, device=batch_input_ids.device, dtype=torch.float32)
        
        past = None
        
        for tid in label_tokens:
            with torch.no_grad():
                # Run the prompt once
                outputs = model(
                    encoder_outputs=encoder_outputs,
                    attention_mask=batch_attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    use_cache=True,
                    past_key_values=past
                )

                # Next token logits from decoder
                next_token_logits = outputs.logits[:, -1, :].float()
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                log_p += log_probs[:, tid] / len(label_tokens)

                # Update past_key_values
                past = outputs.past_key_values

                # Prepare decoder input for next token
                decoder_input_ids = torch.full((batch_size, 1), tid,
                                                device=batch_input_ids.device,
                                                dtype=torch.long)
                
        log_p_list.append(log_p)

    # Normalize across labels
    log_p_tensor = torch.stack(log_p_list)
    log_p_tensor -= torch.logsumexp(log_p_tensor, dim=0, keepdim=True)
    probs = torch.exp(log_p_tensor.T)
    return probs


# def get_model_probs(batch_input_ids: List, batch_attention_mask: List, model: Any, tokenizer: Any, dataset_type: str) -> torch.Tensor:
#     """
#     Gets as input a batch and gives as output the probabilities of each label. The size of the output depends on the dataset_type specified.
#     """
#     dataset_type=dataset_type.split('_')[0]
#     labels = get_labels(dataset_type=dataset_type)
#     batch_size = batch_input_ids.size(0)

#     # Tokenize target labels
#     label_ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    
#     log_p_list = []
#     # Loop over each label
#     for label_tokens in label_ids:
#         log_p = torch.zeros(batch_size, device=batch_input_ids.device, dtype=torch.float32)
#         batch_generated_ids = batch_input_ids.clone()
#         batch_generated_mask = batch_attention_mask.clone()

#         # Loop over each token
#         for tid in label_tokens:
#             with torch.no_grad():
#                 outputs = model(input_ids=batch_generated_ids, attention_mask=batch_generated_mask)
#                 next_token_logits = outputs.logits[:, -1, :]
#                 log_probs = F.log_softmax(next_token_logits.float(), dim=-1)

#                 # Get the probability
#                 log_p += log_probs[:, tid]

#             # Feed the chosen token as next input to get next token prob to the whole batch
#             tid_batch = torch.full((batch_size, 1), tid, device=batch_input_ids.device, dtype=batch_input_ids.dtype)
#             batch_generated_ids = torch.cat([batch_generated_ids, tid_batch], dim=-1)
#             batch_generated_mask = torch.cat([batch_generated_mask, torch.ones(batch_size, 1, device=batch_attention_mask.device)], dim=-1)
            
#         log_p_list.append(log_p)

    # # Normalise
    # log_p_tensor = torch.stack(log_p_list).float()
    # log_p_tensor -= torch.logsumexp(log_p_tensor, dim=0, keepdim=True)
    # probs = torch.exp(log_p_tensor.T)
    # return probs


class MyDataset_def(Dataset):

    def __init__(self, dataframe, tokenizer, dataset_type, prompt_max_length, label_max_length, chat_template, training=False):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type.split('_')[0]
        self.prompt_max_length = prompt_max_length
        self.label_max_length = label_max_length
        self.training = training
        self.chat_template = chat_template
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        gold_label = item['label']

        # Format the prompt depending on dataset_type
        if self.dataset_type == "mnli":
            sentence1 = "premise"
            sentence2 = "hypothesis"
            labels = "'contradiction', 'neutral' or 'entailment'"
        elif self.dataset_type == "qnli":
            sentence1 = "Sentence"
            sentence2 = "Question"
            labels = "'not entailment' or 'entailment'"
        elif self.dataset_type == "scitail":
            sentence1 = "premise"
            sentence2 = "hypothesis"
            labels = "'neutral' or 'entails'"
        else:
            raise ValueError(f"Invalid type: {self.dataset_type}. Choose one of 'mnli', 'qnli' or 'scitail'.")
        
        prompt = (f"Given the definitions for each label:\n"
                  f"Entailment: The hypothesis must be true if the premise is true.\n"
                  f"Neutral: The truth of the hypothesis cannot be determined from the premise.\n" 
                  f"Does the {sentence1} entail the {sentence2}? Answer exactly one word in lowercase: {labels}.\n"
                  f"{sentence1}: {item[sentence1.lower()]} \n{sentence2}: {item[sentence2.lower()]} \nAnswer:")
        
        # Tokenise prompt
        if self.chat_template:
            messages = [
            {"role": "user", "content": prompt}
            ]

            encoding = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                padding="max_length",
                max_length=self.prompt_max_length,
                return_tensors="pt"
            )
        else:
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
        max_length=self.label_max_length,
        add_special_tokens=False,
        return_tensors="pt"
        )
        
        return {"input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels_ids": gold_label_ids["input_ids"].squeeze(),
            "labels": gold_label,
            "prompt": prompt}
    

class MyDataset_few_shot(Dataset):

    def __init__(self, dataframe, examples, tokenizer, dataset_type, prompt_max_length, label_max_length, chat_template, training=False):
        self.dataframe = dataframe
        self.examples = examples
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type.split('_')[0]
        self.prompt_max_length = prompt_max_length
        self.label_max_length = label_max_length
        self.training = training
        self.chat_template = chat_template
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        gold_label = item['label']

        # Format the prompt depending on dataset_type
        if self.dataset_type == "mnli":
            sentence1 = "premise"
            sentence2 = "hypothesis"
            labels = "'contradiction', 'neutral' or 'entailment'"
        elif self.dataset_type == "qnli":
            sentence1 = "sentence"
            sentence2 = "question"
            labels = "'not entailment' or 'entailment'"
        elif self.dataset_type == "scitail":
            sentence1 = "premise"
            sentence2 = "hypothesis"
            labels = "'neutral' or 'entails'"
        else:
            raise ValueError(f"Invalid type: {self.dataset_type}. Choose one of 'mnli', 'qnli' or 'scitail'.")
        
        prompt = (f"Examples:\n{self.examples} \nGiven the above examples as reference does the {sentence1} entail the {sentence2} in the following case? "
                  f"Answer exactly one word in lowercase as in the examples: {labels}. \n{sentence1}: {item[sentence1.lower()]} \n{sentence2}: {item[sentence2.lower()]} \nAnswer:")
        
        # Tokenise prompt
        if self.chat_template:
            messages = [
            {"role": "user", "content": prompt}
            ]

            encoding = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                padding="max_length",
                max_length=self.prompt_max_length,
                return_tensors="pt"
            )
        else:
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
        max_length=self.label_max_length,
        add_special_tokens=False,
        return_tensors="pt"
        )
        
        return {"input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels_ids": gold_label_ids["input_ids"].squeeze(),
            "labels": gold_label,
            "prompt": prompt}
    

class MyDataset_MP(Dataset):

    def __init__(self, dataframe, tokenizer, dataset_type, prompt_max_length, label_max_length, chat_template, training=False):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type.split('_')[0]
        self.prompt_max_length = prompt_max_length
        self.label_max_length = label_max_length
        self.training = training
        self.chat_template = chat_template
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        gold_label = item['label']

        # Format the prompt depending on dataset_type
        if self.dataset_type == "mnli":
            sentence1 = "premise"
            sentence2 = "hypothesis"
            labels = "'contradiction', 'neutral' or 'entailment'"
        elif self.dataset_type == "qnli":
            sentence1 = "sentence"
            sentence2 = "question"
            labels = "'not entailment' or 'entailment'"
        elif self.dataset_type == "scitail":
            sentence1 = "premise"
            sentence2 = "hypothesis"
            labels = "'neutral' or 'entails'"
        else:
            raise ValueError(f"Invalid type: {self.dataset_type}. Choose one of 'mnli', 'qnli' or 'scitail'.")
        
        prompt = (f'For the question: '

                f'{item[sentence2.lower()]}\nand statement: {item[sentence1.lower()]}, determine if the statement provides the answer to the question.\n'

                f'If the statement contains the answer to the question, the status is entailment.\n'
                f'If it does not, the status is not entailment.\n'

                f'As you perform this task, follow these steps:\n'

                f'1. Clarify your understanding of the question and the context sentence.\n'

                f'2. Make a preliminary identification of whether the context sentence contains the answer to the question.\n'

                f'3. Critically assess your preliminary analysis. If you feel unsure about your initial entailment classification, try to reassess it.\n'

                f'4. Confirm your final answer and explain the reasoning behind your choice.\n'

                f'5. Evaluate your confidence (0-100%) in your analysis and provide an explanation for this confidence level.\n'

                f'Provide the answer in your final response as "The status is [entailment / not entailment".')
        
        # Tokenise prompt
        if self.chat_template:
            messages = [
            {"role": "user", "content": prompt}
            ]

            encoding = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                padding="max_length",
                max_length=self.prompt_max_length,
                return_tensors="pt"
            )
        else:
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
        max_length=self.label_max_length,
        add_special_tokens=False,
        return_tensors="pt"
        )
        
        return {"input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels_ids": gold_label_ids["input_ids"].squeeze(),
            "labels": gold_label,
            "prompt": prompt}


class MyDataset_CoT(Dataset):

    def __init__(self, dataframe, tokenizer, dataset_type, prompt_max_length, chat_template, label_max_length, training=False):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type.split('_')[0]
        self.prompt_max_length = prompt_max_length
        self.label_max_length = label_max_length
        self.training = training
        self.chat_template = chat_template
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        gold_label = item['label']

        # Format the prompt depending on dataset_type
        if self.dataset_type == "mnli":
            sentence1 = "premise"
            sentence2 = "hypothesis"
            labels = "'contradiction', 'neutral' or 'entailment'"
        elif self.dataset_type == "qnli":
            sentence1 = "Sentence"
            sentence2 = "Question"
            labels = "'not entailment' or 'entailment'"
        elif self.dataset_type == "scitail":
            sentence1 = "premise"
            sentence2 = "hypothesis"
            labels = "'neutral' or 'entails'"
        else:
            raise ValueError(f"Invalid type: {self.dataset_type}. Choose one of 'mnli', 'qnli' or 'scitail'.")
        
        prompt = (f"Does the {sentence1} entail the {sentence2}? "
                  f"Provide an answer in one word and then explain the steps you took to reach that conclusion: {labels}. \n{sentence1}: {item[sentence1.lower()]} \n{sentence2}: {item[sentence2.lower()]} \nAnswer: Let's think step by step.")
        
        # Tokenise prompt
        if self.chat_template:
            messages = [
            {"role": "user", "content": prompt}
            ]

            encoding = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                padding="max_length",
                max_length=self.prompt_max_length,
                return_tensors="pt"
            )
        else:
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
        max_length=self.label_max_length,
        add_special_tokens=False,
        return_tensors="pt"
        )
        
        return {"input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels_ids": gold_label_ids["input_ids"].squeeze(),
            "labels": gold_label,
            "prompt": prompt}
    
# def predict_fn(texts, model, tokenizer, dataset_type):
#     """
#     Predict using prompt
#     """
#     # Ensure texts are str
#     if isinstance(texts, numpy.ndarray):
#         texts = texts.tolist()

#     inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
#     probs = get_model_probs(inputs['input_ids'], inputs['attention_mask'], model, tokenizer, dataset_type)
#     probs = F.normalize(probs, p=1, dim=1)

#     return probs.cpu().numpy()

# def predict_fn_pretokenized(input_ids, attention_mask, model, tokenizer, dataset_type):
#     """
#     Predict using pre-tokenized input_ids and attention_mask.
#     """
#     probs = get_model_probs(input_ids, attention_mask, model, tokenizer, dataset_type)
#     probs = F.normalize(probs, p=1, dim=1)

#     return probs.cpu().numpy()
    

# def forward_pass_fn(input_embeds, attention_mask, model, pred_label_id, tokenizer, class_names):
#     """
#     This function gives the logit of the next token only, which is an approximation for the whole sequence
#     """
#     # Get outputs using embeds
#     outputs = model(inputs_embeds=input_embeds.requires_grad_(), attention_mask=attention_mask)

#     # Get logits for the last token
#     last_token_logits = outputs.logits[:, -1, :]

#     # Get the predicted label and get the id
#     pred_label = class_names[pred_label_id]
#     pred_label_first_token_id = tokenizer.encode(pred_label, add_special_tokens=False)[0]

#     return last_token_logits[:, pred_label_first_token_id]

# def add_text_to_visualizer(attributions, pred_prob, pred_label, true_label, delta, tokens, data_records):
#     # Convert attributions to a list
#     attributions = attributions.sum(dim=2)[0].detach().cpu().numpy()
#     attributions = attributions / (numpy.max(numpy.abs(attributions)) + 1e-10)

#     # Create a VisualizationDataRecord
#     data_records.append(viz.VisualizationDataRecord(
#         word_attributions=attributions,
#         pred_prob=pred_prob,
#         pred_class=pred_label,
#         true_class=true_label,
#         attr_class=pred_label,
#         convergence_score=delta,
#         attr_score=attributions.sum(),
#         raw_input_ids=tokens
#     ))

# def interpret_example_IG(model, tokenizer, example_id, ig, data_records, class_names, pred_label_id, pred_prob, dataset):
#     input_ids = dataset[example_id]['input_ids'].unsqueeze(0).to(model.device)
#     attention_mask = dataset[example_id]['attention_mask'].unsqueeze(0).to(model.device)

#     interpretable_embeddings = InterpretableEmbeddingBase(model.get_input_embeddings(), "embeds")

#     # Get embeddings from the model's embedding layer using input_ids
#     input_embeddings = interpretable_embeddings.indices_to_embeddings(input_ids)

#     # Define a baseline to compare
#     baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id).to(model.device)
#     baseline_embeddings = interpretable_embeddings.indices_to_embeddings(baseline_ids)
    
#     # Call IG on embeddings
#     attributions, delta = ig.attribute(
#         inputs=input_embeddings,
#         baselines=baseline_embeddings,
#         additional_forward_args=(attention_mask,),
#         n_steps=50,
#         internal_batch_size=1,
#         return_convergence_delta=True
#     )

#     # Get true label, predicted label, probability and tokens as text
#     true_label = dataset[example_id]['labels']
#     pred_label = class_names[pred_label_id]
#     tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

#     # Add example to be visualised
#     add_text_to_visualizer(attributions, pred_prob, pred_label, true_label, delta, tokens, data_records)