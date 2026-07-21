import os
import utils
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from tqdm import tqdm

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
utils.hf_login("HF_TOKEN")

#### DEFINE PARAMS #####
BATCH_SIZE = 32
# Quantization config
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

###### DEFINE MODELS #######
model_ids = [
    'google/gemma-2-9b-it',
    'meta-llama/Llama-3.1-8B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3'
]

##### LOAD DATASETS #####
qnli_val = pd.read_csv('qnli_val_clean.csv')
qnli_val["label"] = qnli_val["label"].map({0: "entailment", 1: "not entailment"})

mnli_m_val = pd.read_csv('mnli_m_val_clean.csv')
mnli_m_val["label"] = mnli_m_val["label"].map({0: "entailment", 1: "neutral", 2: "contradiction"})

scitail_test = pd.read_csv('scitail_test_clean.csv')

#### COMBINE DATASETS AND EXAMPLE SETS ####
datasets = [
    ['qnli', qnli_val],
    ['mnli_m', mnli_m_val],
    ['scitail', scitail_test]
]

# LOOP OVER EACH MODEL FOR EVALUATION
for model_id in model_ids:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation="eager"
        )

    # Add padding token to the tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Change the model to evaluation mode
    model.eval()
    model.config.use_cache = True

    # LOOP OVER DATASETS - EXAMPLE SETS
    for dataset_name, dataset in datasets:
                
        # Create Checkpoint Path
        checkpoint_params = {
            'model_id': model_id,
            'dataset_type': dataset_name,
            'training_mode': 'zero_shot',
            'seed_idx': None
        }
        checkpoint_path = utils.create_checkpoint_path(params=checkpoint_params)

        # Find the max_length for tokenization to avoid wasting computing.
        safe_max_length = utils.find_max_length(dataset, tokenizer=tokenizer, dataset_type=dataset_name, chat_template=True, examples=None, kind='zero_shot')

        # Define dataset and create a dataloader.
        dataset_test = utils.MyDataset(dataframe=dataset,
                                                examples=None,
                                                tokenizer=tokenizer,
                                                dataset_type=dataset_name,
                                                prompt_max_length=safe_max_length,
                                                label_max_length=3,
                                                chat_template=True)

        dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

        # Load checkpoint if it exists
        predicted_labels, gold_labels, all_probs, start_batch = utils.load_checkpoint(checkpoint_path=checkpoint_path)
        labels = utils.get_labels(dataset_name)

        # Loop over the batches
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):

                # Continue from last checkpoint
                if i < start_batch:
                    continue

                input_ids_batch = batch["input_ids"].to(model.device) # Move to GPU
                attention_mask_batch = batch["attention_mask"].to(model.device) # Move to GPU
                gold_labels_batch = batch["labels"] # Keep to CPU

                batch_probs = utils.get_model_probs(batch_input_ids=input_ids_batch,
                                            batch_attention_mask=attention_mask_batch,
                                            dataset_type=dataset_name,
                                            model=model,
                                            tokenizer=tokenizer)

                batch_pred_indices = torch.argmax(batch_probs, dim=1)
                batch_pred_labels = [labels[i] for i in batch_pred_indices]

                predicted_labels.extend(batch_pred_labels)
                gold_labels.extend(gold_labels_batch)
                all_probs.extend(batch_probs.cpu().tolist())

                # Save checkpoint
                if i % 50 == 0 or i == len(dataloader) - 1:
                    torch.save({"predicted_labels": predicted_labels,
                                "gold_labels": gold_labels,
                                'all_probs': all_probs,
                                "batch_no": i+1}, checkpoint_path)

                    print(f"Checkpoint saved: {i+1}, {checkpoint_path}")
