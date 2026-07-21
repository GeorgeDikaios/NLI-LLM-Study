import utils
import pandas as pd

model_ids = [
    'google/gemma-2-9b-it',
    'meta-llama/Llama-3.1-8B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3'
]

datasets = [
    'mnli_m',
    'qnli',
    'scitail'
]

shot_names = ['1shot_entails', '1shot_neutral', '2shot', '3shot', '5shot', '8shot', '10shot', '15shot']
seed_idxs = [i for i in range(10)]

rows = []

for model_id in model_ids:
    for dataset in datasets:
        label_order = utils.get_labels(dataset)
        for shot_name in shot_names:
            for seed_idx in seed_idxs:
                checkpoint_params = {
                            'model_id': model_id,
                            'dataset_type': dataset,
                            'training_mode': shot_name,
                            "seed_idx": seed_idx
                        }
                checkpoint_path = utils.create_checkpoint_path(checkpoint_params)
                preds, gold_labels, probs, _ = utils.load_checkpoint(checkpoint_path)

                shot_count, condition = utils.parse_shot_name(shot_name)

                for item_idx, (gold, pred, prob_row) in enumerate(zip(gold_labels, preds, probs)):
                    row = {
                        'model': model_id,
                        'dataset': dataset,
                        'shot_count': shot_count,
                        'condition': condition,
                        'seed_idx': seed_idx,
                        'item_idx': item_idx,
                        'gold_label': gold,
                        'prediction': pred,
                        'correct': int(gold == pred),
                    }

                    for i, label_name in enumerate(label_order):
                        row[f'prob_{label_name}'] = prob_row[i]
                    rows.append(row)

for model_id in model_ids:
    for dataset in datasets:
        label_order = utils.get_labels(dataset)
        checkpoint_params = {
            'model_id': model_id,
            'dataset_type': dataset,
            'training_mode': 'zero_shot',
            'seed_idx': None
        }
        checkpoint_path = utils.create_checkpoint_path(checkpoint_params)
        preds, gold_labels, probs, _ = utils.load_checkpoint(checkpoint_path)

        for item_idx, (gold, pred, prob_row) in enumerate(zip(gold_labels, preds, probs)):
                    row = {
                        'model': model_id,
                        'dataset': dataset,
                        'shot_count': 0,
                        'condition': None,
                        'seed_idx': None,
                        'item_idx': item_idx,
                        'gold_label': gold,
                        'prediction': pred,
                        'correct': int(gold == pred),
                    }

                    for i, label_name in enumerate(label_order):
                        row[f'prob_{label_name}'] = prob_row[i]
                    rows.append(row)

df = pd.DataFrame(rows)
df.to_parquet(f'Results_combined.parquet', index=False)
