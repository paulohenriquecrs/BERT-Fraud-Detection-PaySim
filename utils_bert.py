"""Utility functions for BERT-based fraud detection."""

from dataclasses import dataclass
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

#####################################################################################################################
# MLM pretraining
#####################################################################################################################


class FraudMLMDataset(Dataset):
    """Dataset for Masked Language Modeling (MLM) pretraining of BERT on fraud detection data."""

    def __init__(
        self, dataframe, values2tokens: Dict[str, int], special_tokens: Dict[str, int]
    ):
        self.data = dataframe.reset_index(drop=True)

        self.cls_id = special_tokens["[CLS]"]
        self.sep_id = special_tokens["[SEP]"]
        self.values2tokens = values2tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw_tokens = self.data.iloc[idx].values.tolist()

        tokens_ids = [self.values2tokens[token] for token in raw_tokens]

        input_ids = [self.cls_id] + tokens_ids + [self.sep_id]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


@dataclass
class FraudDataCollatorForMLM:
    """Data collator for Masked Language Modeling (MLM) pretraining of BERT on fraud detection data."""

    mask_token_id: int
    cls_token_id: int
    sep_token_id: int
    vocab_size: int
    mlm_probability: float = 0.15
    num_special_tokens: int = 3  # [CLS], [SEP], [MASK]

    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([e["input_ids"] for e in examples])
        attention_mask = torch.stack([e["attention_mask"] for e in examples])

        labels = input_ids.clone()

        # Mask candidates: not CLS, SEP
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_mask = (input_ids == self.cls_token_id) | (
            input_ids == self.sep_token_id
        )

        probability_matrix.masked_fill_(special_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Labels only for masked positions
        labels[~masked_indices] = -100

        # 80% -> [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.mask_token_id

        # 10% -> random token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )

        # Random value tokens only, avoid special tokens
        random_words = torch.randint(
            low=self.num_special_tokens,
            high=self.vocab_size,
            size=labels.shape,
            dtype=torch.long,
        )
        input_ids[indices_random] = random_words[indices_random]

        # remaining 10% stay unchanged

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


#####################################################################################################################
# Fine tuning for classification
#####################################################################################################################


class FraudClassificationDataset(Dataset):
    """Dataset for fine-tuning BERT on fraud detection classification task."""

    def __init__(
        self,
        dataframe,
        values2tokens: Dict[str, int],
        special_tokens: Dict[str, int],
        labels: List[int],
    ):
        self.data = dataframe.reset_index(drop=True)
        self.labels = torch.tensor(labels, dtype=torch.long)

        self.cls_id = special_tokens["[CLS]"]
        self.sep_id = special_tokens["[SEP]"]
        self.values2tokens = values2tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw_tokens = self.data.iloc[idx].values.tolist()

        # Direct dictionary lookup (no fallback needed)
        tokens_ids = [self.values2tokens[token] for token in raw_tokens]

        input_ids = [self.cls_id] + tokens_ids + [self.sep_id]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": self.labels[idx],
        }


def compute_metrics(eval_pred: tuple) -> Dict[str, float]:
    """Compute evaluation metrics for binary classification Trainer."""

    logits, labels = eval_pred

    # Probability of fraud
    probabilities = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    # Default threshold
    predictions = (probabilities >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    pr_auc = average_precision_score(labels, probabilities)
    roc_auc = roc_auc_score(labels, probabilities)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }
