# BERT Representation Learning for Fraud Detection

A small scale experiment exploring how BERT style Transformer encoders can be used for **representation learning on structured, non textual data**.

The goal of this project was not to build the best possible fraud detection model. In fact, using a Transformer for this specific problem is clearly more machinery than necessary: the dataset has a very small vocabulary, every transaction contains the same seven attributes, and the resulting input sequence is extremely short.

The purpose was instead to get hands on experience with the **encoder and representation learning side of Transformers**, by building a BERT style model from scratch, pretraining it with Masked Language Modeling, and then transferring the learned representations to a supervised fraud detection task.

This project is intended as a starting point for exploring similar ideas in richer problems such as recommendation systems, user preference modeling, entity representations, and event prediction.

## Motivation

Transformers are commonly associated with language models, but the underlying architecture can also be applied to structured data.

A transaction can be represented as a short sequence of discrete tokens corresponding to its attributes:

```text
[CLS]
TRANSFER
AMOUNT_BIN_6
ORIGIN_BALANCE_BIN_8
ORIGIN_NEW_BALANCE_BIN_7
DESTINATION_BALANCE_BIN_4
DESTINATION_NEW_BALANCE_BIN_5
SYNTH_HOUR_14
[SEP]
```

The Transformer first learns from these transactions using a self supervised Masked Language Modeling objective.

For example:

```text
[CLS]
TRANSFER
AMOUNT_BIN_6
[MASK]
ORIGIN_NEW_BALANCE_BIN_7
SYNTH_HOUR_14
[SEP]
```

The model learns to reconstruct the masked value from the surrounding transaction context.

The fraud labels are intentionally excluded from this stage.

After pretraining, the learned encoder is transferred to a downstream fraud classification task.

```text
Structured transaction
        ↓
Discretization and tokenization
        ↓
Masked Language Modeling
        ↓
Pretrained BERT encoder
        ↓
Fraud classification
```

The main question is therefore:

> Can a small Transformer learn useful representations of structured transaction data through self supervised pretraining, and can those representations transfer to a downstream task?

## Dataset

The project uses the **PaySim** synthetic mobile money transaction dataset.

The model uses seven transaction attributes:

| Feature                                | Representation       |
| -------------------------------------- | -------------------- |
| Transaction type                       | Categorical token    |
| Transaction amount                     | Monetary value bin   |
| Origin balance before transaction      | Monetary value bin   |
| Origin balance after transaction       | Monetary value bin   |
| Destination balance before transaction | Monetary value bin   |
| Destination balance after transaction  | Monetary value bin   |
| Transaction time                       | Synthetic hour token |

The original sender and receiver account identifiers are removed.

The fraud label is used only for supervised classification and is never provided during MLM pretraining.

## Data preprocessing

### Monetary values

All monetary features are discretized into ten broad intervals:

```text
0
0 to 10K
10K to 50K
50K to 100K
100K to 250K
250K to 500K
500K to 750K
750K to 1M
1M to 3M
above 3M
```

The same monetary vocabulary is shared across:

```text
amount
oldbalanceOrg
newbalanceOrig
oldbalanceDest
newbalanceDest
```

For example, the same value bin represents the same monetary range regardless of which monetary field it comes from.

This discretization serves two purposes.

First, it provides a simple way to transform numerical attributes into categorical tokens suitable for a BERT style vocabulary.

Second, the deliberately broad bins remove exact monetary values. Together with the removal of account IDs, this makes it harder to indirectly identify an account from highly specific balance values appearing across different transactions.

This comes at the cost of losing some numerical precision, which is an intentional tradeoff for this experiment.

### Transaction time

PaySim represents time using simulation steps, where each step corresponds to one hour.

The absolute simulation step is replaced with the hour of the simulated day:

```python
hour = step % 24
```

This produces 24 possible hour tokens.

Using the hour preserves coarse temporal information without making every simulation step a separate token.

### Account identifiers

The sender and receiver account identifiers are removed.

The dataset contains a very large number of account IDs, which would create a large vocabulary with limited value for this particular experiment.

The focus is instead on learning representations from transaction attributes and their relationships.

## Data split

The dataset is divided into training, validation, and test sets using stratified random splitting based on the fraud label.

The resulting proportions are approximately:

```text
Training      75%
Validation    10%
Test          15%
```

Fraud is extremely rare, so stratification is used to preserve a similar fraud ratio across the splits.

A random transaction level split is used for this exploratory experiment rather than a strict temporal or account level split.

Because account identifiers are removed and monetary values are heavily discretized, the model also has less opportunity to indirectly recognize individual accounts across the split.

The final test set contains 954,393 transactions, including 1,232 fraudulent transactions.

## Transaction vocabulary

The final vocabulary contains only 42 tokens:

| Token group         |  Count |
| ------------------- | -----: |
| Special tokens      |      3 |
| Transaction types   |      5 |
| Monetary value bins |     10 |
| Synthetic hours     |     24 |
| **Total**           | **42** |

Each transaction is converted into a fixed length sequence:

```text
[CLS]
TYPE
AMOUNT
OLD_ORIGIN_BALANCE
NEW_ORIGIN_BALANCE
OLD_DESTINATION_BALANCE
NEW_DESTINATION_BALANCE
HOUR
[SEP]
```

Because every transaction contains the same fields in the same order, all sequences have the same length and no padding is required.

The fixed ordering also means that the BERT positional embeddings implicitly provide information about which field a token represents.

## Baseline

A `HistGradientBoostingClassifier` is used as a traditional machine learning baseline.

Categorical features are ordinally encoded and balanced class weights are used to account for the strong class imbalance.

The classification threshold is selected on the validation set by maximizing F1, then applied unchanged to the test set.

The baseline achieves a strong PR AUC, but its selected operating point produces substantially lower precision than the Transformer models.

## BERT architecture

The Transformer is initialized from scratch rather than using pretrained language model weights.

The architecture is intentionally small:

| Parameter               | Value |
| ----------------------- | ----: |
| Vocabulary size         |    42 |
| Hidden dimension        |   128 |
| Feed forward dimension  |   512 |
| Transformer layers      |     4 |
| Attention heads         |     4 |
| Maximum sequence length |     9 |
| Token type vocabulary   |     1 |

This is deliberately a small BERT implementation designed for experimentation rather than scale.

## Masked Language Modeling

The first training stage is completely self supervised.

Fraud labels are ignored and the model learns only from the transaction attributes.

15% of eligible tokens are selected for MLM using the standard BERT style corruption strategy:

```text
80% → [MASK]
10% → random token
10% → unchanged
```

The objective is to reconstruct the original token at the selected positions.

The MLM stage is implemented using custom PyTorch components:

* A transaction dataset that converts structured attributes into token IDs
* A custom MLM data collator
* Hugging Face `BertForMaskedLM`
* Hugging Face `Trainer`

After three epochs, the MLM validation loss decreased from 1.6985 to 1.6657.

| Epoch | Training loss | Validation loss |
| ----: | ------------: | --------------: |
|     1 |        1.7199 |          1.6985 |
|     2 |        1.6644 |          1.6757 |
|     3 |        1.6591 |          1.6657 |

This indicates that the encoder learned predictable relationships between the transaction attributes before being exposed to fraud labels.

## Downstream fraud classification

The pretrained BERT encoder is then transferred to a binary fraud classification task.

The MLM prediction head is discarded and replaced by a newly initialized classification head with two output classes:

```text
Normal
Fraud
```

Two transfer settings are evaluated.

### Full fine tuning

The BERT encoder and the classification head are jointly optimized using the fraud labels.

### Frozen pretrained encoder

The BERT encoder is frozen after MLM pretraining, and only the classification head is trained.

This second experiment provides a direct test of how useful the representations learned during MLM are without adapting the Transformer to the downstream fraud task.

## Results

The main results are summarized below.

| Model                  |     PR AUC |         F1 |  Precision |     Recall |
| ---------------------- | ---------: | ---------: | ---------: | ---------: |
| HistGradientBoosting   |     0.8625 |     0.5799 |     0.4359 | **0.8661** |
| Frozen pretrained BERT |     0.6834 |     0.7315 |     0.8631 |     0.6347 |
| Full BERT fine tuning  | **0.8737** | **0.8674** | **0.9806** |     0.7776 |

The fully fine tuned BERT model achieves the strongest overall results, with a PR AUC of 0.8737 and an F1 score of 0.8674.

The frozen BERT experiment is particularly interesting from a representation learning perspective. With the Transformer completely frozen, the classification head still reaches an F1 score of 0.7315. The encoder therefore learned information during MLM that can transfer to fraud classification without updating the backbone.

At the same time, full fine tuning improves both PR AUC and F1 substantially compared with the frozen representation. This shows that the MLM representation is useful, but supervised adaptation still provides significant task specific value.

The HistGradientBoosting baseline achieves a similar PR AUC to the fully fine tuned BERT model, 0.8625 versus 0.8737. Its selected threshold gives higher recall, but at the cost of much lower precision, resulting in an F1 score of 0.5799.

This illustrates why the comparison cannot be reduced to a single metric. PR AUC evaluates ranking quality across thresholds, while F1 reflects the particular precision and recall tradeoff obtained at the selected threshold.

The results therefore do **not** suggest that BERT is inherently better than gradient boosted trees for this tabular fraud problem. The main finding is that a small Transformer can learn useful structured transaction representations through MLM and then transfer those representations to a downstream classification task.

## Why use BERT here?

For this dataset alone, there are much simpler approaches.

The vocabulary contains only 42 tokens. Each transaction has seven fixed attributes, producing a sequence of only nine positions once `[CLS]` and `[SEP]` are added.

A tree based model is therefore a much more natural choice if the sole objective is efficient fraud prediction.

Using BERT is intentional.

This project was created as a controlled introduction to adapting encoder based Transformers to structured entities rather than text. The value of the experiment is in understanding the **representation learning workflow**, not in arguing that a Transformer is the most efficient architecture for PaySim.

## What I learned

The project provided hands on experience with the full self supervised representation learning pipeline:

```text
Structured data
      ↓
Discretization
      ↓
Entity vocabulary
      ↓
Custom PyTorch dataset
      ↓
Custom MLM masking
      ↓
BERT pretraining
      ↓
Learned representations
      ↓
Downstream classification
      ↓
Evaluation
```

In particular, it provided practical experience with:

* Building a BERT style architecture from scratch
* Creating a vocabulary for structured data
* Converting tabular attributes into token sequences
* Implementing a custom MLM data collator
* Training with Hugging Face `Trainer`
* Transferring an MLM checkpoint to a classification architecture
* Comparing frozen and fully fine tuned representations
* Working with highly imbalanced classification
* Selecting thresholds using validation data
* Comparing Transformer representations with a strong tabular baseline

The project also reinforced an important distinction between **self supervised representation learning** and **task specific fine tuning**.

## Limitations

This experiment is intentionally small and has several limitations.

**Synthetic dataset**

PaySim is synthetic, so the learned patterns should not be interpreted as representative of real financial fraud.

**Limited numerical representation**

The monetary features are heavily discretized into broad bins, so exact numerical information is lost.

**No account identifiers**

Removing account identifiers prevents account specific memorization, but it also removes potentially useful behavioural information associated with persistent accounts.

**Random transaction split**

The evaluation uses a random transaction level split rather than a strict temporal or account level split. A real deployment scenario would require a more realistic evaluation protocol.

**Very short sequences**

The Transformer receives only seven transaction attributes. There is therefore little reason for self attention to be necessary for this particular task.

**Limited hyperparameter search**

The configurations were chosen to validate the idea and understand the workflow. The project did not perform an extensive search over learning rates, architectures, batch sizes, masking strategies, or training duration.

## Future directions

This project was intended as a **kickoff into Transformer based representation learning**, rather than an endpoint for fraud detection.

The next direction is to explore problems where entity representations and contextual interactions are more central to the task.

One particularly interesting direction is **recommendation and user preference modeling**.

For example, a user could be represented through a sequence of movie interactions:

```text
[CLS]
MOVIE_A RATING_5
MOVIE_B RATING_3
MOVIE_C RATING_5
MOVIE_D RATING_2
...
[SEP]
```

A BERT style encoder could then learn representations of movies and user preferences through self supervised pretraining.

Those representations could later be used for tasks such as:

* Predicting whether a user will like a movie
* Ranking movies for a user
* Learning useful movie embeddings
* Predicting future interactions
* Transferring pretrained representations to smaller downstream tasks

This would provide a richer setting for the same ideas explored here, with a larger vocabulary, longer sequences, repeated entities, and more meaningful contextual relationships.

I am also interested in exploring other **encoder only Transformer architectures** and alternative self supervised objectives for structured data.

The broader goal is to understand how Transformer based representation learning can be used to learn transferable embeddings for entities, preferences, and interactions that are not naturally expressed as text.

## Repository structure

```text
.
│
└── fraud_detection_bert.ipynb
└── utils_bert.py
└── utils_metrics.py
└── README.md
```

`utils_bert.py` contains the custom datasets, MLM data collator, and training metrics.

`utils_metrics.py` contains threshold selection and final evaluation utilities, including PR AUC, ROC AUC, F1, precision, recall, and confusion matrix calculations.

The evaluation utility selects the classification threshold using validation data and then applies that threshold to the test set, keeping threshold selection separate from final test evaluation.

## Final takeaway

This project was deliberately simple.

The objective was to move beyond using Transformers primarily for text generation and gain practical experience with the **encoder and representation learning side of BERT**.

A small Transformer was trained from scratch on structured transaction data using Masked Language Modeling, then transferred to fraud classification.

The most useful outcome is not that BERT replaced a traditional tabular model. Instead, the experiment demonstrated a complete workflow for:

```text
self supervised pretraining
        ↓
learned representations
        ↓
transfer learning
        ↓
downstream task
```

This provides a starting point for exploring more interesting applications of Transformer based representation learning, particularly recommendation systems, user preference modeling, and entity based prediction.
