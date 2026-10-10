# NLP Sprint 04 — From Counts to Embeddings

## Overview

You will implement three ideas covered in class:

1. Add-one bigram language modeling
2. Distributional semantics with PPMI and cosine similarity
3. Dense word-vector analogies

The competition contains 60 questions, with 20 questions for each task.

Each question has four answer options labeled 0, 1, 2, and 3.

Your goal is to implement the required algorithms, select the correct option for each question, and submit your predictions as a CSV file.

## Task 1 — Add-one Bigram Language Model

Train your language model using lm_corpus.txt.

Use the following conventions:

- lowercase all text
- tokenize using the regex `[a-z]+`
- add `<s>` and `</s>` as sentence boundaries
- the prediction vocabulary consists of all lexical word types plus `</s>`
- `<s>` is used only as a context token and is not included in vocabulary size `V`

Use add-one smoothing:

```text
P(w_i | w_{i-1})
        = (C(w_{i-1}, w_i) + 1)
            / (C(w_{i-1}) + V)
```

For each candidate sentence, compute its average negative log-likelihood:

```text
NLL(W)
    = -(1 / N) * sum_i log P(w_i | w_{i-1})
```

Include the prediction of </s> when computing the score.

Choose the candidate with the lowest average NLL.

## Task 2 — PPMI Semantic Similarity

Use semantic_corpus.txt.

Build a word-word term-context matrix using the following conventions:

- lowercase all text
- tokenize using the regex `[a-z]+`
- use a context window of ±2
- do not cross sentence boundaries
- do not count the target word itself as its own context
- every ordered target-context occurrence contributes +1

For a target word w and context word c, compute:

```text
PMI(w, c)
    = log2(P(w, c) / (P(w) * P(c)))

PPMI(w, c)
    = max(PMI(w, c), 0)
```

Represent each word using its PPMI row vector.

For each question:

    obtain the PPMI vector of the target word
    obtain the PPMI vector of each candidate
    compute cosine similarity
    choose the candidate with the largest cosine similarity

Use:

```text
cosine(u, v)
    = dot(u, v) / (norm(u) * norm(v))
```

If either vector has norm 0, define its cosine similarity as 0.

## Task 3 — Dense Embedding Analogies

Use the dense vectors provided in embeddings.csv.

Each analogy has the form:

```text
a : b :: c : ?
```

Construct the query vector:

```text
q = v_b - v_a + v_c
```

Then compute the cosine similarity between q and each candidate vector.

Choose the candidate with the largest cosine similarity.

### Example

For the analogy:

man : woman :: king : ?

construct:

```text
q = v_woman - v_man + v_king
```

Then compare q with the candidate vectors using cosine similarity.

## Evaluation

Submissions are evaluated using Accuracy:

```text
Accuracy
        = number of correct predictions
            / total number of questions
```

The competition contains:
| | |
| --- | ---: |
| Bigram Language Model | 20 |
| PPMI + Cosine Similarity | 20 |
| Embedding Analogies | 20 |
| Total | 60 |

Because every question has four answer options, random guessing has an expected accuracy of approximately 25%.

## Submission Format

Your submission must contain exactly two columns:

```csv
id,label
Q001,2
Q002,0
Q003,3
```

The label column must contain one of:

```text
0, 1, 2, 3
```

Use sample_submission.csv as the submission template.

Before submitting, make sure that:

- every question ID appears exactly once
- there are no missing predictions
- all labels are integers in `{0, 1, 2, 3}`
- the column names remain exactly `id` and `label`

## Rules

- This is an individual exercise unless the instructor states otherwise.
- Do not use external datasets.
- Do not use pretrained embeddings.
- Do not use `gensim` or a nearest-neighbor package to bypass the requested implementations.
- Python standard library, NumPy, and pandas are allowed.
- Do not hard-code answers.
- Do not share answers or submission files during the competition.

The intended solution can be completed entirely in the provided starter notebook.

Good luck!
