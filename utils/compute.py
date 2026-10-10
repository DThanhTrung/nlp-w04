import numpy as np

def _prob_bigram(
    cnt_matrix: np.ndarray,
    prev_word_id: int,
    cur_word_id: int,
    V: int
) -> float :
    return (
        (cnt_matrix[prev_word_id, cur_word_id] + 1) 
        /
        (np.sum(cnt_matrix[prev_word_id, :]) + V)
    )


def compute_nll(
    sentence: list[list[str]],
    cnt_matrix: np.ndarray,
    word_to_id: dict[str, int],
    V: int
) -> float :
    N = len(sentence) - 2

    sum_log_prob = sum(
        np.log(
            _prob_bigram(
                cnt_matrix, 
                word_to_id[sentence[i]],
                word_to_id[sentence[i + 1]],
                V
            )
        ) for i in range (len(sentence) - 1)
    )

    return -1 / N * sum_log_prob

def _compute_pmi(
    target_word: str,
    context_word: str,
    co_occurrence_matrix: np.ndarray,
    word_to_id: dict[str, int]
) -> float :
    target_id, context_id = word_to_id[target_word], word_to_id[context_word]

    N = np.sum(co_occurrence_matrix)

    return (
        (co_occurrence_matrix[target_id, context_id] * N)
        /
        (np.sum(co_occurrence_matrix[target_id, :]) * np.sum(co_occurrence_matrix[context_id, :]))

        + 1e-10
    )

def _compute_pairwise_ppmi(
    target_word: str,
    context_word: str,
    co_occurrence_matrix: np.ndarray,
    word_to_id: dict[str, int]
) -> float :
    pmi = _compute_pmi(
        target_word,
        context_word,
        co_occurrence_matrix,
        word_to_id
    )

    return np.log2(np.maximum(pmi, 1))

def compute_ppmi(
    target_word: str,
    vocabulary: list[str],
    co_occurrence_matrix: np.ndarray,
    word_to_id: dict[str, int]
) -> np.ndarray :    
    return np.array([
        _compute_pairwise_ppmi(target_word, context_word, co_occurrence_matrix, word_to_id)
        for context_word in vocabulary
    ])