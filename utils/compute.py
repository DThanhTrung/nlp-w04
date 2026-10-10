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