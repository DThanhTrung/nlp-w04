import numpy as np

def build_bigram_matrix(
    sentences: list[list[int]]
) -> tuple[np.ndarray, list[str], dict[str, int]] :
    vocabulary = sorted(
        word 
        for sentence in sentences
        for word in sentence
    )

    word_to_id = {
        word: index
        for (index, word) in enumerate(vocabulary)
    }

    cnt_matrix = np.zeros(
        (len(vocabulary), len(vocabulary)),
        dtype=np.int32
    )

    for sentence in sentences :
        for prev_word, cur_word in zip(sentence, sentence[1:]) :
            prev_id, cur_id = word_to_id[prev_word], word_to_id[cur_word]

            cnt_matrix[prev_id, cur_id] += 1

    return (
        cnt_matrix,
        vocabulary,
        word_to_id
    )