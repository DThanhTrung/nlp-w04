import numpy as np

def _build_vocabulary(
    sentences: list[list[str]]
) -> list[str] :
    return sorted({
        word
        for sentence in sentences
        for word in sentence
    })

def _build_word_to_id(
    vocabulary: list[str]
) -> dict[str, int] :
    return {
        word: index
        for (index, word) in enumerate(vocabulary)
    }


def build_bigram_matrix(
    sentences: list[list[int]]
) -> tuple[np.ndarray, list[str], dict[str, int]] :
    vocabulary = _build_vocabulary(sentences)

    word_to_id = _build_word_to_id(vocabulary)

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


def build_co_occurrence_matrix(
    sentences : list[list[str]],
    window: int = 2
) -> tuple[np.ndarray, list[str], dict[str, int]] :
    assert(2 == window)

    vocabulary = _build_vocabulary(sentences)

    word_to_id = _build_word_to_id(vocabulary)

    cnt_matrix = np.zeros(
        (len(vocabulary), len(vocabulary)),
        dtype=np.int32
    )

    for sentence in sentences :
        for target_index, target_word in enumerate(sentence) :
            target_id = word_to_id[target_word]
            left = max(0, target_index - window)
            right = min(len(sentence), target_index + window + 1)

            for context_index in range(left, right) :
                if context_index == target_index :
                    continue

                context_id = word_to_id[sentence[context_index]]
                cnt_matrix[target_id, context_id] += 1

    return (
        cnt_matrix,
        vocabulary,
        word_to_id
    )