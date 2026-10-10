def load_data_for_task_1(
    data_path: str = "data/lm_corpus.txt"
) -> list[str] :
    sentences = []

    with open(data_path, "r") as f :
        for line in f :
            sentences.append(line.strip())

    return sentences


def load_data_for_task_2(
    data_path: str = "data/semantic_corpus.txt"
) -> list[str] :
    sentences = []

    with open(data_path, "r") as f :
        for line in f :
            sentences.append(line.strip())

    return sentences