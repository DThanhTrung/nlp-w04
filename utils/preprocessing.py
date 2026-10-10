def _add_s_single_sentence(
    sentence: list[str]
) -> list[str]:
    return ["<s>"] + sentence + ["</s>"]


def _split_sentences(
    raw_sentences: list[str]
) -> list[list[str]] :
    return [
        list(map(str.lower, s.split())) for s in raw_sentences
    ]


def add_s(
    sentences: list[list[str]]
) -> list[list[str]] :
    return [
        _add_s_single_sentence(s) for s in sentences
    ]


def preprocessing_pipeline(
    raw_sentences: list[str]
) -> list[list[str]] :
    sentences = _split_sentences(raw_sentences)
    return add_s(sentences)