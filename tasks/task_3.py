import numpy as np
import pandas as pd

def get_answer_for_task_3(
    test_dir : str = "data/test.csv",
    embedding_dir : str = "data/embeddings.csv"
) :
    test_df = pd.read_csv(test_dir)[40:]

    test_df_ids = test_df["id"]

    test_df_words = test_df[["arg" + str(i) for i in range (1, 4)]]
    questions_words = list(test_df_words.itertuples(index=False, name=None))

    test_df_options = test_df[["option_" + str(i) for i in range (4)]]
    questions_options = list(test_df_options.itertuples(index=False, name=None))

    embedding_df = pd.read_csv(embedding_dir)

    labels = []
    for words, options in zip(questions_words, questions_options) :
        word_vectors = np.array([
            embedding_df[word == embedding_df["word"]][["d" + str(i) for i in range (40)]]
            for word in words
        ]).squeeze()

        q = word_vectors[1, :] - word_vectors[0, :] + word_vectors[2, :]
        q /= np.linalg.norm(q)

        option_vectors = np.array([
            embedding_df[word == embedding_df["word"]][["d" + str(i) for i in range (40)]]
            for word in options
        ]).squeeze()

        option_vectors_norms = np.linalg.norm(option_vectors, axis=1, keepdims=True)
        option_vectors /= option_vectors_norms

        cosine_sim = option_vectors @ q
        labels.append(cosine_sim.argmax())

    result_df = pd.DataFrame({
        "id": test_df_ids.to_numpy(),
        "label": labels,
    })

    return result_df