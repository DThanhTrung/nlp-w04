import numpy as np
import pandas as pd

from utils import (
    load_data_for_task_2, 
    preprocessing_pipeline,
    build_co_occurrence_matrix,
    compute_ppmi
)

def get_answer_for_task_2(
    test_dir : str = "data/test.csv"
) :
    raw_sentences = load_data_for_task_2()
    sentences = preprocessing_pipeline(raw_sentences)

    co_occurrence_matrix, vocabulary, word_to_id = build_co_occurrence_matrix(sentences)

    test_df = pd.read_csv(test_dir)[20:40]

    test_df_ids = test_df["id"]
    target_words = list(test_df["arg1"])
    test_df_options = test_df[["option_" + str(i) for i in range (4)]]

    questions_options = list(test_df_options.itertuples(index=False, name=None))
    labels = []

    for target_word, options in zip(target_words, questions_options) :
        target_vector = compute_ppmi(target_word, vocabulary, co_occurrence_matrix, word_to_id)

        options = list(options)

        cosines = []
        for context_word in options :
            context_vector = compute_ppmi(context_word, vocabulary, co_occurrence_matrix, word_to_id)

            cosines.append(
                np.dot(target_vector, context_vector) / (np.linalg.norm(target_vector) * np.linalg.norm(context_vector))
            )

        cosines = np.array(cosines)
        labels.append(cosines.argmax())

    result_df = pd.DataFrame({
        "id": test_df_ids.to_numpy(),
        "label": labels,
    })

    return result_df