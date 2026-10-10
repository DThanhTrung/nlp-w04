import numpy as np
import pandas as pd

from utils import (
    load_data_for_task_1, 
    preprocessing_pipeline,
    build_bigram_matrix,
    compute_nll
)

def main(
    test_dir : str = "data/test.csv"
) :
    raw_sentences = load_data_for_task_1()
    sentences = preprocessing_pipeline(raw_sentences)

    cnt_matrix, vocabulary, word_to_id = build_bigram_matrix(sentences)
    V = len(vocabulary)

    test_df = pd.read_csv(test_dir)[:20]
    test_df_ids = test_df["id"]
    test_df_options = test_df[["option_" + str(i) for i in range (4)]]

    questions_options = list(test_df_options.itertuples(index=False, name=None))
    labels = []

    for options in questions_options :
        options_splitted = preprocessing_pipeline(options)

        points = np.array(
            [compute_nll(option, cnt_matrix, word_to_id, V) for option in options_splitted]
        )
        labels.append(points.argmin())

    result_df = pd.DataFrame({
        "id": test_df_ids.to_numpy(),
        "label": labels,
    })

    print(result_df)
    return result_df


if __name__ == "__main__" :
    task_1_ans = main()