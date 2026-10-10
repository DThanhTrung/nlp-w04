import pandas as pd

from tasks import (
    get_answer_for_task_1,
    get_answer_for_task_2,
    get_answer_for_task_3
)

if __name__ == "__main__" :
    task_1_answer = get_answer_for_task_1()
    task_2_answer = get_answer_for_task_2()
    task_3_answer = get_answer_for_task_3()

    final_answer = pd.concat([
        task_1_answer,
        task_2_answer,
        task_3_answer
    ])

    final_answer.to_csv("submission.csv", index=None)