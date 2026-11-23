import numpy as np
import pandas as pd

TRAIN_DATA_PATH = "data/w12/kaggle_data_upload/emb_train.npy"
VALID_DATA_PATH = "data/w12/kaggle_data_upload/emb_valid.npy"
TEST_DATA_PATH = "data/w12/kaggle_data_upload/emb_test.npy"

TRAIN_LABEL_PATH = "data/w12/kaggle_data_upload/train.csv"
VALID_LABEL_PATH = "data/w12/kaggle_data_upload/valid.csv"
TEST_LABEL_PATH = "data/w12/kaggle_data_upload/test.csv"



def load_data(
    data_path: str = TRAIN_DATA_PATH,
    label_path: str = TRAIN_LABEL_PATH
) :
    data = np.load(data_path)
    label = pd.read_csv(label_path)['label'].to_numpy()

    return data, label


def load_train_data() :
    return load_data(data_path=TRAIN_DATA_PATH, label_path=TRAIN_LABEL_PATH)

def load_valid_data() :
    return load_data(data_path=VALID_DATA_PATH, label_path=VALID_LABEL_PATH)

def load_test_data() :
    test_data = np.load(TEST_DATA_PATH)

    test_id = pd.read_csv(TEST_LABEL_PATH)['id'].to_numpy()
    return test_data, test_id