import random
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

random.seed(123)


class InteractionDataset(Dataset):
    """Wrapper, convert <user, item, rel_int, neg_item> Tensor into Pythorch Dataset"""

    def __init__(self, user_tensor, item_tensor, rel_int_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.rel_int_tensor = rel_int_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return (
            self.user_tensor[index],
            self.item_tensor[index],
            self.rel_int_tensor[index],
            self.target_tensor[index],
        )

    def __len__(self):
        return self.user_tensor.size(0)


data_path = "data/processed.csv"

df = pd.read_csv(data_path, converters={"relational_interval": literal_eval})

user_pool = set(df["userId"].unique())
item_pool = set(df["itemId"].unique())

df_test = df[df["set"] == "test"].copy()
df_val = df[df["set"] == "val"].copy()
df_train = df[(df["set"] == "train")].copy()

print("The size of the training set is: {}".format(len(df_train)))


# function that returns the train and test set
def get_train_test_val():
    return df_test, df_train, df_val


# function that returns the number of users and items
def get_n_users_items():
    return df.userId.nunique(), df.itemId.nunique()


def get_negatives():
    return df_negative


# build the training set
def instance_a_train_loader(num_negatives, batch_size):
    users, items, rel_int, interests = [], [], [], []
    train_stream = df_train.merge(
        df_negative[["userId", "negative_items"]], on="userId"
    )

    for row in train_stream.itertuples():
        users.append(int(row.userId))
        items.append(int(row.itemId))
        interests.append(int(row.y))

        # add -1 to the rel_int until arriving at the max(50 reps)
        ri = row.relational_interval
        ri = np.pad(ri, (0, 50 - len(ri)), constant_values=-1)
        rel_int.append(ri)

    dataset = InteractionDataset(
        user_tensor=torch.LongTensor(users),
        item_tensor=torch.LongTensor(items),
        rel_int_tensor=torch.FloatTensor(np.array(rel_int)),
        target_tensor=torch.LongTensor(interests),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# create the evaluation dataset (user x item consumption sequences)
def evaluate_data():
    test_users, test_items, test_rel_int, test_listen = [], [], [], []

    for row in df_val.itertuples():
        ri = row.relational_interval
        ri = np.pad(ri, (0, 50 - len(ri)), constant_values=-1)

        test_rel_int.append(ri)
        test_users.append(int(row.userId))
        test_items.append(int(row.itemId))
        test_listen.append(int(row.y))

    return [
        torch.LongTensor(test_users),
        torch.LongTensor(test_items),
        torch.FloatTensor(np.array(test_rel_int)),
        torch.FloatTensor(test_listen),
    ]
