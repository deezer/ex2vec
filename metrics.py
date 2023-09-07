import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score


class EvalMetrics(object):
    def __init__(self) -> None:
        self._subjects = None

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
        subjects: list, [test_users, test_items, test_scores]
        """

        assert isinstance(subjects, list)
        test_users, test_items, test_scores, test_y = (
            subjects[0],
            subjects[1],
            subjects[2],
            subjects[3],
        )

        test_set = pd.DataFrame(
            {"user": test_users, "item": test_items, "score": test_scores, "y": test_y}
        )

        self._subjects = test_set

    def cal_acc(self):
        """compute accuracy"""
        test_set = self._subjects
        test_set["pred"] = 0
        test_set.loc[test_set["score"] >= 0.5, "pred"] = 1

        pred = test_set["pred"].values
        true = test_set["y"].values
        return accuracy_score(true, pred)

    def cal_recall(self):
        """compute recall"""
        test_set = self._subjects
        test_set["pred"] = 0
        test_set.loc[test_set["score"] >= 0.5, "pred"] = 1

        pred = test_set["pred"].values
        true = test_set["y"].values
        return recall_score(true, pred)

    def cal_f1(self):
        """compute weighted F1 score"""
        test_set = self._subjects
        test_set["pred"] = 0
        test_set.loc[test_set["score"] >= 0.5, "pred"] = 1

        pred = test_set["pred"].values
        true = test_set["y"].values
        return f1_score(true, pred, average="weighted")
