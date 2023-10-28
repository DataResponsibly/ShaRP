"""
Quantities of Interest.
"""


class QoI:
    """
    TODO: add documentation
    """

    def __init__(self, model):
        self.model = model

    def estimate(self, rows):
        return self.model.predict(rows)

    def calculate(self, q1, q2):
        return (self.estimate(q1) - self.estimate(q2)).mean()


class BCFlipped(QoI):
    """
    TODO: add documentation
    eq 4 from paper
    """

    def calculate(self, q1, q2):
        y_pred_mod = self.estimate(q2)
        y_pred = self.estimate(q1)
        return 1 - (y_pred_mod == y_pred).astype(int).mean()


class BCLikelihood(QoI):
    """
    TODO: add documentation
    eq 3 from paper
    """

    def __init__(self, model, label):
        self.model = model
        self.label = label

    def estimate(self, rows):
        return self.model.predict_proba(rows)[:, self.label].mean()
