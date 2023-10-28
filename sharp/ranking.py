import numpy as np


class Ranking:
    def __init__(self, my_function, df):
        self.my_rank_function = my_function
        self.dataset = df
        self.ranks = self.clearbox_ranking()

    def clearbox_ranking(self):
        n_rows = self.dataset.shape[0]
        new_df = self.dataset.copy()
        # Rank based on weighted sum and sort
        new_df['Score'] = self.my_rank_function(self.dataset)
        # NOTE: Larger sum is better
        new_df = new_df.sort_values('Score', ascending=False)
        new_df['Rank'] = range(1, n_rows + 1)

        # Return df, score, and rank columns
        return new_df

    def blackbox_ranking(self):
        new_df = self.clearbox_ranking()
        # Return only rank
        return new_df['Rank']

    def predict_score(self, datapoint):
        ranking_score = self.my_rank_function(datapoint)
        return np.array(ranking_score)

    def predict_rank(self, datapoint):
        ranking_score = self.predict_score(datapoint)
        ranking_rank = []
        for rank in ranking_score:
            ranking_rank.append(self.ranks[self.ranks['Score'] > rank].shape[0] + 1)
        return np.array(ranking_rank)

    def get_rank(self, datapoint):
        if datapoint in self.ranks:
            return self.ranks.loc[[datapoint]]
        return None

    def get_all(self):
        return self.ranks
