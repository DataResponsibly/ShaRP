from qii.ranking import Ranking
from itertools import combinations
import numpy as np
import pandas as pd


class QoI:
    def __init__(self, model):
        self.model = model

    def estimate(self, rows):
        return self.model.predict(rows)

    def calculate(self, q1, q2):
        return (self.estimate(q1) - self.estimate(q2)).mean()

    def pairwise(self, q1, q2, q3, q4):
        return self.calculate(q1, q2) - self.calculate(q3, q4)


class BCFlipped(QoI):
    # eq 4 from paper
    def calculate(self, q1, q2):
        y_pred_mod = self.estimate(q2)
        y_pred = self.estimate(q1)
        return 1 - (y_pred_mod == y_pred).astype(int).mean()


class BCLikelihood(QoI):
    # eq 3 from paper
    def __init__(self, model, label):
        self.model = model
        self.label = label

    def estimate(self, rows):
        return self.model.predict_proba(rows)[:, self.label].mean()


class RankingRank(QoI):
    def __init__(self, dataset, ranking_function):
        self.ranking = Ranking(ranking_function, dataset)

    def estimate(self, rows):
        return self.ranking.predict_rank(rows)

    def calculate(self, q1, q2):
        return (self.estimate(q2) - self.estimate(q1)).mean()

    def get_ranking(self):
        return self.ranking.get_all()


class RankingScore(QoI):
    def __init__(self, dataset, ranking_function):
        self.ranking = Ranking(ranking_function, dataset)

    def estimate(self, rows):
        return self.ranking.predict_score(rows)

    def get_ranking(self):
        return self.ranking.get_all()


class RankingTopK(QoI):
    def __init__(self, dataset, ranking_function, k):
        self.ranking = Ranking(ranking_function, dataset)
        self.k = k

    def estimate(self, rows):
        ranks = self.ranking.predict_rank(rows)
        return (ranks <= self.k).astype(int)

    def get_ranking(self):
        return self.ranking.get_all()


class RankingNoFunction(QoI):
    def __init__(self, dataset, K, score_col_name, rank_col_name):
        self.ranking = dataset
        self.ranking.rename(columns={score_col_name: "Score", rank_col_name: "Rank"})
        self.K = K

        self.coal_contributions = self.calculate_coalition_contributions()

    def get_table(self):
        return self.coal_contributions

    def estimate(self, conditions):
        # TODO, this is only ranking
        try:
            rank = self.coal_contributions[pd.eval(conditions) == True]['Avg Rank'].values
            score = self.coal_contributions[pd.eval(conditions) == True]['Avg Score'].values
            topK = self.coal_contributions[pd.eval(conditions) == True]['Avg topK'].values
        except:
            print("Didn't find the entry")
            self.print_table()
            rank = 0
            score = 0
            topK = 0
        return rank, score, topK

    def calculate(self, conditions2, conditions1):
        rank2, score2, topK2 = self.estimate(conditions2)
        rank1, score1, topK1 = self.estimate(conditions1)
        return rank2 - rank1, score1 - score2, topK1 - topK2

    def calculate_coalition_contributions(self):
        ftr_names = self.ranking.columns.tolist()
        coal_ftr_names = ftr_names
        coal_ftr_names.remove('Score')
        coal_ftr_names.remove('Rank')

        # Create a row that contains null everywhere and the average rank and average score of the entire df
        all_null = {'size': [self.ranking.shape[0]],
                    'Avg Score': [self.ranking['Score'].mean()],
                    'Avg Rank': [self.ranking['Rank'].mean()],
                    'Avg topK': [self.K / self.ranking.shape[0]]
                    }

        # Save the avg score and avg rank of the coalition in a dataframe
        coalitions = pd.DataFrame(all_null)

        # Calculate coalition avg score an avg rank
        for set_size in range(1, len(coal_ftr_names) + 1):
            for set_columns in combinations(coal_ftr_names, set_size):
                # Get number of items per coalition
                temp_df = self.ranking.groupby(by=list(set_columns), as_index=False, group_keys=True)

                # Get sum of scores per coalition
                temp_df2 = pd.merge(temp_df.size(), temp_df['Score'].sum(), on=set_columns)

                # Get sum of ranks per coalition
                temp_df3 = pd.merge(temp_df2, temp_df['Rank'].sum(), on=set_columns)

                temp_df3['topK'] = temp_df['Rank'].apply(lambda x: (x <= self.K).sum())['Rank']

                temp_df = temp_df3

                # Calculate average sum and rank
                temp_df['Avg Score'] = temp_df['Score'] / temp_df['size']
                temp_df['Avg Rank'] = temp_df['Rank'] / temp_df['size']
                temp_df['Avg topK'] = temp_df['topK'] / temp_df['size']
                temp_df.drop(columns=['Rank', 'Score'], inplace=True)

                # Save coalition in the df
                coalitions = pd.concat([coalitions, temp_df], ignore_index=True)

        return coalitions
