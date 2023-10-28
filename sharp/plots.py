from matplotlib import pyplot as plt
from qii import group_set_qii, group_marginal_qii, shapley_score, banzhaf_score
import pandas as pd
from itertools import combinations
from mlresearch.utils import parallel_loop
from pathlib import Path
import math
import seaborn as sns


# Graph of fig 2a,2b / eq 5 from paper
def global_unary_plot(qoi, dataset, rng, sample_size):
    def _function(col):
        qii = group_set_qii(
            qoi=qoi,
            columns=col,
            dataset=dataset,
            random_state=rng,
            sample_size=sample_size,
        )
        return col, qii.mean()

    unary_quant = []
    for col in dataset.columns:
        qii_temp = _function(col)
        unary_quant.append(qii_temp)

    unary_results = pd.DataFrame(
        [(c, q) for c, q in unary_quant], columns=["Features", "Unary"]
    )
    ax = (
        unary_results.set_index("Features")
        .sort_values("Unary", ascending=False)
        .plot.bar()
    )
    ax.set_ylabel("Score")
    ax.set_title("Global Scores (Unary QII)")


def global_set_combo_plot(qoi, dataset, rng, sample_size, limit, num):
    # currently this implementation is quite time consuming, parallelizing for now; but must be optimized later.
    def _function(col_set):
        qii = group_set_qii(
            qoi=qoi,
            columns=col_set,
            dataset=dataset,
            random_state=rng,
            sample_size=sample_size,
        )
        return col_set, qii.mean()

    set_quant = []
    for cols in list(combinations(dataset.columns, num)):
        qii_temp = _function(cols)
        if limit:
            if qii_temp[1] > 0.075:
                set_quant.append(qii_temp)
        else:
            set_quant.append(qii_temp)
    # %%
    set_results = pd.DataFrame(
        [(" + ".join(c), q) for c, q in set_quant], columns=["Feature sets", "Set"]
    )
    ax = (
        set_results.set_index("Feature sets")
        .sort_values("Set", ascending=False)
        .plot.bar()
    )
    ax.set_ylabel("Score")
    ax.set_title("Global Scores (Set QII) - Over 0.075")


def global_marginal_plot(qoi, dataset, rng, sample_size):
    # currently this implementation is quite time consuming, parallelizing for now; but must be optimized later.
    def _function(feature):
        return (
            feature,
            group_marginal_qii(
                qoi=qoi,
                column=feature,
                set_columns=dataset.columns.drop([feature]),
                dataset=dataset,
                random_state=rng,
                sample_size=sample_size,
            ).mean(),
        )

    marginal_quant = []
    for cols in dataset.columns:
        qii_temp = _function(cols)
        marginal_quant.append(qii_temp)

    marginal_results = pd.DataFrame(
        [(c, q) for c, q in marginal_quant], columns=["Features", "Marginal"]
    )
    ax = (
        marginal_results.set_index("Features")
        .sort_values("Marginal", ascending=False)
        .plot.bar()
    )
    ax.set_ylabel("Score")
    ax.set_title("Global Scores (Marginal QII)")


def importance_plot(
    qoi, row_index, dataset, rng, sample_size, function_type, show=True
):
    if function_type.lower() == "banzhaf":
        function_type = "Banzhaf"
        score_function = banzhaf_score
    else:
        function_type = "Shapley"
        score_function = shapley_score

    # currently this implementation is quite time consuming, parallelizing for now; but must be optimized later.
    def _function(feature, row):
        return feature, score_function(
            qoi=qoi,
            row=row,
            dataset=dataset,
            target=feature,
            random_state=rng,
            iterate_time=sample_size,
        )

    shapley_quant = []
    for cols in dataset.columns:
        qii_temp = _function(cols, dataset.loc[[row_index]])
        shapley_quant.append(qii_temp)

    shapley_results = pd.DataFrame(
        [(c, q) for c, q in shapley_quant], columns=["Features", function_type]
    )

    if show:
        ax = (
            shapley_results.set_index("Features")
            .sort_values(function_type, ascending=False)
            .plot.bar()
        )
        ax.set_ylabel("Score")
        ax.set_title(function_type + " Scores")

    return shapley_results


def fig1(qoi, dataset, rng, sample_size):
    def _function(col):
        qii = group_set_qii(
            qoi=qoi,
            columns=col,
            dataset=dataset,
            random_state=rng,
            sample_size=sample_size,
        )
        return col, qii

    unary_ind_quant = parallel_loop(
        _function,
        iterable=dataset.columns,
        n_jobs=-1,
        progress_bar=True,
        description="Retrieving QII",
    )
    # Figure 1
    ax = pd.DataFrame(dict(unary_ind_quant)).apply(max, axis=1).plot.hist(bins=10)
    ax.set_xlabel("Maximum Influence of some input")
    ax.set_ylabel("Number of individuals")


def group_disparity_plot(
    qoi, column, dataset, random_state=42, sample_size=30, save_pic=False
):
    # Figure 2 - (c) and (d) - cat_0 - A11 vs A14
    def _function(col, x):
        return col, group_set_qii(
            qoi=qoi,
            columns=col,
            dataset=x,
            random_state=random_state,
            sample_size=sample_size,
        )

    unary_quant = {}
    for group in dataset[column].unique():
        unary_quant_group = parallel_loop(
            lambda col: _function(col, dataset[dataset[column] == group]),
            iterable=dataset.columns,
            n_jobs=-1,
            progress_bar=True,
            description=("Retrieving QII - " + str(group)),
        )
        unary_quant[group] = unary_quant_group

    dfs_results = []
    for name, quant in unary_quant.items():
        results = pd.DataFrame(
            [(c, q[0]) for c, q in quant], columns=["Features", name]
        ).set_index("Features")
        dfs_results.append(results)

    dfs_cat_0s = pd.concat(dfs_results, axis=1)

    for cols in list(combinations(dfs_cat_0s.columns, 2)):
        ax = (dfs_cat_0s[cols[0]] - dfs_cat_0s[cols[1]]).plot.bar()
        ax.set_ylabel("Score difference")
        ax.set_title("QII on Group Disparity by Race, " + str(cols))

        plt.axhline(y=0.01, color="black", linestyle="--", linewidth=1, label="Avg")

        plt.show()


def global_contributions(
    qoi,
    df,
    seed,
    sample_size,
    ranking_function,
    filepath,
    strata=10,
    aggr_type="Shapley",
):
    # Iterate over the entire dataset
    iterable = list(df.index)

    open(filepath, "x")

    # Calculate ShaRP for each item
    result_cols = ["Label", "Score", "Score bucket", "Feature", "Contribution"]

    def _qoi_exp(ind):
        # Create dataframe for the results
        df_sharp = pd.DataFrame(columns=result_cols)
        temp2 = importance_plot(qoi, ind, df, seed, sample_size, aggr_type, False)
        for _, t_row in temp2.iterrows():
            score = ranking_function(df.loc[[ind]])[0]
            temp = [
                [ind, score, str(round(score, 1)), t_row["Features"], t_row["Shapley"]]
            ]
            temp3 = pd.DataFrame(data=temp, columns=result_cols)
            df_sharp = pd.concat([df_sharp, temp3], ignore_index=True)
        df_sharp.to_csv(filepath, mode="a", index=False, header=False)

    parallel_loop(
        _qoi_exp, iterable, n_jobs=-1, progress_bar=True, description="Global ShaRP"
    )

    # Create Rank column
    df_contrs = pd.read_csv(Path(filepath), names=result_cols)
    df_ranks = qoi.get_ranking()
    df_all = df_ranks.merge(df_contrs, right_on="Label", left_index=True)

    rows = len(df_ranks.index)
    df_all["Rank Stratum"] = [
        "<10%"
        if math.floor((rank - 1) / (rows / strata)) == 0
        else str(math.floor((rank - 1) / (rows / strata)) * 10)
        + "-\n"
        + str(math.floor(((rank - 1) / (rows / strata)) + 1) * 10)
        + "%"
        for rank in df_all["Rank"]
    ]

    # Save dataframe to file
    filepath = Path(filepath)
    df_all.to_csv(filepath)

    return df_all
