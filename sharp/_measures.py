"""
Quantitative Input Influence measures.

- Set qii
- Unary qii
- Marginal qii
- Shapley
- Banzhaff
"""


def _unary(row_idx, col_idx, X, classifier, sample_size, rng):
    pass


def _set():
    pass


def _marginal():
    pass


def _shapley():
    pass


def _banzhaff():
    pass


MEASURES = {"unary", "set", "marginal", "shapley", "banzhaff"}

# TEMP


def _set_qii_row(row, columns, dataset, classifier, sample_size, rng):
    """
    Calculates the QII for a single or set of attributes in a single row.

    Parameters
    ----------
      columns [str, list]:
          The attribute that we are going to explain.
      row [pandas.series]:
          The dataframe row we are explaining.
      dataset [pandas.dataframe]:
          The dataset to use in order to test the classifier.
      classifier [function]:
          The machine learning model we used to predict the data.
      sample_size [int], default=30:
          how many times we calculate the QII.

      return [int]:
          the QII score of the attribute,
          -- how this attribute contribute to the machine.
    """
    # Set up data point, original prediction, and values to replace with
    row = row.to_frame().T.copy() if row.shape[0] != 1 else row.copy()
    y_pred = classifier.predict(row)

    # Drop the row we are explaining
    # TODO : eliminate copy here probably (but do drop the row if it exists)
    if row.index[0] in dataset.index:
        temp_dataset = dataset.drop([row.index[0]])
    else:
        temp_dataset = dataset.copy()

    # Draw new samples uniformly at random
    mod_rows = temp_dataset.sample(n=sample_size, axis=0)

    # Unary or Set, make a list of columns
    # TODO : Maybe check that the column is a real column and warn the user?
    if type(columns) == str:
        cols = [columns]
    else:
        cols = columns

    # Keep original values for the columns not in "columns"
    # TODO : Speed this up!!
    for col in dataset.columns:
        if col not in columns:
            mod_rows[col] = np.repeat(row[col].values, sample_size)

    # # Modify original row `sample_size` times and get predictions
    y_pred_mod = classifier.predict(mod_rows)

    # Return score
    return 1 - (y_pred_mod == y_pred).astype(int).mean()


def _marginal_qii_row(row, column, set_columns, dataset, classifier, sample_size, rng):
    """
    Calculates the marginal QII for a single or set of attributes in a single row.

    Parameters
    ----------
      row [pandas.series]:
          The dataframe row we are explaining.
      column [str]:
          The column we are explaining.
      set_columns [str, list]:
          The attribute (list) that we are going to use for the marginal.
      dataset [pandas.dataframe]:
          The dataset to use in order to test the classifier.
      classifier [function]:
          The machine learning model we used to predict the data.
      sample_size [int], default=30:
          how many times we calculate the QII.

      return [int]:
          the QII score of the attribute,
          -- how this attribute contribute to the machine.
    """
    # Set up data point, original prediction, and values to replace with
    row = row.to_frame().T.copy() if row.shape[0] != 1 else row

    # Drop the row we are explaining
    # TODO : eliminate copy here probably (but do drop the row if it exists)
    if row.index[0] in dataset.index:
        temp_dataset = dataset.drop([row.index[0]])
    else:
        temp_dataset = dataset.copy()

    # Draw new samples uniformly at random
    mod_rows2 = temp_dataset.sample(n=sample_size, axis=0)

    # TODO : Make sure "column" is not in "columns"
    # Make a list of columns
    # if type(set_columns) == str:
    #     if set_columns != column:
    #         cols = [set_columns]
    #     else:
    #         cols = []
    # else:
    #     cols = set_columns

    # Keep original values for the columns not in "columns"
    # TODO : Speed this up!!
    for col in dataset.columns:
        if (col not in set_columns) & (col != column):
            mod_rows2[col] = np.repeat(row[col].values, sample_size)

    # For mod_rows1, also remove "column"
    mod_rows1 = mod_rows2.copy()
    mod_rows1[column] = np.repeat(row[column].values, sample_size)

    # # Modify original row `sample_size` times and get predictions
    y_pred_mod1 = classifier.predict(mod_rows1)

    # Modify original row `sample_size` times (again) and get predictions
    y_pred_mod2 = classifier.predict(mod_rows2)

    # Return score
    return 1 - (y_pred_mod1 == y_pred_mod2).astype(int).mean()
