import numpy as np


def _set_qii_row(qoi, row, columns, dataset, sample_size, rng):
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

    # Drop the row we are explaining (if exists)
    if row.index[0] in dataset.index:
        temp_dataset = dataset.drop([row.index[0]])
    else:
        temp_dataset = dataset.copy()

    # Draw new samples uniformly at random
    mod_rows = temp_dataset.sample(n=sample_size, axis=0, random_state=rng)

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

    # Return score
    return qoi.calculate(row, mod_rows)


def _marginal_qii_row(
    qoi,
    row,
    column,
    set_columns,
    dataset,
    sample_size,
    rng
):
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
    if row.index[0] in dataset.index:
        temp_dataset = dataset.drop([row.index[0]])
    else:
        temp_dataset = dataset.copy()

    # Draw new samples uniformly at random
    mod_rows2 = temp_dataset.sample(n=sample_size, axis=0, random_state=rng)

    # Keep original values for the columns not in "columns"
    # TODO : Speed this up!!
    for col in dataset.columns:
        if (col not in set_columns) & (col != column):
            mod_rows2[col] = np.repeat(row[col].values, sample_size)

    # For mod_rows1, also remove "column"
    mod_rows1 = mod_rows2.copy()
    mod_rows1[column] = np.repeat(row[column].values, sample_size)

    # Return score
    return qoi.calculate(mod_rows1, mod_rows2)
