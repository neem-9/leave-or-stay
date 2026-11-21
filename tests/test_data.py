"""Data contract tests: catch a renamed column before it reaches training or production."""

import pandas as pd
import pytest

from src import schema
from src.train import train


def test_training_data_matches_schema():
    """The data file the pipeline will train on has every column the model needs."""
    schema.validate(pd.read_csv(train.DATA_FILE))


def test_missing_column_gives_a_clear_error():
    df = pd.DataFrame({"tenure_months": [1], "contract": ["one-year"]})
    with pytest.raises(schema.SchemaError, match="monthly_charges"):
        schema.validate(df)
