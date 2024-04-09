from pathlib import Path

import polars as pl

COLS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
]


class Loader:
    def __init__(self):
        self.DATA_PATH = Path(__file__).parent.parent / "data"

    def load_data(self, file):
        cols = COLS
        if file == "train":
            cols = cols + ["TARGET"]
        return (
            pl.read_csv(self.DATA_PATH / f"application_{file}.csv")
            .to_pandas()
            .set_index("SK_ID_CURR")[cols]
        )
