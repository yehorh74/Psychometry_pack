import pandas as pd
import numpy as np
from .recode import Recode

def _check_dataframe(self, min_rows=2, min_cols=2):
    """
    Sprawdza poprawność danych wejściowych
    """

    if not isinstance(self, pd.DataFrame):
        raise TypeError("Dane muszą być DataFrame")

    if self.shape[0] < min_rows:
        raise ValueError("Za mało obserwacji")

    if self.shape[1] < min_cols:
        raise ValueError("Za mało zmiennych")

    return self

def _dropna_rows(self):
    """
    Usuwa wiersze z brakami danych
    """
    return self.dropna(axis=0)

def _sum_items(
    self,
    kolumny,
    new_col=None,
    skipna=True
):
    """
    Sumuje wskazane kolumny w wierszach
    """

    suma = self[kolumny].sum(axis=1, skipna=skipna)

    if new_col is not None:
        df = self.copy()
        df[new_col] = suma
        return df

    return suma

def _recode_values(self, kolumny_do_rek, wart, wart_rek, keep_other=False):
    """
    Rekoduje wartości w wybranych kolumnach
    """
    return Recode.recode_values(
        self,
        kolumny_do_rek,
        wart,
        wart_rek,
        keep_other=keep_other
    )

pd.DataFrame.check = _check_dataframe
pd.DataFrame.dropna_rows = _dropna_rows
pd.DataFrame.sum_items = _sum_items
pd.DataFrame.recode_values = _recode_values