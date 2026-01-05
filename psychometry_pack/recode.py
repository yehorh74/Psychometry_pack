import pandas as pd
import numpy as np


class Recode:
    """
    Klasa do rekodowania wartości w DataFrame
    """

    @staticmethod
    def recode_values(
        df,
        kolumny_do_rek,
        wart,
        wart_rek,
        keep_other=False
    ):
        """
        Rekoduje wartości w wybranych kolumnach DataFrame.

        Parametry
        ----------
        df : pd.DataFrame
            Dane wejściowe
        kolumny_do_rek : list
            Lista kolumn do rekodowania
        wart : list
            Lista wartości oryginalnych
        wart_rek : list
            Lista wartości po rekodowaniu
        keep_other : bool, default False
            Jeśli False – wartości spoza mapy zamieniane są na NaN
            Jeśli True  – wartości spoza mapy pozostają bez zmian

        Zwraca
        -------
        pd.DataFrame
            DataFrame po rekodowaniu
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df musi być pandas DataFrame")

        if len(wart) != len(wart_rek):
            raise ValueError("wart i wart_rek muszą mieć tę samą długość")

        df_rek = df.copy()

        # Tworzymy słownik mapowania
        mapa = dict(zip(wart, wart_rek))

        for kol in kolumny_do_rek:
            if kol not in df_rek.columns:
                raise ValueError(f"Kolumna {kol} nie istnieje w DataFrame")

            if keep_other:
                df_rek[kol] = df_rek[kol].replace(mapa)
            else:
                df_rek[kol] = df_rek[kol].map(mapa)

        return df_rek