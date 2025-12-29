import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

class Correlate:

    @staticmethod
    def rho_spearman(df, kol_ref, kolumny=None):
        wyniki = {}

        # Jeśli nie podano listy kolumn do analizy, bierzemy wszystkie poza kol_ref
        if kolumny is None:
            kolumny = [k for k in df.columns if k != kol_ref]

        for kol in kolumny:
            data = df[[kol, kol_ref]].dropna()

            if len(data) < 2:
                wyniki[kol] = {"rho": np.nan, "p_value": np.nan, "N": len(data)}
                continue

            # Obliczamy współczynnik rho Spearmana i wartość p
            rho, p_value = spearmanr(data[kol], data[kol_ref])

            wyniki[kol] = {"rho": rho, "p_value": p_value, "N": len(data)}

        df_rho = pd.DataFrame.from_dict(wyniki, orient="index")

        return df_rho

    @staticmethod
    def r_pearson(df, factor, wyniki_test):
        wyniki = {}

        for kol in wyniki_test:
            data = df[[kol, factor]].dropna()

            if len(data) < 2:
                wyniki[kol] = {"r": np.nan, "p_value": np.nan, "N": len(data)}
                continue

            # Obliczamy współczynnik korelacji Pearsona i wartość p
            r, p_value = pearsonr(data[factor], data[kol])

            wyniki[kol] = {
                "r": r,
                "p_value": p_value,
                "N": len(data)
            }

        df_pearson = pd.DataFrame.from_dict(wyniki, orient="index")

        return df_pearson