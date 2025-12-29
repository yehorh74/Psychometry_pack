import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (
    calculate_bartlett_sphericity,
    calculate_kmo
)

class FactorAnalysis:
    """
    Klasa do eksploracyjnej analizy czynnikowej (EFA)
    """

    def __init__(
        self,
        n_factors=None,
        rotation="varimax",
        method="principal",
        loading_cutoff=None
    ):
        self.n_factors = n_factors
        self.rotation = rotation
        self.method = method
        self.loading_cutoff = loading_cutoff

        self.eigenvalues = None
        self.loadings = None
        self.factor_scores = None
        self.kmo = None
        self.bartlett = None

    def fit(self, df):
        """
        Dopasowanie modelu EFA
        """
        df_clean = df.dropna()

        #Testy adekwatności
        chi2, p = calculate_bartlett_sphericity(df_clean)
        _, kmo_model = calculate_kmo(df_clean)

        self.kmo = kmo_model
        self.bartlett = (chi2, p)

        #Eigenvalues
        fa_ev = FactorAnalyzer(rotation=None)
        fa_ev.fit(df_clean)
        ev, _ = fa_ev.get_eigenvalues()
        self.eigenvalues = ev

        #Analiza czynnikowa
        fa = FactorAnalyzer(
            n_factors=self.n_factors,
            rotation=self.rotation,
            method=self.method
        )
        fa.fit(df_clean)

        #Ładunki czynnikowe
        loadings = pd.DataFrame(
            fa.loadings_,
            index=df_clean.columns,
            columns=[f"Factor{i+1}" for i in range(fa.loadings_.shape[1])]
        )

        loadings = loadings.where(loadings.abs() >= self.loading_cutoff)
        self.loadings = loadings

        #Wartości czynnikowe
        scores = fa.transform(df_clean)
        self.factor_scores = pd.DataFrame(
            scores,
            columns=[f"Factor_{i+1}" for i in range(scores.shape[1])]
        )

        return self

    def scree_plot(self):
        """
        Wykres osypiska
        """
        x = range(1, len(self.eigenvalues) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(x, self.eigenvalues, marker="o")
        plt.axhline(1, linestyle="--")
        plt.xlabel("Czynnik")
        plt.ylabel("Wartość własna")
        plt.title("Scree plot")
        plt.grid(True)
        plt.show()

    def save_results(self, prefix="efa"):
        """
        Zapis wyników do plików CSV
        """
        pd.DataFrame(
            {"eigenvalues": self.eigenvalues},
            index=range(1, len(self.eigenvalues) + 1)
        ).to_csv(f"{prefix}_eigenvalues.csv")

        self.loadings.to_csv(f"{prefix}_loadings.csv")

        self.factor_scores.to_csv(f"{prefix}_factor_scores.csv", index=False)

        pd.DataFrame({
            "KMO": [self.kmo],
            "Chi-square": [self.bartlett[0]],
            "p-value": [self.bartlett[1]]
        }).to_csv(f"{prefix}_tests.csv", index=False)