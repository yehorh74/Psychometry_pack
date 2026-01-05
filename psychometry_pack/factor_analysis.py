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
        Dopasowanie modelu EFA.
        Jeśli n_factors=None → obliczane są tylko eigenvalues i testy (scree plot).
        Jeśli n_factors podano → wykonywana jest pełna analiza czynnikowa.
        """

        # --- WALIDACJA ---
        if not isinstance(df, pd.DataFrame):
            raise TypeError("fit() oczekuje pandas DataFrame")

        df_clean = df.dropna()

        if df_clean.shape[0] < 2:
            raise ValueError("Za mało obserwacji po usunięciu braków danych")

        if df_clean.shape[1] < 2:
            raise ValueError("Za mało zmiennych do analizy czynnikowej")

        # --- TESTY ADEKWATNOŚCI ---
        chi2, p = calculate_bartlett_sphericity(df_clean)
        _, kmo_model = calculate_kmo(df_clean)

        self.kmo = kmo_model
        self.bartlett = (chi2, p)

        # --- EIGENVALUES (ZAWSZE) ---
        fa_ev = FactorAnalyzer(rotation=None)
        fa_ev.fit(df_clean)
        ev, _ = fa_ev.get_eigenvalues()
        self.eigenvalues = ev

        if self.n_factors is None:
            self.loadings = None
            self.factor_scores = None
            return self

        # --- PEŁNA ANALIZA CZYNNIKOWA ---
        if not isinstance(self.n_factors, int) or self.n_factors < 1:
            raise ValueError("n_factors musi być liczbą całkowitą ≥ 1")

        fa = FactorAnalyzer(
            n_factors=self.n_factors,
            rotation=self.rotation,
            method=self.method
        )
        fa.fit(df_clean)

        # --- ŁADUNKI CZYNNIKOWE ---
        loadings = pd.DataFrame(
            fa.loadings_,
            index=df_clean.columns,
            columns=[f"Factor{i+1}" for i in range(fa.loadings_.shape[1])]
        )

        if self.loading_cutoff is not None:
            loadings = loadings.where(loadings.abs() >= self.loading_cutoff)

        self.loadings = loadings

        # --- WARTOŚCI CZYNNIKOWE ---
        scores = fa.transform(df_clean)
        self.factor_scores = pd.DataFrame(
            scores,
            columns=[f"Factor_{i+1}" for i in range(scores.shape[1])]
        )

        return self
    
    def add_factor_scores(self, df):
        """
        Zwraca DataFrame z dołączonymi wartościami czynnikowymi.
        DataFrame musi mieć te same wiersze co oryginalny df użyty w fit().
        """
        if self.factor_scores is None:
            raise RuntimeError("Najpierw wykonaj fit() z n_factors")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Oczekiwano pandas DataFrame")

        # Zakładamy, że df i self.factor_scores mają takie same indeksy i długości
        if len(df) != len(self.factor_scores):
            raise ValueError("Niezgodna liczba obserwacji między df a factor_scores")

        df = df.reset_index(drop=True)
        factor_scores = self.factor_scores.reset_index(drop=True)

        return pd.concat([df, factor_scores], axis=1)
    
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

    def save_results(self, prefix="efa", how="csv"):
        """
        Zapis wyników do plików CSV lub Excel.

        Parametry:
        - prefix (str): prefiks nazwy pliku
        - how (str): format zapisu, "csv" lub "excel" (domyślnie "csv")
        """
        if how == "excel":
            with pd.ExcelWriter(f"{prefix}_results.xlsx") as writer:
                pd.DataFrame(
                    {"eigenvalues": self.eigenvalues},
                    index=range(1, len(self.eigenvalues) + 1)
                    ).to_excel(writer, sheet_name="Eigenvalues")

                self.loadings.to_excel(writer, sheet_name="Loadings")

                self.factor_scores.to_excel(writer, sheet_name="Factor_Scores", index=False)

                pd.DataFrame({
                    "KMO": [self.kmo],
                    "Chi-square": [self.bartlett[0]],
                    "p-value": [self.bartlett[1]]
                    }).to_excel(writer, sheet_name="Tests", index=False)
        else:
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