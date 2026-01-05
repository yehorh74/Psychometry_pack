import pandas as pd
import numpy as np

class Alfa:
    
    def __init__(self, statistics, sort=None):
        self.statistics = statistics
        self.sort = sort
        self.results = None

    def fit(self, df_itemy):
        # Usuwamy wiersze z brakującymi wartościami
        df = df_itemy.dropna(axis=0)

        # Sprawdzamy, czy po usunięciu NaN mamy wystarczającą liczbę obserwacji
        if df.shape[0] < 2:
            raise ValueError("Za mało obserwacji po usunięciu NaN")

        # Sprawdzamy, czy mamy co najmniej 2 itemy (kolumny)
        if df.shape[1] < 2:
            raise ValueError("Alfa Cronbacha wymaga co najmniej 2 itemów")

        def _alfa_cronbacha_itemy(df_itemy):
            wyniki = {}

            for x in df_itemy.columns:
                # Tworzymy tymczasowy DataFrame bez kolumny x
                df_temp = df_itemy.drop(columns=[x])

                def wariancja(df_temp):
                    # Obliczamy wariancję poszczególnych kolumn
                    srednia = df_temp.mean()
                    kwad_odchyl_sred = (df_temp - srednia) ** 2
                    suma_kolumn = kwad_odchyl_sred.sum()
                    N = len(df_temp.index) - 1
                    return suma_kolumn / N

                def wariancja_calkowita(df_temp):
                    # Obliczamy wariancję całkowitą sumy punktów w wierszu
                    suma_punktow = df_temp.sum(axis=1)
                    srednia = suma_punktow.mean()
                    N = len(suma_punktow) - 1
                    return ((suma_punktow - srednia) ** 2).sum() / N

                # Suma wariancji po usunięciu kolumny x
                wariancja_suma = wariancja(df_temp).sum()
                # Wariancja całkowita sumy punktów
                wariancja_ogol = wariancja_calkowita(df_temp)
                N = df_temp.shape[1]

                # Obliczenie alfy Cronbacha po usunięciu itemu x
                alfa = (N / (N - 1)) * (1 - (wariancja_suma / wariancja_ogol))
                wyniki[x] = alfa

            # Zwracamy wynik jako DataFrame
            df_alfa_del = pd.DataFrame.from_dict(wyniki, orient="index", columns=["Alfa_if_item_deleted"])
            return df_alfa_del

        def _r(df_itemy):
            wyniki = {}

            # Obliczamy sumę punktów dla każdego wiersza (respondenta)
            total = df_itemy.sum(axis=1)

            for kol in df_itemy.columns:
                Xi = df_itemy[kol]
                rest = total - Xi

                # Obliczamy odchylenia standardowe itemu i sumy pozostałych itemów
                sd_X = np.std(Xi, ddof=1)
                sd_rest = np.std(rest, ddof=1)

                # Obliczamy kowariancję między itemem a sumą pozostałych
                cov_matrix = np.cov(Xi, rest, ddof=1)
                cov_11 = cov_matrix[1, 1]
                cov_10 = cov_matrix[1, 0]

                # Korelacja item–reszta (item–total correlation)
                r_it = cov_10 / (sd_X * sd_rest)
                wyniki[kol] = {"Scale_variance": cov_11, "Item_total_correlation": r_it}

            df_r = pd.DataFrame.from_dict(wyniki, orient="index")
            return df_r

        def _srednie(df_itemy):
            wyniki = {}

            for col in df_itemy.columns:
                # Usuwamy kolumnę col
                df_temp = df_itemy.drop(columns=[col])
                # Suma pozostałych itemów
                suma_wierszy = df_temp.sum(axis=1)
                # Średnia suma po usunięciu itemu
                srednia = suma_wierszy.mean()
                wyniki[col] = srednia

            df_mean = pd.DataFrame.from_dict(wyniki, orient="index")
            df_mean.columns = ['Mean']
            return df_mean

        if self.statistics == "item_dropped":
            # Obliczamy wskaźniki po usunięciu pojedynczego itemu
            wynik_alfa_del = _alfa_cronbacha_itemy(df)
            wynik_r = _r(df)
            wynik_mean = _srednie(df)

            # Łączymy wyniki w jedną tabelę
            tabela = pd.concat([wynik_alfa_del, wynik_r, wynik_mean], axis=1)
        
        else:
            raise ValueError(f"Nieznana wartość parametru statistics: {self.statistics}")

        if self.sort is not None:
            
            if self.sort == "alfa":
                tabela = tabela.sort_values(
                    by="Alfa_if_item_deleted", ascending=False
                )

            elif self.sort == "corr":
                tabela = tabela.sort_values(
                    by="Item_total_correlation", ascending=False
                )
            
            elif self.sort == "cov":
                tabela = tabela.sort_values(
                    by="Scale_variance", ascending=False
                )
                
            elif self.sort == "mean":
                tabela = tabela.sort_values(
                    by="Mean", ascending=False
                )
                
            else:
                raise ValueError(f"Nieznana wartość sort: {self.sort}")

        self.results = tabela
        return tabela

    @staticmethod
    def alfa_cronbacha(df_itemy):
        wyniki = {}

        # Usuwamy wiersze z brakującymi wartościami
        df = df_itemy.dropna(axis=0)

        if df.shape[0] < 2:
            raise ValueError("Za mało obserwacji po usunięciu NaN")

        if df.shape[1] < 2:
            raise ValueError("Alfa Cronbacha wymaga co najmniej 2 itemów")

        def wariancja(df_itemy):
            # Obliczamy wariancję kolumn
            srednia = df_itemy.mean()
            kwad_odchyl_sred = (df_itemy - srednia) ** 2
            suma_kolumn = kwad_odchyl_sred.sum()
            N = len(df_itemy.index) - 1
            return suma_kolumn / N

        def wariancja_calkowita(df_itemy):
            # Obliczamy wariancję całkowitą sumy punktów
            suma_punktow = df_itemy.sum(axis=1)
            srednia = suma_punktow.mean()
            N = len(suma_punktow) - 1
            return ((suma_punktow - srednia) ** 2).sum() / N

        wariancja_suma = wariancja(df_itemy).sum()
        wariancja_ogol = wariancja_calkowita(df_itemy)
        N = df_itemy.shape[1]

        # Obliczamy współczynnik alfa Cronbacha
        alfa = (N / (N - 1)) * (1 - (wariancja_suma / wariancja_ogol))

        wyniki["Alfa_Cronbacha"] = alfa
        wyniki["N"] = N

        df_alfa = pd.DataFrame.from_dict(wyniki, orient="index")

        return df_alfa