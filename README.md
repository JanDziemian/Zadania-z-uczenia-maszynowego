# Przetwarzanie i Klasyfikacja Danych: CIFAR-100, Wine Quality & California Housing

![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-Latest-green.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)

Kompleksowy projekt z zakresu uczenia maszynowego i głębokiego uczenia (Deep Learning) zrealizowany w środowisku Jupyter Notebook. Projekt obejmuje pełny proces przetwarzania danych, filtrowanie, wstępną obróbkę (preprocessing), budowę oraz ewaluację zaawansowanych modeli predykcyjnych (klasyfikacja oraz regresja) dla danych obrazowych i tabelarycznych.

---

##  Architektura i Zakres Projektu

Projekt został podzielony na trzy główne moduły dopasowane do specyfiki przetwarzanych danych i zadań:

```
├──  Zadanie 1: Klasyfikacja Obrazów (CIFAR-100)
│   ├── Selektancja i filtrowanie klas pojazdów
│   ├── Normalizacja wartości pikseli [0, 1]
│   └── Splotowe Sieci Neuronowe (CNN)
│
├──  Zadanie 2: Klasyfikacja Tabelaryczna (Wine Quality)
│   ├── Binarizacja zmiennej objaśnianej (Jakość wina)
│   ├── Skalowanie cech (StandardScaler / Z-score)
│   └── Klasyfikacja binarna z użyciem sieci neuronowej
│
└──  Zadanie 3: Regresja Danych Tabelarycznych (California Housing)
    ├── Predykcja mediany cen nieruchomości
    ├── Standaryzacja cech geograficznych i demograficznych
    └── Głęboka sieć neuronowa do zadań regresyjnych (MSE/MAE)
```

---

##  Technologia i Biblioteki

* **Język programowania:** Python 3.12+
* **Deep Learning:** TensorFlow / Keras (`layers`, `models`, `callbacks`, `regularizers`)
* **Uczenie Maszynowe:** Scikit-Learn (`fetch_california_housing`, `train_test_split`, `StandardScaler`, `metrics`)
* **Analiza i Obliczanie:** NumPy, Pandas
* **Wizualizacja Danych:** Matplotlib, Seaborn

---

##  Opis Zbiorów Danych i Preprocessing

### Zadanie 1: Podzbiór CIFAR-100 (Obrazy)
Wyselekcjonowany podzbiór klas dotyczących pojazdów z wyjściowego zbioru CIFAR-100 w celu skrócenia czasu uczenia i skupienia modelu na konkretnej dziedzinie.

* **Wybrane klasy:**
  1.  `Bicycle` (Rower)
  2.  `Bus` (Autobus)
  3.  `Motorcycle` (Motocykl)
  4.  `Pickup` (Ciężarówka typu pickup)
  5.  `Train` (Pociąg)
* **Kroki Preprocessingu:**
  * Ekstrakcja tylko próbek należących do zdefiniowanych 5 klas.
  * Normalizacja macierzy pikseli: $X_{norm} = \frac{X}{255.0}$.
  * Przemapowanie etykiet rzadkich z CIFAR-100 na ciągły indeks od `0` do `4`.

### Zadanie 2: Wine Quality – Red Wine (Dane Tabelaryczne UCI)
Zbiór danych z UCI Machine Learning Repository zawierający właściwości fizykochemiczne czerwonego wina.

* **Cel:** Klasyfikacja binarna jakości wina na podstawie parametrów chemicznych (kwasowość, zawartość cukru, pH, zawartość alkoholu itp.).
* **Kroki Preprocessingu:**
  * **Binarizacja:** Podział próbek wg kryterium jakości:
    * `1` (Dobre wino): Ocena $\ge 6$
    * `0` (Słabe wino): Ocena $< 6$
  * **Podział danych:** 80% zbiór treningowy, 20% zbiór testowy.
  * **Standaryzacja:** Zastosowanie `StandardScaler` do przeskalowania cech do średniej $0$ i odchylenia standardowego $1$.

### Zadanie 3: California Housing (Regresja)
Klasyczny zbiór danych służący do przewidywania ciągłej wartości mediany cen domów w blokach urbanistycznych w Kalifornii.

* **Cel:** Predykcja wartości zmiennej objaśnianej `MedHouseVal` (mediana wartości domu).
* **Cechy:** Dochód mieszkańców (`MedInc`), wiek nieruchomości (`HouseAge`), średnia liczba izb (`AveRooms`), położenie geograficzne (`Latitude`, `Longitude`) i inne.
* **Kroki Preprocessingu:**
  * Pobranie danych bezpośrednio z `sklearn.datasets.fetch_california_housing`.
  * Podział na zbiór treningowy, walidacyjny i testowy.
  * Standaryzacja cech liczbowych za pomocą `StandardScaler`.
  * Budowa gęstej sieci neuronowej (MLP) z funkcjami aktywacji ReLU oraz warstwą wyjściową z pojedynczym neuronem liniowym.

---

##  Funkcje Pomocnicze i Diagnostyka

W notatniku zaimplementowano własne moduły diagnostyczne ułatwiające analizę zbieżności i skuteczności modeli:

1. **`plot_learning_curves(history, metric_name)`**
   * Wykres funkcji straty (*Loss*) oraz wybranej metryki (np. *Accuracy*, *MAE*, *MSE*) w funkcji epok dla zbiorów treningowego i walidacyjnego.
   * Umożliwia natychmiastową detekcję przeuczenia (*overfitting*) lub niedouczenia (*underfitting*).

2. **`plot_confusion_matrix(y_true, y_pred, class_names)`**
   * Wygenerowanie macierzy pomyłek w postaci czytelnej mapy ciepła (*Seaborn Heatmap*) dla zadań klasyfikacji.
   * Pozwala na dokładną analizę klas, z którymi model ma największe trudności.

---

##  Instrukcja Uruchomienia

### 1. Klonowanie repozytorium
```bash
git clone https://github.com/twoj-login/nazwa-repozytorium.git
cd nazwa-repozytorium
```

### 2. Tworzenie środowiska wirtualnego (opcjonalnie, ale zalecane)
```bash
python -m venv venv
# Na systemie Windows:
venv\Scripts\activate
# Na systemach Linux/macOS:
source venv/bin/activate
```

### 3. Instalacja zależności
```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn jupyter
```

### 4. Uruchomienie Notatnika Jupyter
```bash
jupyter notebook
```
Otwórz plik `.ipynb` w przeglądarce i uruchom poszczególne komórki.
