# Bitcoin Trading Bot

Automatyczny system tradingowy dla Bitcoin oparty na modelu LSTM, analizie technicznej i analizie sentymentu rynkowego.

## Funkcjonalności

- Predykcja cen Bitcoin za pomocą modelu LSTM
- Analiza techniczna używająca popularnych wskaźników (RSI, MACD, Bollinger Bands)
- Generowanie sygnałów LONG, SHORT i FLAT
- Zarządzanie ryzykiem i wielkością pozycji
- Backtesting na danych historycznych
- Trading na żywo przez API lub webhook

## Wymagania

- Python 3.8+
- TensorFlow 2.6+
- TA-Lib
- Pozostałe zależności w pliku `requirements.txt`

## Instalacja

1. Sklonuj repozytorium:
```
git clone https://github.com/s0wa48/AgentBTC.git
cd AgentBTC
```

2. Zainstaluj zależności:
```
pip install -r requirements.txt
```

3. Zainstaluj TA-Lib zgodnie z instrukcjami dla Twojego systemu operacyjnego.

4. Skonfiguruj bota edytując plik `config.py`:
   - Podaj swoje klucze API do giełdy
   - Dostosuj parametry handlowe i zarządzania ryzykiem

## Użycie

### Trenowanie modelu
```
python train.py --epochs 50
```

### Testowanie strategii na danych historycznych
```
python backtest.py --start_date 2023-01-01 --end_date 2023-12-31 --plot
```

### Uruchomienie handlu na żywo
Najpierw w trybie demo:
```
python run_live.py --demo --interval 60
```

Następnie w trybie rzeczywistym (używaj ostrożnie!):
```
python run_live.py --interval 60
```

## Struktura projektu

- `bot.py` - Główna klasa bota integrująca wszystkie komponenty
- `analyzer.py` - Analiza rynku, predykcje LSTM i wskaźniki techniczne
- `trader.py` - Wykonywanie transakcji i zarządzanie pozycjami
- `config.py` - Konfiguracja systemu
- `train.py` - Skrypt do trenowania modelu LSTM
- `backtest.py` - Backtesting strategii
- `run_live.py` - Uruchamianie handlu na żywo

## Uwaga

Trading kryptowalut wiąże się z ryzykiem finansowym. Ten projekt jest dostarczany "tak jak jest", bez żadnych gwarancji. Autor nie ponosi odpowiedzialności za jakiekolwiek straty poniesione w wyniku używania tego systemu.

Przed uruchomieniem handlu na żywo z prawdziwymi pieniędzmi, zaleca się dokładne przetestowanie strategii i zrozumienie działania systemu.

## Dalszy rozwój

Możliwe kierunki rozwoju systemu:
- Dodanie większej liczby wskaźników technicznych
- Implementacja bardziej zaawansowanych modeli ML
- Dodanie wsparcia dla większej liczby par tradingowych
- Udoskonalenie analizy sentymentu z większej liczby źródeł
- Interfejs użytkownika do monitorowania i kontroli
