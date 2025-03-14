"""
Konfiguracja dla Bitcoin Trading Bot
Zawiera wszystkie parametry konfiguracyjne systemu
"""

import os

# Ścieżki
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
HISTORICAL_DATA_DIR = os.path.join(DATA_DIR, "historical")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved_models")

# Utworzenie katalogów, jeśli nie istnieją
for dir_path in [DATA_DIR, HISTORICAL_DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Parametry ogólne
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"  # 1h, 4h, 1d, etc.
DEBUG_MODE = False

# Giełda
EXCHANGE = {
    "name": "binance",
    "api_key": "YOUR_API_KEY",
    "api_secret": "YOUR_API_SECRET",
    "use_futures": True,
    "testnet": True  # Ustaw na False dla produkcji
}

# Webhook do systemu transakcyjnego
WEBHOOK_URL = "http://51.83.255.214:80/api/receive-signal"

# Konfiguracja powiadomień email
EMAIL_NOTIFICATION = {
    "sender_email": "msmaciejsowinski@gmail.com",  # Email nadawcy (wymaga konfiguracji)
    "sender_password": "fgip iuls zpwm avzs",  # Hasło aplikacyjne (dla Gmail)
    "smtp_server": "smtp.gmail.com",  # Serwer SMTP
    "smtp_port": 587,  # Port SMTP
    "recipient_email": "msmaciejsowinski@gmail.com",  # Email odbiorcy
    "send_on_position_change": True,  # Wysyłaj powiadomienia przy zmianie pozycji
}

# Parametry modelu LSTM 


MODEL = {
    "sequence_length": 60,
    "batch_size": 32,
    "epochs": 50,
    "validation_split": 0.2,
    "lstm_units": [100, 50],
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "prediction_threshold": 0.5  # Próg zmiany procentowej dla sygnału
}

# Wskaźniki techniczne
TECHNICAL_INDICATORS = [
    "rsi", "macd", "bollinger_bands", "ema", "atr"
]

SENTIMENT_ANALYSIS = {
    "enabled": True,                   # Czy używać rozszerzonej analizy sentymentu
    "use_funding_rate": True,          # Czy uwzględniać funding rate
    "use_long_short_ratio": True,      # Czy uwzględniać stosunek long/short
    "use_exchange_flows": True,        # Czy uwzględniać przepływy giełdowe
    "weights": {                       # Wagi poszczególnych wskaźników
        "fear_greed": 0.4,             # Waga dla Fear & Greed Index
        "funding_rate": 0.3,           # Waga dla funding rate
        "long_short_ratio": 0.2,       # Waga dla long/short ratio
        "exchange_flows": 0.1          # Waga dla przepływów giełdowych
    },
    "exchanges": {                     # Konfiguracja dla różnych giełd
        "primary": "binance",          # Główna giełda do pobierania danych
        "secondary": "bybit"           # Zapasowa giełda
    },
    "api_keys": {                      # Klucze API do różnych serwisów
        "glassnode": "",               # Klucz API do Glassnode (opcjonalny)
        "cryptoquant": ""              # Klucz API do CryptoQuant (opcjonalny)
    },
    "cache_expiry": {                  # Czasy ważności cache (w minutach)
        "fear_greed": 60,              # Fear & Greed Index
        "funding_rate": 30,            # Funding rate
        "long_short_ratio": 15,        # Long/Short ratio
        "exchange_flows": 60           # Przepływy giełdowe
    }
}



# Parametry generowania sygnałów
SIGNALS = {
    "model_weight": 0.9,        # Waga modelu LSTM
    "tech_analysis_weight": 0.1, # Waga analizy technicznej
    "sentiment_weight": 0.3,     # Waga sentymentu
    "long_threshold": 0.3,       # Próg dla sygnału LONG
    "short_threshold": -0.3,     # Próg dla sygnału SHORT
}




# Zarządzanie ryzykiem
RISK = {
    "max_leverage": 0.5,         # Maksymalna dźwignia
    "risk_per_trade": 0.1,      # 1% kapitału na trade
    "stop_loss_pct": 0.02,       # 2% stop loss
    "take_profit_pct": 0.045,     # 4% take profit
    "max_open_positions": 5,
    "max_drawdown_pct": 0.1     # Maksymalne dopuszczalne obniżenie 15%
}

# Parametry backtestingu
BACKTEST = {
    "start_date": "2020-01-01",
    "end_date": "2025-03-10",
    "initial_balance": 10000
}

# Parametry tradingu na żywo
LIVE_TRADING = {
    "check_interval_minutes": 5,  # Sprawdzaj co godzinę
    "notification_email": "msmaciejsowinski@gmail.com"
}

# Proste logowanie
import logging
logging.basicConfig(
    level=logging.INFO if not DEBUG_MODE else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, "bot.log")),
        logging.StreamHandler()
    ]
)
