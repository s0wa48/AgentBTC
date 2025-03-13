"""
Moduł analizy rynku
Zawiera funkcje do analizy danych, predykcji i generowania sygnałów
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import ccxt
import logging
import requests
import random
from datetime import datetime
import ta

import config

logger = logging.getLogger("analyzer")

class MarketAnalyzer:
    """
    Klasa do analizy rynku kryptowalut:
    - Pobieranie danych
    - Analiza techniczna
    - Analiza sentymentu
    - Predykcja cen modelem LSTM
    """
    
    def __init__(self):
        """Inicjalizacja analizatora rynku."""
        logger.info("Inicjalizacja analizatora rynku")
        self.exchange = self._initialize_exchange()
        self.model = None
        self.scaler_price = MinMaxScaler(feature_range=(0, 1))
        self.scaler_features = MinMaxScaler(feature_range=(0, 1))
    
    def _initialize_exchange(self):
        """Inicjalizacja połączenia z giełdą."""
        try:
            exchange_id = config.EXCHANGE["name"]
            exchange_class = getattr(ccxt, exchange_id)
            
            exchange = exchange_class({
                'apiKey': config.EXCHANGE["api_key"],
                'secret': config.EXCHANGE["api_secret"],
                'enableRateLimit': True,
                'options': {'defaultType': 'future' if config.EXCHANGE["use_futures"] else 'spot'}
            })
            
            # Testnet
            if config.EXCHANGE["testnet"]:
                if hasattr(exchange, 'set_sandbox_mode'):
                    exchange.set_sandbox_mode(True)
            
            logger.info(f"Połączono z giełdą {exchange_id}")
            return exchange
        
        except Exception as e:
            logger.error(f"Błąd podczas inicjalizacji giełdy: {e}")
            return None
    
    def fetch_market_data(self, symbol=None, timeframe=None, limit=1000, since=None):
        """
        Pobieranie danych historycznych z giełdy.
        
        Args:
            symbol (str): Symbol instrumentu
            timeframe (str): Interwał czasowy
            limit (int): Liczba świec do pobrania
            since (int): Timestamp początkowy w ms
            
        Returns:
            DataFrame: DataFrame z danymi historycznymi
        """
        try:
            if symbol is None:
                symbol = config.SYMBOL
            if timeframe is None:
                timeframe = config.TIMEFRAME
            
            if self.exchange is None:
                raise Exception("Brak połączenia z giełdą")
            
            logger.info(f"Pobieranie {limit} świec dla {symbol} na interwale {timeframe}")
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Dodanie wskaźników technicznych
            df = self.add_technical_indicators(df)
            
            logger.info(f"Pobrano {len(df)} świec historycznych")
            return df
        
        except Exception as e:
            logger.error(f"Błąd podczas pobierania danych: {e}")
            return pd.DataFrame()
    
    def add_technical_indicators(self, df):
        """
        Dodanie wskaźników technicznych do danych.
        
        Args:
            df (DataFrame): DataFrame z danymi OHLCV
            
        Returns:
            DataFrame: DataFrame z dodanymi wskaźnikami
        """
        try:
            if df.empty:
                return df
            
            # Upewnienie się, że dane są typu float64 do obliczeń
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = df[col].astype('float64')
            
            # RSI
            if len(df) >= 14:
                df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            
            # MACD
            if len(df) >= 26:
                macd_indicator = ta.trend.MACD(
                    df['close'], 
                    window_slow=26, 
                    window_fast=12, 
                    window_sign=9
                )
                df['macd'] = macd_indicator.macd()
                df['macdsignal'] = macd_indicator.macd_signal()
                df['macdhist'] = macd_indicator.macd_diff()
            
            # Bollinger Bands
            if len(df) >= 20:
                bollinger = ta.volatility.BollingerBands(
                    df['close'], 
                    window=20, 
                    window_dev=2
                )
                df['bb_upper'] = bollinger.bollinger_hband()
                df['bb_middle'] = bollinger.bollinger_mavg()
                df['bb_lower'] = bollinger.bollinger_lband()
            
            # EMA
            if len(df) >= 9:
                df['ema9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            if len(df) >= 20:
                df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            if len(df) >= 50:
                df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            if len(df) >= 200:
                df['ema200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
            
            # ATR - Average True Range
            if len(df) >= 14:
                df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            
            # Stochastic Oscillator
            if len(df) >= 14:
                stoch = ta.momentum.StochasticOscillator(
                    df['high'], 
                    df['low'], 
                    df['close'], 
                    window=14, 
                    smooth_window=3
                )
                df['stoch_k'] = stoch.stoch()
                df['stoch_d'] = stoch.stoch_signal()
            
            # ADX - Average Directional Index
            if len(df) >= 14:
                adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
                df['adx'] = adx.adx()
                df['adx_pos'] = adx.adx_pos()
                df['adx_neg'] = adx.adx_neg()
            
            # CCI - Commodity Channel Index
            if len(df) >= 20:
                df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
            
            # Williams %R
            if len(df) >= 14:
                df['willr'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()
            
            # OBV - On Balance Volume
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            
            # Procentowa zmiana ceny
            df['pct_change'] = df['close'].pct_change()
            
            # Dodatkowe cechy pochodne
            if all(col in df.columns for col in ['close', 'bb_upper', 'bb_lower']):
                # Pozycja ceny w kanale Bollinger Bands (0-1)
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            if all(col in df.columns for col in ['close', 'ema20']):
                # Odległość ceny od EMA20
                df['ema20_distance'] = (df['close'] - df['ema20']) / df['ema20']
            
            return df
        
        except Exception as e:
            logger.error(f"Błąd podczas dodawania wskaźników technicznych: {e}")
            return df
    
    def load_model(self, model_path=None):
        """
        Wczytanie modelu LSTM.
        
        Args:
            model_path (str): Ścieżka do modelu
            
        Returns:
            bool: True jeśli wczytano pomyślnie
        """
        try:
            if model_path is None:
                # Szukaj najnowszego modelu
                models = [f for f in os.listdir(config.MODELS_DIR) if f.endswith('.h5')]
                
                if not models:
                    logger.warning("Nie znaleziono żadnego modelu")
                    return False
                
                # Wybierz najnowszy model
                model_path = os.path.join(config.MODELS_DIR, 
                                         max(models, key=lambda x: os.path.getmtime(os.path.join(config.MODELS_DIR, x))))
            
            self.model = load_model(model_path)
            
            # Wczytanie skaler'ów jeśli istnieją
            try:
                import pickle
                scaler_price_path = model_path.replace('.h5', '_price_scaler.pkl')
                scaler_features_path = model_path.replace('.h5', '_features_scaler.pkl')
                
                if os.path.exists(scaler_price_path):
                    with open(scaler_price_path, 'rb') as f:
                        self.scaler_price = pickle.load(f)
                
                if os.path.exists(scaler_features_path):
                    with open(scaler_features_path, 'rb') as f:
                        self.scaler_features = pickle.load(f)
            except Exception as e:
                logger.warning(f"Nie udało się wczytać scalerów: {e}")
            
            logger.info(f"Model wczytany: {model_path}")
            return True
        
        except Exception as e:
            logger.error(f"Błąd podczas wczytywania modelu: {e}")
            return False
    
    def get_feature_list(self):
        """Zwraca listę cech używanych przez model."""
        base_features = ['open', 'high', 'low', 'close', 'volume']
        
        # Dodatkowe cechy zależne od konfiguracji
        technical_features = []
        if "rsi" in config.TECHNICAL_INDICATORS:
            technical_features.append('rsi')
        if "macd" in config.TECHNICAL_INDICATORS:
            technical_features.extend(['macd', 'macdsignal', 'macdhist'])
        if "bollinger_bands" in config.TECHNICAL_INDICATORS:
            technical_features.extend(['bb_upper', 'bb_middle', 'bb_lower'])
        if "ema" in config.TECHNICAL_INDICATORS:
            technical_features.extend(['ema9', 'ema20', 'ema50', 'ema200'])
        if "atr" in config.TECHNICAL_INDICATORS:
            technical_features.append('atr')
        
        # Jeśli nie ma żadnych wskaźników w konfiguracji, dodaj domyślne
        if not technical_features:
            technical_features = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'ema20']
        
        return base_features + technical_features
    
    def preprocess_data(self, df):
        """
        Przetwarzanie danych dla modelu LSTM.
        
        Args:
            df (DataFrame): DataFrame z danymi historycznymi
            
        Returns:
            tuple: X, y dane dla modelu
        """
        try:
            # Sprawdzenie czy DataFrame ma wystarczającą liczbę wierszy
            if len(df) < config.MODEL["sequence_length"]:
                logger.error(f"Za mało danych do przetworzenia: {len(df)} < {config.MODEL['sequence_length']}")
                return None, None
            
            # Usunięcie wierszy z NaN
            df = df.dropna()
            
            # Lista cech
            feature_columns = self.get_feature_list()
            
            # Separujemy cenę zamknięcia do predykcji
            price_data = df['close'].values.reshape(-1, 1)
            scaled_price = self.scaler_price.fit_transform(price_data)
            
            # Sprawdzenie czy wszystkie kolumny istnieją
            missing_columns = [col for col in feature_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Brakujące kolumny: {missing_columns}")
                feature_columns = [col for col in feature_columns if col in df.columns]
            
            # Skalujemy cechy
            feature_data = df[feature_columns].values
            scaled_features = self.scaler_features.fit_transform(feature_data)
            
            X, y = [], []
            sequence_length = config.MODEL["sequence_length"]
            
            for i in range(len(scaled_features) - sequence_length):
                X.append(scaled_features[i:i + sequence_length])
                y.append(scaled_price[i + sequence_length])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania danych: {e}")
            return None, None
    
    def predict_price_movement(self, df):
        """
        Przewidywanie ruchu ceny na podstawie modelu LSTM.
        
        Args:
            df (DataFrame): DataFrame z danymi historycznymi
            
        Returns:
            dict: Słownik z predykcją
        """
        try:
            # Sprawdzenie czy model jest wczytany
            if self.model is None:
                if not self.load_model():
                    logger.error("Brak modelu do predykcji")
                    return None
            
            # Aktualna cena
            current_price = df['close'].iloc[-1]
            
            # Przetworzenie danych dla modelu
            X, _ = self.preprocess_data(df)
            
            if X is None:
                logger.error("Błąd podczas przetwarzania danych dla predykcji")
                return None
            
            # Predykcja
            X_pred = X[-1:] # Ostatnia sekwencja
            scaled_prediction = self.model.predict(X_pred)
            
            # Odwrócenie skalowania
            predicted_price = self.scaler_price.inverse_transform(scaled_prediction)[0][0]
            
            # Obliczanie procentowej zmiany
            percent_change = ((predicted_price - current_price) / current_price) * 100
            
            # Określenie kierunku ruchu
            threshold = config.MODEL["prediction_threshold"]
            if abs(percent_change) < threshold:
                direction = "FLAT"
            elif percent_change > 0:
                direction = "UP"
            else:
                direction = "DOWN"
            
            # Przygotowanie wyniku
            prediction = {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'percent_change': percent_change,
                'direction': direction
            }
            
            logger.info(f"Predykcja: Aktualna cena={current_price:.2f}, "
                       f"Przewidywana={predicted_price:.2f}, Zmiana={percent_change:.2f}%, "
                       f"Kierunek={direction}")
            
            return prediction
        
        except Exception as e:
            logger.error(f"Błąd podczas predykcji ceny: {e}")
            return None
    
    def generate_ta_signal(self, df):
        """
        Generowanie sygnału na podstawie analizy technicznej.
        
        Args:
            df (DataFrame): DataFrame z danymi i wskaźnikami
            
        Returns:
            str: Sygnał ('LONG', 'SHORT', 'FLAT')
        """
        try:
            if df.empty or len(df) < 14:  # Minimalna liczba punktów
                return "FLAT"
            
            # Zliczanie głosów dla każdego sygnału
            votes = {"LONG": 0, "SHORT": 0, "FLAT": 0}
            
            # 1. RSI
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                
                if rsi < 30:
                    votes["LONG"] += 1
                elif rsi > 70:
                    votes["SHORT"] += 1
                else:
                    votes["FLAT"] += 1
            
            # 2. MACD
            if all(col in df.columns for col in ['macd', 'macdsignal']):
                if len(df) >= 2:
                    macd_current = df['macd'].iloc[-1]
                    macd_prev = df['macd'].iloc[-2]
                    signal_current = df['macdsignal'].iloc[-1]
                    signal_prev = df['macdsignal'].iloc[-2]
                    
                    # Przecięcie w górę (byczy)
                    if macd_prev < signal_prev and macd_current > signal_current:
                        votes["LONG"] += 1
                    # Przecięcie w dół (niedźwiedzi)
                    elif macd_prev > signal_prev and macd_current < signal_current:
                        votes["SHORT"] += 1
                    else:
                        # Trend na podstawie wartości MACD
                        if macd_current > 0:
                            votes["LONG"] += 0.5
                        elif macd_current < 0:
                            votes["SHORT"] += 0.5
                        else:
                            votes["FLAT"] += 0.5
            
            # 3. Bollinger Bands
            if all(col in df.columns for col in ['bb_upper', 'bb_lower']):
                current_price = df['close'].iloc[-1]
                upper_band = df['bb_upper'].iloc[-1]
                lower_band = df['bb_lower'].iloc[-1]
                
                if current_price > upper_band:
                    votes["SHORT"] += 1  # Potencjalne przewartościowanie
                elif current_price < lower_band:
                    votes["LONG"] += 1   # Potencjalne niedowartościowanie
                else:
                    # Miejsce w kanale Bollingera
                    width = upper_band - lower_band
                    if width > 0:
                        position = (current_price - lower_band) / width
                        if position < 0.3:
                            votes["LONG"] += 0.5
                        elif position > 0.7:
                            votes["SHORT"] += 0.5
                        else:
                            votes["FLAT"] += 0.5
            
            # 4. EMA
            if all(col in df.columns for col in ['ema9', 'ema20']):
                ema9 = df['ema9'].iloc[-1]
                ema20 = df['ema20'].iloc[-1]
                price = df['close'].iloc[-1]
                
                # Trend wzrostowy: EMA9 > EMA20 i cena > EMA9
                if ema9 > ema20 and price > ema9:
                    votes["LONG"] += 1
                # Trend spadkowy: EMA9 < EMA20 i cena < EMA9
                elif ema9 < ema20 and price < ema9:
                    votes["SHORT"] += 1
                else:
                    votes["FLAT"] += 0.5
            
            # 5. Stochastic Oscillator
            if all(col in df.columns for col in ['stoch_k', 'stoch_d']):
                stoch_k = df['stoch_k'].iloc[-1]
                stoch_d = df['stoch_d'].iloc[-1]
                
                if stoch_k < 20 and stoch_d < 20:
                    votes["LONG"] += 1  # Wyprzedanie
                elif stoch_k > 80 and stoch_d > 80:
                    votes["SHORT"] += 1  # Wykupienie
                elif stoch_k > stoch_d and stoch_k < 80 and stoch_d < 80:
                    votes["LONG"] += 0.5  # Przecięcie w górę w normalnym zakresie
                elif stoch_k < stoch_d and stoch_k > 20 and stoch_d > 20:
                    votes["SHORT"] += 0.5  # Przecięcie w dół w normalnym zakresie
            
            # 6. ADX
            if 'adx' in df.columns and 'adx_pos' in df.columns and 'adx_neg' in df.columns:
                adx = df['adx'].iloc[-1]
                adx_pos = df['adx_pos'].iloc[-1]
                adx_neg = df['adx_neg'].iloc[-1]
                
                if adx > 25:  # Silny trend
                    if adx_pos > adx_neg:
                        votes["LONG"] += 1  # Silny trend wzrostowy
                    else:
                        votes["SHORT"] += 1  # Silny trend spadkowy
                else:
                    votes["FLAT"] += 0.5  # Słaby trend
            
            # 7. CCI
            if 'cci' in df.columns:
                cci = df['cci'].iloc[-1]
                
                if cci < -100:
                    votes["LONG"] += 0.5  # Wyprzedanie
                elif cci > 100:
                    votes["SHORT"] += 0.5  # Wykupienie
            
            # Wybór sygnału z największą liczbą głosów
            signal = max(votes.items(), key=lambda x: x[1])[0]
            
            logger.info(f"Sygnał TA: {signal} (głosy: LONG={votes['LONG']}, "
                       f"SHORT={votes['SHORT']}, FLAT={votes['FLAT']})")
            
            return signal
        
        except Exception as e:
            logger.error(f"Błąd podczas generowania sygnału TA: {e}")
            return "FLAT"  # W przypadku błędu zwracamy neutralny sygnał
    
    def analyze_market_sentiment(self):
        """
        Analiza sentymentu rynkowego (uproszczona).
        
        Returns:
            float: Wartość sentymentu w zakresie [-1, 1]
        """
        try:
            # W tej uproszczonej wersji używamy tylko Fear & Greed Index
            # i losowego szumu dla symulacji innych źródeł
            
            # Fear & Greed Index
            try:
                url = "https://api.alternative.me/fng/"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and len(data['data']) > 0:
                        fear_greed_value = int(data['data'][0]['value'])
                        # Normalizacja do [-1, 1]
                        normalized_fear_greed = (fear_greed_value - 50) / 50
                        logger.info(f"Fear & Greed Index: {fear_greed_value} -> {normalized_fear_greed:.2f}")
                        
                        # Dodanie niewielkiego losowego szumu
                        noise = random.uniform(-0.1, 0.1)
                        sentiment = normalized_fear_greed + noise
                        sentiment = max(min(sentiment, 1.0), -1.0)  # Ograniczenie do [-1, 1]
                        
                        return sentiment
            except Exception as e:
                logger.warning(f"Błąd podczas pobierania Fear & Greed Index: {e}")
            
            # 2. Próba pobierania danych o rynku BTC
            try:
                # Pobranie aktualnej ceny i 24h change z CoinGecko
                cg_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"
                response = requests.get(cg_url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'bitcoin' in data and 'usd_24h_change' in data['bitcoin']:
                        change_24h = data['bitcoin']['usd_24h_change']
                        # Normalizacja 24h change do [-1, 1]
                        # Zakładamy, że ±20% to ekstremalne przypadki
                        normalized_change = max(min(change_24h / 20, 1.0), -1.0)
                        logger.info(f"BTC 24h change: {change_24h:.2f}% -> {normalized_change:.2f}")
                        
                        # W przypadku braku F&G Index, zwróć zmianę 24h jako sentyment
                        return normalized_change
            except Exception as e:
                logger.warning(f"Błąd podczas pobierania danych z CoinGecko: {e}")
            
            # Jeśli nie udało się pobrać żadnych danych, zwracamy losową wartość
            return random.uniform(-0.3, 0.3)  # Losowy sentyment z przewagą neutralnego
            
        except Exception as e:
            logger.error(f"Błąd podczas analizy sentymentu: {e}")
            return 0.0  # Neutralny sentyment w przypadku błędu
