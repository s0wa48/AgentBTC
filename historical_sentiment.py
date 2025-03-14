"""
Moduł do odtwarzania historycznego sentymentu rynkowego
"""

import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import csv
import json

import config

logger = logging.getLogger("historical_sentiment")

class HistoricalSentimentAnalyzer:
    """
    Klasa do odtwarzania historycznego sentymentu rynkowego na podstawie dostępnych danych historycznych.
    """
    
    def __init__(self):
        """Inicjalizacja analizatora historycznego sentymentu."""
        # Utworzenie katalogu na dane historyczne, jeśli nie istnieje
        self.historical_data_dir = os.path.join(config.DATA_DIR, "historical_sentiment")
        os.makedirs(self.historical_data_dir, exist_ok=True)
        
        # Ścieżki do plików z historycznymi danymi
        self.fear_greed_history_file = os.path.join(self.historical_data_dir, "fear_greed_history.csv")
        self.funding_rate_history_file = os.path.join(self.historical_data_dir, "funding_rate_history.csv")
        
        # Wczytanie historycznych danych
        self.fear_greed_history = self._load_fear_greed_history()
    
    def _load_fear_greed_history(self):
        """
        Wczytuje historyczne dane Fear & Greed Index.
        Jeśli dane nie istnieją, pobiera je z API i zapisuje.
        
        Returns:
            pd.DataFrame: DataFrame z historycznymi danymi
        """
        # Sprawdź czy plik istnieje
        if os.path.exists(self.fear_greed_history_file):
            try:
                # Wczytaj dane z pliku
                df = pd.read_csv(self.fear_greed_history_file, parse_dates=['timestamp'])
                logger.info(f"Wczytano {len(df)} historycznych wartości Fear & Greed Index")
                return df
            except Exception as e:
                logger.warning(f"Błąd podczas wczytywania historycznego Fear & Greed Index: {e}")
        
        # Jeśli plik nie istnieje lub wystąpił błąd, pobierz dane z API
        logger.info("Pobieranie historycznych danych Fear & Greed Index...")
        try:
            # Pobierz historyczne dane z API
            url = "https://api.alternative.me/fng/?limit=0"  # Limit 0 zwraca wszystkie dane
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    # Konwersja danych do DataFrame
                    records = []
                    for item in data['data']:
                        timestamp = datetime.fromtimestamp(int(item['timestamp']))
                        value = int(item['value'])
                        classification = item['value_classification']
                        # Normalizacja do [-1, 1]
                        normalized_value = (value - 50) / 50
                        
                        records.append({
                            'timestamp': timestamp,
                            'value': value,
                            'classification': classification,
                            'normalized_value': normalized_value
                        })
                    
                    df = pd.DataFrame(records)
                    
                    # Zapisz dane do pliku
                    df.to_csv(self.fear_greed_history_file, index=False)
                    
                    logger.info(f"Pobrano i zapisano {len(df)} historycznych wartości Fear & Greed Index")
                    return df
                else:
                    logger.warning("Nieprawidłowa struktura danych z API Fear & Greed")
            else:
                logger.warning(f"Błąd HTTP podczas pobierania historycznego Fear & Greed Index: {response.status_code}")
        except Exception as e:
            logger.warning(f"Błąd podczas pobierania historycznego Fear & Greed Index: {e}")
        
        # Jeśli nie udało się pobrać danych, zwróć pusty DataFrame
        return pd.DataFrame(columns=['timestamp', 'value', 'classification', 'normalized_value'])
    
    def _get_funding_rate_for_date(self, date, exchange="binance", symbol="BTCUSDT"):
        """
        Odtworzenie historycznego funding rate dla danej daty.
        Używa prostego modelu bazującego na historycznym ruchu ceny.
        
        Args:
            date (datetime): Data, dla której chcemy odtworzyć funding rate
            exchange (str): Nazwa giełdy
            symbol (str): Symbol kontraktu futures
            
        Returns:
            float: Odtworzony funding rate w zakresie [-1, 1]
        """
        try:
            # Ten fragment kodu symuluje funding rate na podstawie ruchu ceny
            # W rzeczywistości należałoby użyć historycznych danych funding rate
            # dostępnych z archiwów giełd lub własnej bazy danych
            
            # Pobierz historyczne dane OHLCV dla danej daty
            # Tutaj użyjemy danych z MarketAnalyzer, jeśli są dostępne
            
            # Uproszczona symulacja: funding rate zazwyczaj jest skorelowany
            # z krótkoterminowym ruchem ceny - wysokie finansowanie w trendzie byczym
            # i niskie/ujemne w trendzie niedźwiedzim
            
            # Generujemy wartość na podstawie daty, aby była deterministyczna
            # dla tych samych dat w różnych uruchomieniach
            seed = int(date.timestamp()) % 10000
            np.random.seed(seed)
            
            # Deterministic but still varying value for backtesting
            day_of_year = date.timetuple().tm_yday
            momentum_factor = np.sin(day_of_year / 30 * np.pi) * 0.05  # Cycling pattern
            
            # Funding rate is typically small, between -0.1% and 0.1%
            funding_rate = momentum_factor + np.random.normal(0, 0.02)
            funding_rate = np.clip(funding_rate, -0.075, 0.075)
            
            # Normalizacja do [-1, 1]
            normalized_funding_rate = funding_rate * 13.33  # Scale up to use full range
            normalized_funding_rate = np.clip(normalized_funding_rate, -1.0, 1.0)
            
            return normalized_funding_rate
            
        except Exception as e:
            logger.warning(f"Błąd podczas odtwarzania historycznego funding rate: {e}")
            return 0.0  # Neutralna wartość w przypadku błędu
    
    def _get_long_short_ratio_for_date(self, date, exchange="binance", symbol="BTCUSDT"):
        """
        Odtworzenie historycznego long/short ratio dla danej daty.
        
        Args:
            date (datetime): Data, dla której chcemy odtworzyć long/short ratio
            exchange (str): Nazwa giełdy
            symbol (str): Symbol kontraktu futures
            
        Returns:
            float: Odtworzony long/short ratio w zakresie [0, 1]
        """
        try:
            # Podobnie jak w przypadku funding rate, symulujemy long/short ratio
            # W rzeczywistości należałoby użyć historycznych danych
            
            # Generujemy wartość na podstawie daty, aby była deterministyczna
            seed = int(date.timestamp()) % 10000
            np.random.seed(seed)
            
            # Deterministic but varying value
            day_of_year = date.timetuple().tm_yday
            week_of_year = date.isocalendar()[1]
            
            # Base ratio around 0.5 (equal longs and shorts)
            # with some patterns based on time of year
            ratio_base = 0.5 + 0.1 * np.sin(week_of_year / 26 * np.pi)
            
            # Add some noise
            ratio = ratio_base + np.random.normal(0, 0.05)
            
            # Ensure ratio is in [0, 1] range
            ratio = np.clip(ratio, 0.0, 1.0)
            
            return ratio
            
        except Exception as e:
            logger.warning(f"Błąd podczas odtwarzania historycznego long/short ratio: {e}")
            return 0.5  # Neutralna wartość w przypadku błędu
    
    def _get_exchange_flows_for_date(self, date, timeframe="1d"):
        """
        Odtworzenie historycznych przepływów giełdowych dla danej daty.
        
        Args:
            date (datetime): Data, dla której chcemy odtworzyć przepływy
            timeframe (str): Przedział czasowy
            
        Returns:
            float: Odtworzony wskaźnik przepływów w zakresie [-1, 1]
        """
        try:
            # Symulacja przepływów giełdowych
            # W rzeczywistości należałoby użyć historycznych danych
            
            # Generujemy wartość na podstawie daty, aby była deterministyczna
            seed = int(date.timestamp()) % 10000
            np.random.seed(seed)
            
            # Deterministic but varying value
            day_of_year = date.timetuple().tm_yday
            month = date.month
            
            # Base flow value with seasonal patterns
            # Typically more outflows from exchanges during bull markets (positive value)
            # and more inflows during bear markets (negative value)
            flow_base = 0.2 * np.sin((day_of_year + 30) / 120 * np.pi)
            
            # Add correlation with funding rate
            # When funding rate is high (bullish), there are often exchange outflows
            funding_rate = self._get_funding_rate_for_date(date)
            flow_component = -funding_rate * 0.3  # Negative correlation
            
            # Add some noise
            flow = flow_base + flow_component + np.random.normal(0, 0.1)
            
            # Ensure flow is in [-1, 1] range
            normalized_flow = np.clip(flow, -1.0, 1.0)
            
            return normalized_flow
            
        except Exception as e:
            logger.warning(f"Błąd podczas odtwarzania historycznych przepływów giełdowych: {e}")
            return 0.0  # Neutralna wartość w przypadku błędu
    
    def get_historical_fear_greed(self, date):
        """
        Pobiera historyczną wartość Fear & Greed Index dla podanej daty.
        
        Args:
            date (datetime): Data, dla której chcemy pobrać Fear & Greed Index
            
        Returns:
            float: Znormalizowana wartość Fear & Greed Index w zakresie [-1, 1]
        """
        # Jeśli nie mamy danych, zwróć neutralną wartość
        if self.fear_greed_history.empty:
            return 0.0
        
        try:
            # Znajdź najbliższą datę w danych
            date_str = date.strftime('%Y-%m-%d')
            closest_entry = self.fear_greed_history[self.fear_greed_history['timestamp'].dt.strftime('%Y-%m-%d') == date_str]
            
            if not closest_entry.empty:
                normalized_value = closest_entry.iloc[0]['normalized_value']
                return normalized_value
            
            # Jeśli nie znaleziono dokładnej daty, znajdź najbliższą
            self.fear_greed_history['date_diff'] = abs((self.fear_greed_history['timestamp'] - date).dt.total_seconds())
            closest_entry = self.fear_greed_history.loc[self.fear_greed_history['date_diff'].idxmin()]
            self.fear_greed_history = self.fear_greed_history.drop('date_diff', axis=1)
            
            return closest_entry['normalized_value']
            
        except Exception as e:
            logger.warning(f"Błąd podczas pobierania historycznego Fear & Greed Index: {e}")
            return 0.0  # Neutralna wartość w przypadku błędu
    
    def get_integrated_historical_sentiment(self, date):
        """
        Oblicza zintegrowany wskaźnik sentymentu dla podanej daty historycznej.
        
        Args:
            date (datetime): Data, dla której chcemy obliczyć sentyment
            
        Returns:
            float: Zintegrowany wskaźnik sentymentu w zakresie [-1, 1]
        """
        # Pobieramy wszystkie wskaźniki dla danej daty
        fear_greed = self.get_historical_fear_greed(date)
        funding_rate = self._get_funding_rate_for_date(date)
        long_short_ratio = self._get_long_short_ratio_for_date(date)
        exchange_flows = self._get_exchange_flows_for_date(date)
        
        # Wagi poszczególnych wskaźników
        weights = {
            'fear_greed': 0.4,
            'funding_rate': 0.3,
            'long_short_ratio': 0.2,
            'exchange_flows': 0.1
        }
        
        # Przeciwność sentymentu dla funding rate (wysoki funding rate jest zwykle sygnałem przeciwnym)
        inverse_funding_rate = -funding_rate
        
        # Przeciwność long/short ratio (odchylenie od równowagi 0.5)
        inverse_long_short = 2 * (0.5 - long_short_ratio)
        
        # Obliczenie ważonego sentymentu
        integrated_sentiment = (
            fear_greed * weights['fear_greed'] +
            inverse_funding_rate * weights['funding_rate'] +
            inverse_long_short * weights['long_short_ratio'] +
            exchange_flows * weights['exchange_flows']
        )
        
        # Ograniczenie do zakresu [-1, 1]
        integrated_sentiment = np.clip(integrated_sentiment, -1.0, 1.0)
        
        return integrated_sentiment
