"""
Moduł analizy sentymentu rynkowego
Zawiera funkcje do pomiaru sentymentu rynku kryptowalut
"""

import logging
import requests
import time
import random
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger("sentiment_analyzer")

class SentimentAnalyzer:
    """
    Klasa do analizy sentymentu rynkowego kryptowalut.
    Zawiera metody do pobierania różnych wskaźników sentymentu:
    - Fear & Greed Index
    - Funding rate
    - Long/Short ratio
    - Przepływy giełdowe
    """
    
    def __init__(self, api_keys=None):
        """
        Inicjalizacja analizatora sentymentu.
        
        Args:
            api_keys (dict): Słownik z kluczami API do różnych serwisów
        """
        self.api_keys = api_keys or {}
        
        # Cache dla danych, aby uniknąć zbyt częstych zapytań API
        self.cache = {}
        self.cache_expiry = {}
        self.default_cache_expiry = 300  # 5 minut domyślnie
    
    def get_fear_greed_index(self):
        """
        Pobiera Fear & Greed Index dla rynku krypto.
        
        Returns:
            float: Znormalizowana wartość indeksu w zakresie [-1, 1]
                  gdzie -1 to skrajny strach, 1 to skrajna chciwość
        """
        cache_key = "fear_greed_index"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            return self.cache[cache_key]
        
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    fear_greed_value = int(data['data'][0]['value'])
                    # Normalizacja do [-1, 1]
                    normalized_fear_greed = (fear_greed_value - 50) / 50
                    logger.info(f"Fear & Greed Index: {fear_greed_value} -> {normalized_fear_greed:.2f}")
                    
                    # Zapisanie do cache
                    self.cache[cache_key] = normalized_fear_greed
                    self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=60)  # Ważny przez godzinę
                    
                    return normalized_fear_greed
        except Exception as e:
            logger.warning(f"Błąd podczas pobierania Fear & Greed Index: {e}")
        
        # W przypadku błędu próbujemy pobrać dane z CoinGecko
        try:
            # Pobranie aktualnej ceny i 24h change z CoinGecko
            cg_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"
            response = requests.get(cg_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'bitcoin' in data and 'usd_24h_change' in data['bitcoin']:
                    change_24h = data['bitcoin']['usd_24h_change']
                    # Normalizacja 24h change do [-1, 1]
                    # Zakładamy, że ±20% to ekstremalne przypadki
                    normalized_change = max(min(change_24h / 20, 1.0), -1.0)
                    logger.info(f"BTC 24h change: {change_24h:.2f}% -> {normalized_change:.2f}")
                    
                    # Zapisanie do cache
                    self.cache[cache_key] = normalized_change
                    self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=30)  # Ważny przez 30 minut
                    
                    return normalized_change
        except Exception as e:
            logger.warning(f"Błąd podczas pobierania danych z CoinGecko: {e}")
        
        # Jeśli nie udało się pobrać żadnych danych, zwracamy losową wartość
        random_sentiment = random.uniform(-0.3, 0.3)
        logger.warning(f"Używanie losowego sentymentu: {random_sentiment:.2f}")
        return random_sentiment
    
    def get_funding_rate(self, exchange="binance", symbol="BTCUSDT"):
        """
        Pobiera funding rate z giełdy futures.
        
        Args:
            exchange (str): Nazwa giełdy (binance, bybit, ftx)
            symbol (str): Symbol kontraktu futures
            
        Returns:
            float: Znormalizowana wartość funding rate w zakresie [-1, 1]
                  gdzie wartości ujemne oznaczają presję na spadki,
                  a dodatnie presję na wzrosty
        """
        cache_key = f"funding_rate_{exchange}_{symbol}"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            return self.cache[cache_key]
        
        try:
            funding_rate = 0
            
            if exchange.lower() == "binance":
                url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'lastFundingRate' in data:
                        funding_rate = float(data['lastFundingRate'])
                        
            elif exchange.lower() == "bybit":
                url = f"https://api.bybit.com/v2/public/tickers?symbol={symbol}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'result' in data and len(data['result']) > 0:
                        funding_rate = float(data['result'][0]['funding_rate'])
                        
            # Normalizacja - typowe funding rate są w zakresie -0.1% do 0.1%
            # Wartości powyżej 0.05% lub poniżej -0.05% są już dość ekstremalne
            normalized_funding_rate = np.clip(funding_rate * 20, -1.0, 1.0)
            
            logger.info(f"Funding Rate ({exchange}, {symbol}): {funding_rate:.6f} -> {normalized_funding_rate:.2f}")
            
            # Zapisanie do cache
            self.cache[cache_key] = normalized_funding_rate
            self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=60)  # Ważny przez godzinę
            
            return normalized_funding_rate
        
        except Exception as e:
            logger.warning(f"Błąd podczas pobierania funding rate: {e}")
            return 0.0  # Neutralna wartość w przypadku błędu
    
    def get_long_short_ratio(self, exchange="binance", symbol="BTCUSDT"):
        """
        Pobiera stosunek długich do krótkich pozycji (long/short ratio).
        
        Args:
            exchange (str): Nazwa giełdy
            symbol (str): Symbol kontraktu futures
            
        Returns:
            float: Znormalizowana wartość long/short ratio w zakresie [0, 1]
                  gdzie 0 oznacza, że wszyscy są na krótko,
                  1 oznacza, że wszyscy są na długo,
                  a 0.5 to równowaga
        """
        cache_key = f"long_short_ratio_{exchange}_{symbol}"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            return self.cache[cache_key]
        
        try:
            ratio = 0.5  # Domyślnie zakładamy równowagę
            
            if exchange.lower() == "binance":
                # Binance udostępnia dane bezpośrednio
                url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=5m"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        ratio = float(data[0]['longShortRatio'])
                        # Konwersja na wartość z zakresu [0, 1]
                        ratio = ratio / (1 + ratio)
                        
            elif exchange.lower() == "bybit":
                url = f"https://api.bybit.com/v2/public/account-ratio?symbol={symbol}&period=5min"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'result' in data and len(data['result']) > 0:
                        buy_ratio = float(data['result'][0]['buy_ratio'])
                        ratio = buy_ratio / 100  # Bybit podaje % kupujących
            
            logger.info(f"Long/Short Ratio ({exchange}, {symbol}): {ratio:.2f}")
            
            # Zapisanie do cache
            self.cache[cache_key] = ratio
            self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=15)  # Ważny przez 15 minut
            
            return ratio
        
        except Exception as e:
            logger.warning(f"Błąd podczas pobierania long/short ratio: {e}")
            return 0.5  # Neutralna wartość w przypadku błędu
    
    def get_exchange_flows(self, timeframe="1d"):
        """
        Pobiera informacje o przepływach BTC do/z giełd.
        
        Args:
            timeframe (str): Przedział czasowy ("1h", "1d", "7d")
            
        Returns:
            float: Znormalizowana wartość przepływów w zakresie [-1, 1]
                  gdzie -1 oznacza duże przepływy DO giełd (negatywny sygnał),
                  a 1 oznacza duże przepływy Z giełd (pozytywny sygnał)
        """
        cache_key = f"exchange_flows_{timeframe}"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            return self.cache[cache_key]
        
        # Sprawdzenie czy mamy klucz API do Glassnode
        glassnode_api_key = self.api_keys.get('glassnode')
        if not glassnode_api_key:
            logger.warning("Brak klucza API Glassnode do analizy przepływów giełdowych")
            return 0.0  # Neutralna wartość
        
        try:
            # Pobranie danych z Glassnode
            url = f"https://api.glassnode.com/v1/metrics/transactions/transfers_volume_exchanges_net"
            params = {
                'api_key': glassnode_api_key,
                'a': 'BTC',
                'i': timeframe
            }
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    # Ostatnia wartość w danych
                    net_flow = data[-1]['v']
                    
                    # Normalizacja - typowe wartości są w zakresie -5000 BTC do 5000 BTC
                    # dla dziennego timeframe
                    normalized_flow = np.clip(net_flow / 5000, -1.0, 1.0)
                    
                    logger.info(f"Exchange Net Flow ({timeframe}): {net_flow:.2f} BTC -> {normalized_flow:.2f}")
                    
                    # Zapisanie do cache
                    self.cache[cache_key] = normalized_flow
                    self.cache_expiry[cache_key] = datetime.now() + timedelta(hours=1)  # Ważny przez godzinę
                    
                    return normalized_flow
        
        except Exception as e:
            logger.warning(f"Błąd podczas pobierania danych o przepływach giełdowych: {e}")
        
        # Alternatywnie, możemy spróbować użyć danych z CryptoQuant
        try:
            if 'cryptoquant' in self.api_keys:
                # Implementacja dla CryptoQuant API
                pass
        except Exception as e:
            logger.warning(f"Błąd podczas pobierania danych z CryptoQuant: {e}")
        
        return 0.0  # Neutralna wartość w przypadku błędu
    
    def get_integrated_sentiment(self):
        """
        Oblicza zintegrowany wskaźnik sentymentu na podstawie wszystkich dostępnych danych.
        
        Returns:
            float: Zintegrowany wskaźnik sentymentu w zakresie [-1, 1]
        """
        # Pobieramy wszystkie wskaźniki
        fear_greed = self.get_fear_greed_index()
        funding_rate = self.get_funding_rate()
        long_short_ratio = self.get_long_short_ratio()
        exchange_flows = self.get_exchange_flows()
        
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
        
        logger.info(f"Zintegrowany sentyment: {integrated_sentiment:.2f} [Fear&Greed: {fear_greed:.2f}, "
                   f"Funding: {funding_rate:.2f}, L/S: {long_short_ratio:.2f}, Flows: {exchange_flows:.2f}]")
        
        return integrated_sentiment
