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
        # Ustawienie domyślnych kluczy API jeśli nie są podane
        default_api_keys = {
            'glassnode': '',
            'coingecko': 'CG-qiF2zUuBB98vE14AvzSGJjwn', # Domyślny klucz CoinGecko
            'cryptoquant': ''
        }
        
        # Jeśli podano klucze, nadpisz domyślne
        if api_keys:
            default_api_keys.update(api_keys)
        
        self.api_keys = default_api_keys
        
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
                else:
                    logger.warning(f"Nieprawidłowa struktura danych z API Fear & Greed: {data}")
            else:
                logger.warning(f"Błąd HTTP podczas pobierania Fear & Greed Index: {response.status_code}")
        except Exception as e:
            logger.warning(f"Błąd podczas pobierania Fear & Greed Index (szczegóły): {e}")
        
        # W przypadku błędu próbujemy pobrać dane z CoinGecko
        try:
            # Pobranie aktualnej ceny i 24h change z CoinGecko
            coingecko_api_key = self.api_keys.get('coingecko')
            params = {
                'ids': 'bitcoin',
                'vs_currencies': 'usd',
                'include_24hr_change': 'true'
            }
            
            # Dodaj klucz demo API jeśli jest dostępny
            if coingecko_api_key:
                params['x_cg_demo_api_key'] = coingecko_api_key
            
            cg_url = "https://api.coingecko.com/api/v3/simple/price"
            response = requests.get(cg_url, params=params, timeout=10)
            
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
                else:
                    logger.warning(f"Nieprawidłowa struktura danych z CoinGecko: {data}")
            else:
                logger.warning(f"Błąd HTTP podczas pobierania danych z CoinGecko: {response.status_code}")
        except Exception as e:
            logger.warning(f"Błąd podczas pobierania danych z CoinGecko (szczegóły): {e}")
        
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
                    else:
                        logger.warning(f"Brak pola lastFundingRate w odpowiedzi Binance: {data}")
                else:
                    logger.warning(f"Błąd HTTP podczas pobierania funding rate z Binance: {response.status_code}")
                        
            elif exchange.lower() == "bybit":
                url = f"https://api.bybit.com/v2/public/tickers?symbol={symbol}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'result' in data and len(data['result']) > 0:
                        funding_rate = float(data['result'][0]['funding_rate'])
                    else:
                        logger.warning(f"Nieprawidłowa struktura danych z Bybit: {data}")
                else:
                    logger.warning(f"Błąd HTTP podczas pobierania funding rate z Bybit: {response.status_code}")
                        
            # Normalizacja - typowe funding rate są w zakresie -0.1% do 0.1%
            # Wartości powyżej 0.05% lub poniżej -0.05% są już dość ekstremalne
            normalized_funding_rate = np.clip(funding_rate * 20, -1.0, 1.0)
            
            logger.info(f"Funding Rate ({exchange}, {symbol}): {funding_rate:.6f} -> {normalized_funding_rate:.2f}")
            
            # Zapisanie do cache
            self.cache[cache_key] = normalized_funding_rate
            self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=60)  # Ważny przez godzinę
            
            return normalized_funding_rate
        
        except Exception as e:
            logger.warning(f"Błąd podczas pobierania funding rate (szczegóły): {e}")
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
                    else:
                        logger.warning(f"Pusta lub nieprawidłowa odpowiedź z Binance dla Long/Short Ratio: {data}")
                else:
                    logger.warning(f"Błąd HTTP podczas pobierania Long/Short Ratio z Binance: {response.status_code}")
                        
            elif exchange.lower() == "bybit":
                url = f"https://api.bybit.com/v2/public/account-ratio?symbol={symbol}&period=5min"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'result' in data and len(data['result']) > 0:
                        buy_ratio = float(data['result'][0]['buy_ratio'])
                        ratio = buy_ratio / 100  # Bybit podaje % kupujących
                    else:
                        logger.warning(f"Nieprawidłowa struktura danych z Bybit dla Long/Short Ratio: {data}")
                else:
                    logger.warning(f"Błąd HTTP podczas pobierania Long/Short Ratio z Bybit: {response.status_code}")
            
            logger.info(f"Long/Short Ratio ({exchange}, {symbol}): {ratio:.2f}")
            
            # Zapisanie do cache
            self.cache[cache_key] = ratio
            self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=15)  # Ważny przez 15 minut
            
            return ratio
        
        except Exception as e:
            logger.warning(f"Błąd podczas pobierania long/short ratio (szczegóły): {e}")
            return 0.5  # Neutralna wartość w przypadku błędu
    
    def get_exchange_flows_coingecko(self, timeframe="1d"):
        """
        Alternatywna metoda szacowania przepływów giełdowych przy użyciu CoinGecko API.
        Wykorzystuje zmiany w wolumenie i rezerwach giełd jako proxy dla przepływów.
        
        Args:
            timeframe (str): Przedział czasowy ("1h", "1d", "7d")
        
        Returns:
            float: Znormalizowana wartość przepływów w zakresie [-1, 1]
        """
        cache_key = f"exchange_flows_coingecko_{timeframe}"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            return self.cache[cache_key]
        
        # Pobierz klucz API CoinGecko
        coingecko_api_key = self.api_keys.get('coingecko')
        base_url = "https://api.coingecko.com/api/v3"
        
        logger.info("Próba użycia CoinGecko API do analizy przepływów giełdowych")
        
        try:
            # Pobranie danych o wolumenie dla głównych giełd
            exchanges = ["binance", "coinbase", "kucoin", "huobi", "kraken"]
            total_volume_change = 0
            valid_exchanges = 0
            
            for exchange in exchanges:
                try:
                    # Przygotuj parametry z kluczem API
                    params = {}
                    if coingecko_api_key:
                        params['x_cg_demo_api_key'] = coingecko_api_key
                    
                    url = f"{base_url}/exchanges/{exchange}"
                    response = requests.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # Pobierz zmianę wolumenu jako proxy dla przepływów
                        volume_change = data.get('trade_volume_24h_btc_percentage_change', 0)
                        
                        if volume_change:
                            total_volume_change += volume_change
                            valid_exchanges += 1
                            logger.debug(f"Giełda {exchange}: zmiana wolumenu {volume_change:.2f}%")
                    elif response.status_code == 429:
                        logger.warning(f"Przekroczony limit zapytań CoinGecko dla {exchange}")
                        # Dodaj opóźnienie, aby nie przekraczać limitów
                        time.sleep(5)
                    else:
                        logger.debug(f"Błąd HTTP podczas pobierania danych dla giełdy {exchange}: {response.status_code}")
                except Exception as e:
                    logger.debug(f"Błąd podczas pobierania danych dla giełdy {exchange} (szczegóły): {e}")
            
            # Jeśli nie udało się pobrać danych dla żadnej giełdy
            if valid_exchanges == 0:
                logger.warning("Nie udało się pobrać danych o wolumenie dla żadnej giełdy z CoinGecko")
            
            # Oblicz średnią zmianę wolumenu
            avg_volume_change = total_volume_change / valid_exchanges if valid_exchanges > 0 else 0
            
            # Sprawdź również całkowity wolumen BTC dla lepszego obrazu
            try:
                # Przygotuj parametry z kluczem API
                params = {}
                if coingecko_api_key:
                    params['x_cg_demo_api_key'] = coingecko_api_key
                
                url = f"{base_url}/coins/bitcoin"
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    market_data = data.get('market_data', {})
                    
                    # Zmiana wolumenu BTC może być wskaźnikiem przepływów
                    volume_change_btc = market_data.get('volume_change_percentage_24h', 0)
                    
                    # Pobierz również dane o aktywności on-chain jeśli dostępne
                    total_volume = market_data.get('total_volume', {}).get('usd', 0)
                    market_cap = market_data.get('market_cap', {}).get('usd', 0)
                    
                    # Wskaźnik zmienności (volume/market_cap) może wskazywać na przepływy
                    volatility_indicator = 0
                    if market_cap > 0:
                        volatility_indicator = total_volume / market_cap
                        logger.debug(f"Wskaźnik zmienności BTC (volume/market_cap): {volatility_indicator:.4f}")
                    
                    # Połącz oba wskaźniki
                    combined_indicator = (avg_volume_change * 0.6 + volume_change_btc * 0.4) / 100
                    
                    # Interpretacja: Nagły wzrost wolumenu może wskazywać na przepływy DO giełd (negatywny sygnał)
                    # Ujemny wskaźnik oznacza potencjalne wypływy z giełd (pozytywny sygnał)
                    normalized_flow = -np.clip(combined_indicator, -1.0, 1.0)
                    
                    logger.info(f"Exchange Net Flow (CoinGecko): Wolumen {avg_volume_change:.2f}%, BTC {volume_change_btc:.2f}% -> {normalized_flow:.2f}")
                    
                    # Zapisanie do cache
                    self.cache[cache_key] = normalized_flow
                    self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=30)
                    
                    return normalized_flow
                elif response.status_code == 429:
                    logger.warning(f"Przekroczony limit zapytań CoinGecko dla danych BTC")
                    time.sleep(5)
                else:
                    logger.warning(f"Błąd HTTP podczas pobierania danych o BTC z CoinGecko: {response.status_code}, {response.text}")
            except Exception as e:
                logger.warning(f"Błąd podczas pobierania danych o BTC z CoinGecko (szczegóły): {e}")
            
            return 0.0  # Neutralna wartość w przypadku braku danych
        
        except Exception as e:
            logger.warning(f"Błąd podczas szacowania przepływów giełdowych z CoinGecko (szczegóły): {e}")
            return 0.0  # Neutralna wartość w przypadku błędu
    
    def get_exchange_flows_webscraping(self, timeframe="1d"):
        """
        Alternatywna metoda szacowania przepływów giełdowych przy użyciu web scrapingu.
        Ściąga dane z publicznie dostępnych dashboardów lub alertów.
        
        Args:
            timeframe (str): Przedział czasowy ("1h", "1d", "7d")
        
        Returns:
            float: Znormalizowana wartość przepływów w zakresie [-1, 1]
        """
        cache_key = f"exchange_flows_scraping_{timeframe}"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            return self.cache[cache_key]
        
        try:
            # Scraping danych z Whale Alert
            import re
            from bs4 import BeautifulSoup
            
            # Określenie parametrów scrappingu
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Pobieranie danych z Whale Alert
            url = "https://whale-alert.io/"
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Analiza ostatnich alertów
                alerts = []
                alert_elements = soup.select('.transaction-item')
                
                logger.debug(f"Znaleziono {len(alert_elements)} alertów do analizy")
                
                for alert in alert_elements:
                    try:
                        # Wyciągnij tekst alertu
                        alert_text = alert.get_text()
                        
                        # Sprawdź czy to alert o BTC
                        if 'BTC' in alert_text:
                            # Sprawdź czy to przepływ do/z giełdy
                            is_to_exchange = 'to exchange' in alert_text.lower() or 'to binance' in alert_text.lower() or 'to coinbase' in alert_text.lower()
                            is_from_exchange = 'from exchange' in alert_text.lower() or 'from binance' in alert_text.lower() or 'from coinbase' in alert_text.lower()
                            
                            # Wyciągnij kwotę
                            amount_match = re.search(r'([\d,]+) BTC', alert_text)
                            if amount_match:
                                amount = float(amount_match.group(1).replace(',', ''))
                                
                                # Dodaj do listy alertów
                                alerts.append({
                                    'amount': amount,
                                    'to_exchange': is_to_exchange,
                                    'from_exchange': is_from_exchange,
                                    'time': datetime.now()  # Przybliżony czas, dokładny czas mógłby być wyciągnięty z alertu
                                })
                    except Exception as e:
                        logger.debug(f"Błąd podczas analizy alertu (szczegóły): {e}")
                
                # Analiza przepływów netto
                if alerts:
                    inflow = sum(alert['amount'] for alert in alerts if alert['to_exchange'])
                    outflow = sum(alert['amount'] for alert in alerts if alert['from_exchange'])
                    
                    # Oblicz przepływ netto
                    net_flow = outflow - inflow  # Dodatni oznacza więcej wypływów (pozytywny sygnał)
                    
                    # Normalizacja - typowe wartości są w zakresie -1000 BTC do 1000 BTC dla alertów z ostatnich 24h
                    normalized_flow = np.clip(net_flow / 1000, -1.0, 1.0)
                    
                    logger.info(f"Exchange Net Flow (Web Scraping): {net_flow:.2f} BTC -> {normalized_flow:.2f}")
                    
                    # Zapisanie do cache
                    self.cache[cache_key] = normalized_flow
                    self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=45)
                    
                    return normalized_flow
                else:
                    logger.warning("Nie znaleziono żadnych alertów związanych z przepływem BTC")
            else:
                logger.warning(f"Błąd HTTP podczas pobierania danych z Whale Alert: {response.status_code}")
        
        except ImportError as e:
            logger.warning(f"Brak wymaganych bibliotek do web scrapingu: {e}")
        except Exception as e:
            logger.warning(f"Błąd podczas scrapowania danych o przepływach giełdowych (szczegóły): {e}")
        
        return 0.0  # Neutralna wartość w przypadku błędu
    
    def get_exchange_flows(self, timeframe="1h"):
        """
        Pobiera informacje o przepływach BTC do/z giełd.
        Wykorzystuje alternatywne źródła danych, jeśli Glassnode API nie jest dostępne.
        
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
        
        # Próba użycia Glassnode API, jeśli klucz jest dostępny
        glassnode_api_key = self.api_keys.get('glassnode')
        if glassnode_api_key:
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
                        
                        logger.info(f"Exchange Net Flow (Glassnode, {timeframe}): {net_flow:.2f} BTC -> {normalized_flow:.2f}")
                        
                        # Zapisanie do cache
                        self.cache[cache_key] = normalized_flow
                        self.cache_expiry[cache_key] = datetime.now() + timedelta(hours=1)
                        
                        return normalized_flow
                    else:
                        logger.warning(f"Pusta lub nieprawidłowa odpowiedź z Glassnode: {data}")
                else:
                    logger.warning(f"Błąd HTTP podczas pobierania danych z Glassnode: {response.status_code}")
            except Exception as e:
                logger.warning(f"Błąd podczas pobierania danych z Glassnode (szczegóły): {e}")
        else:
            logger.info("Brak klucza API Glassnode, użycie alternatywnych źródeł danych...")
        
        # Alternatywa 1: CoinGecko API
        try:
            coingecko_flow = self.get_exchange_flows_coingecko(timeframe)
            if coingecko_flow != 0.0:
                return coingecko_flow
        except Exception as e:
            logger.warning(f"Błąd podczas pobierania danych z CoinGecko (szczegóły): {e}")
        
        # Alternatywa 2: Web Scraping (tylko jeśli CoinGecko zawiodło)
        try:
            import bs4
            scraping_flow = self.get_exchange_flows_webscraping(timeframe)
            if scraping_flow != 0.0:
                return scraping_flow
            else:
                logger.warning("Web scraping nie dostarczył danych o przepływach giełdowych")
        except ImportError:
            logger.warning("Biblioteka BeautifulSoup nie jest zainstalowana, web scraping niedostępny")
        except Exception as e:
            logger.warning(f"Błąd podczas pobierania danych przez web scraping (szczegóły): {e}")
        
        # Jeśli wszystkie metody zawiodły, zwróć neutralną wartość
        logger.warning("Wszystkie metody pobierania danych o przepływach giełdowych zawiodły")
        return 0.0
    
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