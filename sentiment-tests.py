"""
Skrypt do testowania modułu analizy sentymentu
"""

import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Dodanie katalogu głównego do ścieżki Pythona
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment_analyzer import SentimentAnalyzer

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("test_sentiment")

def test_all_sentiment_indicators():
    """
    Testuje wszystkie wskaźniki sentymentu i wyświetla wyniki.
    """
    # Inicjalizacja analizatora sentymentu
    analyzer = SentimentAnalyzer(api_keys={
        'glassnode': '',  # Wstaw swój klucz API jeśli posiadasz
        'cryptoquant': '' # Wstaw swój klucz API jeśli posiadasz
    })
    
    # Test Fear & Greed Index
    fear_greed = analyzer.get_fear_greed_index()
    print(f"Fear & Greed Index: {fear_greed:.2f}")
    
    # Test Funding Rate dla różnych giełd
    binance_funding = analyzer.get_funding_rate(exchange="binance", symbol="BTCUSDT")
    bybit_funding = analyzer.get_funding_rate(exchange="bybit", symbol="BTCUSD")
    print(f"Binance Funding Rate: {binance_funding:.4f}")
    print(f"Bybit Funding Rate: {bybit_funding:.4f}")
    
    # Test Long/Short Ratio
    binance_ls_ratio = analyzer.get_long_short_ratio(exchange="binance", symbol="BTCUSDT")
    print(f"Binance Long/Short Ratio: {binance_ls_ratio:.2f}")
    
    # Test Exchange Flows
    exchange_flows = analyzer.get_exchange_flows(timeframe="1d")
    print(f"Exchange Flows (1d): {exchange_flows:.2f}")
    
    # Test zintegrowanego sentymentu
    integrated_sentiment = analyzer.get_integrated_sentiment()
    print(f"Zintegrowany sentyment: {integrated_sentiment:.2f}")
    
    return {
        'fear_greed': fear_greed,
        'binance_funding': binance_funding,
        'bybit_funding': bybit_funding,
        'ls_ratio': binance_ls_ratio,
        'exchange_flows': exchange_flows,
        'integrated': integrated_sentiment
    }

def collect_sentiment_over_time(days=7, interval_hours=6):
    """
    Zbiera dane o sentymencie przez określony okres.
    
    Args:
        days (int): Liczba dni do zebrania danych
        interval_hours (int): Interwał między pomiarami w godzinach
        
    Returns:
        pd.DataFrame: DataFrame z zebranymi danymi
    """
    analyzer = SentimentAnalyzer()
    
    data = []
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    current_time = start_time
    
    print(f"Zbieranie danych o sentymencie od {start_time} do {end_time}...")
    
    while current_time <= end_time:
        print(f"Pobieranie danych dla {current_time}...")
        
        try:
            fear_greed = analyzer.get_fear_greed_index()
            funding_rate = analyzer.get_funding_rate()
            ls_ratio = analyzer.get_long_short_ratio()
            integrated = analyzer.get_integrated_sentiment()
            
            data.append({
                'timestamp': current_time,
                'fear_greed': fear_greed,
                'funding_rate': funding_rate,
                'ls_ratio': ls_ratio,
                'integrated': integrated
            })
            
            # Dodaj sztuczne opóźnienie, aby nie przekroczyć limitów API
            import time
            time.sleep(2)
            
        except Exception as e:
            print(f"Błąd podczas pobierania danych: {e}")
        
        current_time += timedelta(hours=interval_hours)
    
    return pd.DataFrame(data)

def plot_sentiment_data(df):
    """
    Tworzy wykres na podstawie zebranych danych o sentymencie.
    
    Args:
        df (pd.DataFrame): DataFrame z danymi o sentymencie
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(4, 1, 1)
    plt.plot(df['timestamp'], df['fear_greed'], label='Fear & Greed Index')
    plt.title('Fear & Greed Index')
    plt.grid(True)
    
    plt.subplot(4, 1, 2)
    plt.plot(df['timestamp'], df['funding_rate'], label='Funding Rate', color='orange')
    plt.title('Funding Rate')
    plt.grid(True)
    
    plt.subplot(4, 1, 3)
    plt.plot(df['timestamp'], df['ls_ratio'], label='Long/Short Ratio', color='green')
    plt.title('Long/Short Ratio')
    plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3)
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    plt.plot(df['timestamp'], df['integrated'], label='Integrated Sentiment', color='red')
    plt.title('Integrated Sentiment')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis.png')
    plt.show()

def main():
    """Główna funkcja testowa."""
    print("=== Test wskaźników sentymentu ===")
    results = test_all_sentiment_indicators()
    
    print("\n=== Zbieranie danych o sentymencie ===")
    # Zakomentuj poniższą linię, jeśli nie chcesz zbierać danych przez dłuższy czas
    # df = collect_sentiment_over_time(days=1, interval_hours=1)
    # plot_sentiment_data(df)
    
    print("\nTesty zakończone.")

if __name__ == "__main__":
    main()
