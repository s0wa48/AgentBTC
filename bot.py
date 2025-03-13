"""
Główny moduł Bitcoin Trading Bot
Integruje wszystkie komponenty i zarządza cyklem tradingowym
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import time

import config
import analyzer
import trader

logger = logging.getLogger("bot")

class BitcoinTradingBot:
    """
    Główna klasa bota tradingowego łącząca wszystkie komponenty.
    """
    
    def __init__(self):
        """Inicjalizacja bota."""
        logger.info("Inicjalizacja Bitcoin Trading Bot")
        self.analyzer = analyzer.MarketAnalyzer()
        self.trader = trader.Trader()
        self.current_position = "FLAT"  # FLAT, LONG, SHORT
        self.entry_price = 0
        self.position_size = 0
        
    def fetch_data(self):
        """Pobierz aktualne dane rynkowe."""
        df = self.analyzer.fetch_market_data(
            symbol=config.SYMBOL,
            timeframe=config.TIMEFRAME,
            limit=300  # Potrzebujemy wystarczająco danych dla wskaźników
        )
        return df
    
    def analyze_market(self, df):
        """Analiza rynku i wygenerowanie sygnału."""
        # 1. Predykcja z modelu LSTM
        prediction = self.analyzer.predict_price_movement(df)
        
        # 2. Analiza techniczna
        ta_signal = self.analyzer.generate_ta_signal(df)
        
        # 3. Analiza sentymentu (opcjonalnie)
        sentiment = self.analyzer.analyze_market_sentiment()
        
        # 4. Połączenie sygnałów w jeden
        final_signal = self._combine_signals(prediction, ta_signal, sentiment)
        
        return {
            "signal": final_signal,
            "prediction": prediction,
            "ta_signal": ta_signal,
            "sentiment": sentiment
        }
    
    def _combine_signals(self, prediction, ta_signal, sentiment):
        """Połączenie sygnałów z różnych źródeł w jeden."""
        signal_value = 0
        
        # Składowa z modelu LSTM
        if prediction:
            model_direction = prediction.get('direction', 'FLAT')
            model_confidence = abs(prediction.get('percent_change', 0))
            
            if model_direction == "UP":
                signal_value += config.SIGNALS["model_weight"] * min(model_confidence / 5, 1.0)
            elif model_direction == "DOWN":
                signal_value -= config.SIGNALS["model_weight"] * min(model_confidence / 5, 1.0)
        
        # Składowa z analizy technicznej
        if ta_signal == "LONG":
            signal_value += config.SIGNALS["tech_analysis_weight"]
        elif ta_signal == "SHORT":
            signal_value -= config.SIGNALS["tech_analysis_weight"]
        
        # Składowa z sentymentu rynkowego
        signal_value += config.SIGNALS["sentiment_weight"] * sentiment
        
        # Uwzględnienie aktualnej pozycji (inercja)
        inertia_factor = 0.1
        if self.current_position == "LONG":
            signal_value += inertia_factor
        elif self.current_position == "SHORT":
            signal_value -= inertia_factor
        
        # Generowanie finalnego sygnału
        if signal_value > config.SIGNALS["long_threshold"]:
            final_signal = "LONG"
        elif signal_value < config.SIGNALS["short_threshold"]:
            final_signal = "SHORT"
        else:
            final_signal = "FLAT"
            
        logger.info(f"Sygnał: {final_signal} (wartość={signal_value:.4f})")
        return final_signal
    
    def execute_trade(self, signal_data):
        """Wykonanie transakcji na podstawie sygnału."""
        signal = signal_data["signal"]
        
        # Jeśli sygnał jest taki sam jak aktualna pozycja, nic nie robimy
        if signal == self.current_position:
            logger.info(f"Utrzymanie obecnej pozycji: {signal}")
            return {"success": True, "message": f"Pozycja bez zmian: {signal}"}
        
        # Obliczanie wielkości pozycji
        df = self.fetch_data()
        current_price = df['close'].iloc[-1]
        
        risk_per_trade = config.RISK["risk_per_trade"]
        position_size = risk_per_trade
        
        # Wykonanie transakcji
        result = self.trader.execute_trade(
            signal=signal,
            current_position=self.current_position,
            position_size=position_size
        )
        
        # Aktualizacja stanu bota
        if result["success"]:
            self.current_position = signal
            if signal != "FLAT":
                self.entry_price = current_price
                self.position_size = position_size
            else:
                self.entry_price = 0
                self.position_size = 0
        
        return result
    
    def run_trading_cycle(self):
        """Wykonanie pełnego cyklu tradingowego."""
        try:
            # 1. Pobranie aktualnych danych
            df = self.fetch_data()
            if df is None or df.empty:
                logger.error("Brak danych do analizy")
                return {"success": False, "message": "Brak danych"}
            
            # 2. Analiza rynku i wygenerowanie sygnału
            signal_data = self.analyze_market(df)
            
            # 3. Wykonanie transakcji
            trade_result = self.execute_trade(signal_data)
            
            # 4. Zapisanie wyników
            self._log_cycle_results(signal_data, trade_result)
            
            return {
                "success": True,
                "signal": signal_data["signal"],
                "prediction": signal_data["prediction"],
                "trade_result": trade_result
            }
            
        except Exception as e:
            logger.error(f"Błąd podczas cyklu tradingowego: {e}")
            return {"success": False, "error": str(e)}
    
    def _log_cycle_results(self, signal_data, trade_result):
        """Zapisanie wyników cyklu tradingowego."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Podstawowe informacje
        log_data = {
            "timestamp": timestamp,
            "signal": signal_data["signal"],
            "trade_success": trade_result["success"]
        }
        
        # Dodanie szczegółów predykcji
        if "prediction" in signal_data and signal_data["prediction"]:
            prediction = signal_data["prediction"]
            log_data.update({
                "current_price": prediction.get("current_price", 0),
                "predicted_price": prediction.get("predicted_price", 0),
                "percent_change": prediction.get("percent_change", 0)
            })
        
        # Zapisanie do CSV
        log_file = os.path.join(config.RESULTS_DIR, "trading_log.csv")
        
        # Sprawdzenie czy plik istnieje
        file_exists = os.path.isfile(log_file)
        
        # Zapisanie do pliku
        pd.DataFrame([log_data]).to_csv(
            log_file, 
            mode='a', 
            header=not file_exists,
            index=False
        )
    
    def run_live(self, interval_minutes=None):
        """Uruchomienie bota w trybie ciągłym."""
        if interval_minutes is None:
            interval_minutes = config.LIVE_TRADING["check_interval_minutes"]
            
        logger.info(f"Uruchamianie tradingu na żywo z interwałem {interval_minutes} minut")
        
        try:
            while True:
                logger.info("Rozpoczęcie cyklu tradingowego")
                
                result = self.run_trading_cycle()
                
                if not result["success"]:
                    logger.error(f"Błąd w cyklu: {result.get('error', 'Nieznany błąd')}")
                
                logger.info(f"Cykl zakończony. Następny za {interval_minutes} minut")
                
                # Oczekiwanie na następny cykl
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("Trading zatrzymany przez użytkownika")
        except Exception as e:
            logger.error(f"Nieoczekiwany błąd: {e}")
    
    def backtest(self, start_date=None, end_date=None, initial_balance=None):
        """
        Backtesting strategii na danych historycznych.
        
        Args:
            start_date (str): Data początkowa w formacie 'YYYY-MM-DD'
            end_date (str): Data końcowa w formacie 'YYYY-MM-DD'
            initial_balance (float): Początkowy kapitał
        
        Returns:
            dict: Wyniki backtestingu
        """
        from backtest import run_backtest
        
        if start_date is None:
            start_date = config.BACKTEST["start_date"]
        if end_date is None:
            end_date = config.BACKTEST["end_date"]
        if initial_balance is None:
            initial_balance = config.BACKTEST["initial_balance"]
        
        return run_backtest(
            analyzer=self.analyzer,
            start_date=start_date,
            end_date=end_date,
            initial_balance=initial_balance
        )


if __name__ == "__main__":
    # Prosta demonstracja działania bota
    bot = BitcoinTradingBot()
    result = bot.run_trading_cycle()
    print(f"Wynik cyklu: {result}")
