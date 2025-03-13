"""
Skrypt do uruchomienia Bitcoin Trading Bot w trybie na żywo
"""

import os
import time
import argparse
import logging
from datetime import datetime

import config
from bot import BitcoinTradingBot

logger = logging.getLogger("live_trading")

def parse_arguments():
    """Parsowanie argumentów wiersza poleceń."""
    parser = argparse.ArgumentParser(description='Uruchomienie systemu tradingowego na żywo')
    
    parser.add_argument('--interval', type=int, default=config.LIVE_TRADING['check_interval_minutes'],
                        help='Interwał między sprawdzeniami (w minutach)')
    parser.add_argument('--model', type=str,
                        help='Ścieżka do modelu (jeśli różna od domyślnej)')
    parser.add_argument('--demo', action='store_true',
                        help='Uruchomienie w trybie demo (bez rzeczywistych transakcji)')
    
    return parser.parse_args()

def setup_trading_environment(demo_mode=False):
    """
    Konfiguracja środowiska tradingowego.
    
    Args:
        demo_mode (bool): Czy uruchomić w trybie demo
        
    Returns:
        BitcoinTradingBot: Instancja bota
    """
    # Ustawienie trybu demo, jeśli wymagane
    if demo_mode:
        config.DEBUG_MODE = True
        logger.info("Uruchamianie w trybie demo (bez rzeczywistych transakcji)")
    
    # Inicjalizacja bota
    bot = BitcoinTradingBot()
    
    return bot

def notify_startup(demo_mode):
    """
    Wysłanie powiadomienia o uruchomieniu bota.
    
    Args:
        demo_mode (bool): Czy bot działa w trybie demo
    """
    try:
        # Przygotowanie wiadomości
        mode = "DEMO" if demo_mode else "LIVE"
        message = f"Bitcoin Trading Bot uruchomiony w trybie {mode} o {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Zapisanie do logu
        logger.info(message)
        
        # Wysłanie e-mailem, jeśli skonfigurowano
        if config.LIVE_TRADING.get('notification_email'):
            import smtplib
            from email.mime.text import MIMEText
            
            smtp_config = config.LIVE_TRADING.get('smtp', {})
            
            if all(k in smtp_config for k in ['server', 'user', 'password']):
                try:
                    # Przygotowanie wiadomości
                    msg = MIMEText(message)
                    msg['Subject'] = f"Bitcoin Trading Bot - {mode} uruchomiony"
                    msg['From'] = smtp_config['user']
                    msg['To'] = config.LIVE_TRADING['notification_email']
                    
                    # Wysłanie e-maila
                    with smtplib.SMTP(smtp_config['server'], smtp_config.get('port', 587)) as server:
                        server.starttls()
                        server.login(smtp_config['user'], smtp_config['password'])
                        server.send_message(msg)
                    
                    logger.info(f"Powiadomienie e-mail wysłane do {config.LIVE_TRADING['notification_email']}")
                except Exception as e:
                    logger.warning(f"Nie udało się wysłać powiadomienia e-mail: {str(e)}")
    
    except Exception as e:
        logger.error(f"Błąd podczas wysyłania powiadomienia o uruchomieniu: {str(e)}")

def run_live_trading():
    """Główna funkcja uruchamiająca trading na żywo."""
    args = parse_arguments()
    
    # Konfiguracja środowiska
    demo_mode = args.demo or config.DEBUG_MODE
    bot = setup_trading_environment(demo_mode)
    
    # Powiadomienie o uruchomieniu
    notify_startup(demo_mode)
    
    # Potwierdzenie od użytkownika
    if not demo_mode:
        print("\n" + "="*50)
        print("OSTRZEŻENIE: Uruchamiasz system w trybie rzeczywistym!")
        print("System będzie wykonywał rzeczywiste transakcje z użyciem prawdziwych pieniędzy.")
        print("="*50 + "\n")
        
        confirmation = input("Czy na pewno chcesz kontynuować? (wpisz TAK aby potwierdzić): ")
        if confirmation.upper() != "TAK":
            print("Anulowano uruchomienie.")
            return
    
    # Uruchomienie bota
    print(f"\nSystem tradingowy uruchomiony z interwałem {args.interval} minut.")
    print("Naciśnij Ctrl+C, aby zatrzymać.\n")
    
    try:
        bot.run_live(interval_minutes=args.interval)
    except KeyboardInterrupt:
        print("\nSystem tradingowy zatrzymany przez użytkownika.\n")
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas tradingu na żywo: {e}")
        print(f"\nWystąpił błąd: {e}\n")

if __name__ == "__main__":
    print("Bitcoin Trading Bot - Uruchamianie tradingu na żywo")
    run_live_trading()
