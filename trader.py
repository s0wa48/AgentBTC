"""
Moduł wykonywania transakcji
Obsługuje wykonywanie operacji handlowych na podstawie sygnałów
"""

import logging
import requests
import json
import os
from datetime import datetime
import ccxt

import config
# Import modułu powiadomień email
from email_notifier import EmailNotifier

logger = logging.getLogger("trader")

class Trader:
    """
    Klasa do wykonywania transakcji na podstawie sygnałów.
    
    Obsługuje:
    - Wysyłanie sygnałów przez webhook
    - Wykonywanie transakcji przez API giełdy
    - Zarządzanie ryzykiem
    """
    
    def __init__(self):
        """Inicjalizacja tradera."""
        logger.info("Inicjalizacja modułu tradingowego")
        self.webhook_url = config.WEBHOOK_URL
        self.exchange = None  # Inicjalizacja na żądanie
        
        # Inicjalizacja modułu powiadomień email
        self.email_notifier = EmailNotifier()
    
    def _initialize_exchange(self):
        """Inicjalizacja połączenia z giełdą na żądanie."""
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
    
    def execute_trade(self, signal, current_position, position_size):
        """
        Wykonanie transakcji na podstawie sygnału.
        
        Args:
            signal (str): Sygnał tradingowy ('LONG', 'SHORT', 'FLAT')
            current_position (str): Aktualna pozycja ('LONG', 'SHORT', 'FLAT')
            position_size (float): Wielkość pozycji jako procent kapitału (0.0-1.0)
            
        Returns:
            dict: Wynik wykonania transakcji
        """
        try:
            # Jeśli sygnał jest taki sam jak aktualna pozycja, nic nie robimy
            if signal == current_position:
                return {
                    'success': True,
                    'message': f"Utrzymanie obecnej pozycji: {signal}",
                    'position': signal,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Przygotowanie danych transakcji
            timestamp = datetime.now().isoformat()
            symbol = config.SYMBOL
            
            # Określenie akcji
            if current_position == "FLAT" and signal == "LONG":
                action = "OPEN_LONG"
            elif current_position == "FLAT" and signal == "SHORT":
                action = "OPEN_SHORT"
            elif current_position == "LONG" and signal == "FLAT":
                action = "CLOSE_LONG"
            elif current_position == "SHORT" and signal == "FLAT":
                action = "CLOSE_SHORT"
            elif current_position == "LONG" and signal == "SHORT":
                action = "SWITCH_TO_SHORT"  # Zamknij LONG i otwórz SHORT
            elif current_position == "SHORT" and signal == "LONG":
                action = "SWITCH_TO_LONG"   # Zamknij SHORT i otwórz LONG
            else:
                action = "UNKNOWN"
            
            # Parametry zarządzania ryzykiem
            risk_params = {
                'stop_loss_pct': config.RISK['stop_loss_pct'],
                'take_profit_pct': config.RISK['take_profit_pct']
            }
            
            # Wybór metody wykonania transakcji
            if config.DEBUG_MODE:
                result = self._execute_demo_trade(action, symbol, position_size, risk_params)
            elif self.webhook_url:
                result = self._execute_webhook_trade(action, symbol, position_size, risk_params)
            else:
                result = self._execute_exchange_trade(action, symbol, position_size, risk_params)
            
            # Jeśli transakcja zakończona sukcesem i zmieniliśmy pozycję, wysyłamy powiadomienie
            if result["success"] and signal != current_position:
                # Ustalenie ceny dla powiadomienia
                price = result.get('price', None)
                
                # Wysłanie powiadomienia o zmianie pozycji
                if config.EMAIL_NOTIFICATION.get('send_on_position_change', False):
                    # Użyj market_position z rezultatu, jeśli istnieje, w przeciwnym razie użyj signal
                    market_position = result.get('market_position', signal)
                    self.email_notifier.send_trade_notification(
                        trade_action=action,
                        market_position=market_position,
                        symbol=symbol,
                        price=price,
                        details=result
                    )
            
            return result
        
        except Exception as e:
            logger.error(f"Błąd podczas wykonywania transakcji: {e}")
            return {
                'success': False,
                'message': f"Błąd: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def _execute_demo_trade(self, action, symbol, position_size, risk_params):
        """
        Symulacja wykonania transakcji w trybie demo.
        
        Args:
            action (str): Rodzaj akcji
            symbol (str): Symbol instrumentu
            position_size (float): Wielkość pozycji
            risk_params (dict): Parametry zarządzania ryzykiem
            
        Returns:
            dict: Wynik wykonania transakcji
        """
        logger.info(f"[DEMO] Wykonanie transakcji: {action} {symbol} wielkość={position_size:.4f}")
        
        # Symulacja ceny
        import random
        current_price = 50000 * (1 + random.uniform(-0.01, 0.01))
        
        # Zapis do pliku CSV z historią transakcji
        log_dir = config.RESULTS_DIR
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "demo_trades.csv")
        
        # Sprawdzenie czy plik istnieje
        file_exists = os.path.isfile(log_file)
        
        # Przygotowanie danych do zapisu
        trade_data = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'symbol': symbol,
            'price': current_price,
            'position_size': position_size,
            'stop_loss_pct': risk_params['stop_loss_pct'],
            'take_profit_pct': risk_params['take_profit_pct']
        }
        
        # Zapis do CSV
        import csv
        with open(log_file, mode='a', newline='') as file:
            fieldnames = trade_data.keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(trade_data)
        
        # Określenie market_position na podstawie akcji
        if action in ["OPEN_LONG", "SWITCH_TO_LONG"]:
            market_position = "LONG"
        elif action in ["OPEN_SHORT", "SWITCH_TO_SHORT"]:
            market_position = "SHORT"
        elif action in ["CLOSE_LONG", "CLOSE_SHORT"]:
            market_position = "FLAT"
        else:
            market_position = "FLAT"
        
        return {
            'success': True,
            'message': f"[DEMO] Wykonano transakcję: {action}",
            'action': action,
            'market_position': market_position,
            'symbol': symbol,
            'position_size': position_size,
            'price': current_price,
            'stop_loss': current_price * (1 - risk_params['stop_loss_pct']),
            'take_profit': current_price * (1 + risk_params['take_profit_pct']),
            'timestamp': datetime.now().isoformat()
        }
    
    def _execute_webhook_trade(self, action, symbol, position_size, risk_params):
        """
        Wykonanie transakcji przez webhook.
        
        Args:
            action (str): Rodzaj akcji
            symbol (str): Symbol instrumentu
            position_size (float): Wielkość pozycji
            risk_params (dict): Parametry zarządzania ryzykiem
            
        Returns:
            dict: Wynik wykonania transakcji
        """
        try:
            if not self.webhook_url:
                return {
                    'success': False,
                    'message': "Brak skonfigurowanego URL webhooka",
                    'timestamp': datetime.now().isoformat()
                }
            
            # Określenie market_position na podstawie akcji
            if action in ["OPEN_LONG", "SWITCH_TO_LONG"]:
                market_position = "LONG"
            elif action in ["OPEN_SHORT", "SWITCH_TO_SHORT"]:
                market_position = "SHORT"
            elif action in ["CLOSE_LONG", "CLOSE_SHORT"]:
                market_position = "FLAT"
            else:
                market_position = "FLAT"
            
            # Przygotowanie danych do wysłania w nowym formacie
            webhook_data = {
                'api_key': "025609e8-5cd2-474d-8bd5-739c9e94b0a2",
                'order_type': market_position,
                'trade_pair': "BTCUSD",
                'leverage': "0.5"
            }
            
            # Wysłanie danych przez webhook
            response = requests.post(
                self.webhook_url, 
                data=json.dumps(webhook_data),
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook wysłany pomyślnie: {action} -> {market_position}")
                
                return {
                    'success': True,
                    'message': f"Webhook wysłany: {action} -> {market_position}",
                    'action': action,
                    'market_position': market_position,
                    'symbol': symbol,
                    'position_size': position_size,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.error(f"Błąd podczas wysyłania webhooka: {response.status_code}, {response.text}")
                return {
                    'success': False,
                    'message': f"Błąd webhooka: {response.status_code}",
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Błąd podczas wykonywania transakcji przez webhook: {e}")
            return {
                'success': False,
                'message': f"Błąd: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def _execute_exchange_trade(self, action, symbol, position_size, risk_params):
        """
        Wykonanie transakcji bezpośrednio przez API giełdy.
        
        Args:
            action (str): Rodzaj akcji
            symbol (str): Symbol instrumentu
            position_size (float): Wielkość pozycji
            risk_params (dict): Parametry zarządzania ryzykiem
            
        Returns:
            dict: Wynik wykonania transakcji
        """
        try:
            # Leniwa inicjalizacja giełdy
            if self.exchange is None:
                self.exchange = self._initialize_exchange()
            
            if self.exchange is None:
                return {
                    'success': False,
                    'message': "Brak połączenia z giełdą",
                    'timestamp': datetime.now().isoformat()
                }
            
            # Pobranie informacji o koncie
            balance = self.exchange.fetch_balance()
            available_balance = balance['free']['USDT'] if 'USDT' in balance['free'] else 0
            
            # Pobranie aktualnej ceny
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Obliczenie ilości BTC do kupienia/sprzedania
            amount_usd = available_balance * position_size
            amount_btc = amount_usd / current_price
            
            # Określenie market_position na podstawie akcji
            if action in ["OPEN_LONG", "SWITCH_TO_LONG"]:
                market_position = "LONG"
            elif action in ["OPEN_SHORT", "SWITCH_TO_SHORT"]:
                market_position = "SHORT"
            elif action in ["CLOSE_LONG", "CLOSE_SHORT"]:
                market_position = "FLAT"
            else:
                market_position = "FLAT"
            
            # Wykonanie odpowiedniej akcji
            order = None
            
            if action == "OPEN_LONG":
                order = self.exchange.create_market_buy_order(symbol, amount_btc)
            elif action == "OPEN_SHORT":
                order = self.exchange.create_market_sell_order(symbol, amount_btc)
            elif action == "CLOSE_LONG":
                # Zamknięcie pozycji długiej = sprzedaż
                positions = self.exchange.fetch_positions([symbol])
                for position in positions:
                    if position['side'] == 'long' and position['contracts'] > 0:
                        order = self.exchange.create_market_sell_order(symbol, position['contracts'])
            elif action == "CLOSE_SHORT":
                # Zamknięcie pozycji krótkiej = kupno
                positions = self.exchange.fetch_positions([symbol])
                for position in positions:
                    if position['side'] == 'short' and position['contracts'] > 0:
                        order = self.exchange.create_market_buy_order(symbol, position['contracts'])
            elif action == "SWITCH_TO_SHORT":
                # Najpierw zamknij LONG
                positions = self.exchange.fetch_positions([symbol])
                for position in positions:
                    if position['side'] == 'long' and position['contracts'] > 0:
                        self.exchange.create_market_sell_order(symbol, position['contracts'])
                
                # Potem otwórz SHORT
                order = self.exchange.create_market_sell_order(symbol, amount_btc)
            elif action == "SWITCH_TO_LONG":
                # Najpierw zamknij SHORT
                positions = self.exchange.fetch_positions([symbol])
                for position in positions:
                    if position['side'] == 'short' and position['contracts'] > 0:
                        self.exchange.create_market_buy_order(symbol, position['contracts'])
                
                # Potem otwórz LONG
                order = self.exchange.create_market_buy_order(symbol, amount_btc)
            
            # Sprawdzenie wyniku
            if order:
                logger.info(f"Transakcja wykonana: {action} {symbol} po cenie {current_price}")
                return {
                    'success': True,
                    'message': f"Transakcja wykonana: {action}",
                    'action': action,
                    'market_position': market_position,
                    'symbol': symbol,
                    'position_size': position_size,
                    'price': current_price,
                    'order_id': order.get('id'),
                    'amount': amount_btc,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.warning(f"Brak odpowiedzi z giełdy dla akcji: {action}")
                return {
                    'success': False,
                    'message': f"Brak odpowiedzi z giełdy: {action}",
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Błąd podczas wykonywania transakcji przez giełdę: {e}")
            return {
                'success': False,
                'message': f"Błąd giełdy: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def get_current_position(self, symbol=None):
        """
        Pobranie informacji o aktualnej pozycji.
        
        Args:
            symbol (str, optional): Symbol instrumentu
            
        Returns:
            str: Aktualna pozycja ('LONG', 'SHORT', 'FLAT')
        """
        try:
            if symbol is None:
                symbol = config.SYMBOL
            
            # W trybie demo sprawdzamy z pliku
            if config.DEBUG_MODE:
                return self._get_demo_position(symbol)
            
            # W przeciwnym razie sprawdzamy giełdę
            if self.exchange is None:
                self.exchange = self._initialize_exchange()
            
            if self.exchange is None:
                logger.error("Brak połączenia z giełdą do sprawdzenia pozycji")
                return "FLAT"  # Bezpieczna opcja
            
            # Pobranie pozycji
            positions = self.exchange.fetch_positions([symbol])
            
            for position in positions:
                if position['symbol'] == symbol:
                    # Sprawdzenie stanu pozycji
                    if position['side'] == 'long' and position['contracts'] > 0:
                        logger.info(f"Aktualna pozycja: LONG, wielkość={position['contracts']}")
                        return "LONG"
                    elif position['side'] == 'short' and position['contracts'] > 0:
                        logger.info(f"Aktualna pozycja: SHORT, wielkość={position['contracts']}")
                        return "SHORT"
            
            logger.info("Brak aktywnej pozycji (FLAT)")
            return "FLAT"
        
        except Exception as e:
            logger.error(f"Błąd podczas pobierania informacji o pozycji: {e}")
            return "FLAT"  # Bezpieczna opcja w przypadku błędu
    
    def _get_demo_position(self, symbol):
        """
        Pobranie informacji o pozycji w trybie demo (z pliku).
        
        Args:
            symbol (str): Symbol instrumentu
            
        Returns:
            str: Aktualna pozycja ('LONG', 'SHORT', 'FLAT')
        """
        try:
            log_file = os.path.join(config.RESULTS_DIR, "demo_trades.csv")
            
            if not os.path.exists(log_file):
                return "FLAT"
            
            # Wczytanie CSV
            import csv
            trades = []
            with open(log_file, mode='r') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    trades.append(row)
            
            if not trades:
                return "FLAT"
            
            # Ostatnia transakcja
            last_trade = trades[-1]
            action = last_trade.get('action', '')
            
            if action.endswith('_LONG') or action == 'SWITCH_TO_LONG':
                return "LONG"
            elif action.endswith('_SHORT') or action == 'SWITCH_TO_SHORT':
                return "SHORT"
            elif action.startswith('CLOSE_'):
                return "FLAT"
            
            return "FLAT"  # Domyślnie
        
        except Exception as e:
            logger.error(f"Błąd podczas pobierania pozycji demo: {e}")
            return "FLAT"  # Bezpieczna opcja w przypadku błędu