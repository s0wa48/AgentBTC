"""
Skrypt do testowania strategii tradingowej na danych historycznych
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
from datetime import datetime

import config
from analyzer import MarketAnalyzer

logger = logging.getLogger("backtest")

def parse_arguments():
    """Parsowanie argumentów wiersza poleceń."""
    parser = argparse.ArgumentParser(description='Backtesting strategii tradingowej')
    
    parser.add_argument('--start_date', type=str, default=config.BACKTEST['start_date'],
                        help='Data początkowa w formacie YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default=config.BACKTEST['end_date'],
                        help='Data końcowa w formacie YYYY-MM-DD')
    parser.add_argument('--initial_balance', type=float, default=config.BACKTEST['initial_balance'],
                        help='Początkowy kapitał')
    parser.add_argument('--model', type=str,
                        help='Ścieżka do modelu (jeśli różna od domyślnej)')
    parser.add_argument('--plot', action='store_true',
                        help='Generowanie wykresu wyników')
    parser.add_argument('--output', type=str,
                        help='Ścieżka wyjściowa dla wyników')
    
    return parser.parse_args()

def run_backtest(analyzer=None, start_date=None, end_date=None, initial_balance=10000):
    """
    Przeprowadzenie backtestingu strategii.
    
    Args:
        analyzer (MarketAnalyzer, optional): Instancja analizatora
        start_date (str): Data początkowa w formacie 'YYYY-MM-DD'
        end_date (str): Data końcowa w formacie 'YYYY-MM-DD'
        initial_balance (float): Początkowy kapitał
        
    Returns:
        dict: Wyniki backtestingu
    """
    if analyzer is None:
        analyzer = MarketAnalyzer()
    
    if start_date is None:
        start_date = config.BACKTEST['start_date']
    
    if end_date is None:
        end_date = config.BACKTEST['end_date']
    
    # Wczytanie modelu
    if not analyzer.load_model():
        logger.error("Nie udało się wczytać modelu. Najpierw wytrenuj model.")
        return None
    
    # Pobranie danych historycznych
    logger.info(f"Pobieranie danych historycznych od {start_date} do {end_date}")
    df = analyzer.fetch_market_data(timeframe=config.TIMEFRAME, limit=10000)
    
    if df.empty:
        logger.error("Nie udało się pobrać danych historycznych")
        return None
    
    # Filtracja danych według zakresu dat
    df['date'] = df['timestamp'].dt.date
    df = df[(df['date'] >= pd.to_datetime(start_date).date()) & 
            (df['date'] <= pd.to_datetime(end_date).date())]
    
    if len(df) < config.MODEL['sequence_length']:
        logger.warning(f"Za mało danych po filtracji. Używam ostatnich {config.MODEL['sequence_length']} świec bez filtracji.")
    # Użyj ostatnich dostępnych danych zamiast filtrowania
        original_df = analyzer.fetch_market_data(timeframe=config.TIMEFRAME, limit=1000)
        df = original_df.tail(config.MODEL['sequence_length'] * 2)  # Weź więcej danych dla bezpieczeństwa
    
    # Inicjalizacja zmiennych backtestingu
    balance = initial_balance
    position = "FLAT"  # FLAT, LONG, SHORT
    entry_price = 0
    position_size = 0
    trades = []
    balance_history = [{'timestamp': df['timestamp'].iloc[0], 'balance': balance}]
    
    # Minimalna długość sekwencji dla predykcji
    sequence_length = config.MODEL['sequence_length']
    
    # Prowizja transakcyjna
    commission_rate = 0.001  # 0.1%
    
    # Przebieg backtestingu
    for i in range(sequence_length, len(df) - 1):
        # Przygotowanie danych do analizy
        df_subset = df.iloc[:i+1].copy()
        
        # Generowanie sygnału dla bieżącego stanu rynku
        
        # 1. Predykcja z modelu
        prediction = analyzer.predict_price_movement(df_subset)
        
        # 2. Analiza techniczna
        ta_signal = analyzer.generate_ta_signal(df_subset)
        
        # 3. Prosty sentyment (losowy dla backtestingu)
        sentiment = np.random.uniform(-0.3, 0.3)
        
        # 4. Połączenie sygnałów w jeden
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
        if position == "LONG":
            signal_value += inertia_factor
        elif position == "SHORT":
            signal_value -= inertia_factor
        
        # Generowanie finalnego sygnału
        if signal_value > config.SIGNALS["long_threshold"]:
            signal = "LONG"
        elif signal_value < config.SIGNALS["short_threshold"]:
            signal = "SHORT"
        else:
            signal = "FLAT"
        
        # Aktualna cena i następna cena (do oceny skuteczności)
        current_price = df['close'].iloc[i]
        next_price = df['close'].iloc[i+1]
        current_timestamp = df['timestamp'].iloc[i]
        
        # Wykonanie transakcji
        if signal != position:
            # Zamykanie istniejącej pozycji
            if position == "LONG":
                # Uwzględnienie prowizji
                closing_price = current_price * (1 - commission_rate)
                profit_pct = (closing_price - entry_price) / entry_price
                profit_amount = balance * position_size * profit_pct
                balance += profit_amount
                
                trades.append({
                    'timestamp': current_timestamp,
                    'type': 'CLOSE_LONG',
                    'price': current_price,
                    'closing_price': closing_price,
                    'profit_pct': profit_pct * 100,
                    'profit_amount': profit_amount,
                    'balance': balance
                })
                
            elif position == "SHORT":
                # Uwzględnienie prowizji
                closing_price = current_price * (1 + commission_rate)
                profit_pct = (entry_price - closing_price) / entry_price
                profit_amount = balance * position_size * profit_pct
                balance += profit_amount
                
                trades.append({
                    'timestamp': current_timestamp,
                    'type': 'CLOSE_SHORT',
                    'price': current_price,
                    'closing_price': closing_price,
                    'profit_pct': profit_pct * 100,
                    'profit_amount': profit_amount,
                    'balance': balance
                })
            
            # Otwieranie nowej pozycji
            if signal == "LONG" or signal == "SHORT":
                # Obliczanie wielkości pozycji zgodnie z zarządzaniem ryzykiem
                position_size = config.RISK["risk_per_trade"]
                
                # Uwzględnienie prowizji
                if signal == "LONG":
                    entry_price = current_price * (1 + commission_rate)
                else:  # SHORT
                    entry_price = current_price * (1 - commission_rate)
                
                position = signal
                
                trades.append({
                    'timestamp': current_timestamp,
                    'type': f'OPEN_{position}',
                    'price': current_price,
                    'entry_price': entry_price,
                    'position_size': position_size,
                    'balance': balance
                })
            else:
                position = "FLAT"
                entry_price = 0
                position_size = 0
        
        # Zapisanie historii balansu
        balance_history.append({
            'timestamp': current_timestamp,
            'balance': balance
        })
    
    # Zamykanie ostatniej pozycji na koniec backtestingu
    if position != "FLAT":
        last_price = df['close'].iloc[-1]
        last_timestamp = df['timestamp'].iloc[-1]
        
        if position == "LONG":
            closing_price = last_price * (1 - commission_rate)
            profit_pct = (closing_price - entry_price) / entry_price
        elif position == "SHORT":
            closing_price = last_price * (1 + commission_rate)
            profit_pct = (entry_price - closing_price) / entry_price
        
        profit_amount = balance * position_size * profit_pct
        balance += profit_amount
        
        trades.append({
            'timestamp': last_timestamp,
            'type': f'CLOSE_{position}',
            'price': last_price,
            'closing_price': closing_price,
            'profit_pct': profit_pct * 100,
            'profit_amount': profit_amount,
            'balance': balance
        })
        
        balance_history.append({
            'timestamp': last_timestamp,
            'balance': balance
        })
    
    # Konwersja list na DataFrame dla łatwiejszej analizy
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    balance_history_df = pd.DataFrame(balance_history)
    
    # Obliczanie statystyk
    stats = {}
    
    if not trades_df.empty:
        # Filtrowanie transakcji zamknięcia
        close_trades = trades_df[trades_df['type'].str.startswith('CLOSE')]
        
        if not close_trades.empty:
            # Podstawowe statystyki
            winning_trades = close_trades[close_trades['profit_amount'] > 0]
            losing_trades = close_trades[close_trades['profit_amount'] <= 0]
            
            total_trades = len(close_trades)
            winning_count = len(winning_trades)
            
            win_rate = winning_count / total_trades if total_trades > 0 else 0
            
            avg_win = winning_trades['profit_pct'].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades['profit_pct'].mean() if not losing_trades.empty else 0
            
            if not losing_trades.empty and losing_trades['profit_amount'].sum() != 0:
                profit_factor = abs(winning_trades['profit_amount'].sum() / losing_trades['profit_amount'].sum())
            else:
                profit_factor = float('inf')
            
            # Maksymalny drawdown
            balance_history_df['peak'] = balance_history_df['balance'].cummax()
            balance_history_df['drawdown'] = (balance_history_df['peak'] - balance_history_df['balance']) / balance_history_df['peak']
            max_drawdown = balance_history_df['drawdown'].max()
            
            # Zwrot całkowity
            total_return = (balance / initial_balance - 1) * 100
            
            # Sharpe Ratio (uproszczony)
            if 'profit_pct' in close_trades.columns:
                returns = close_trades['profit_pct'] / 100  # Konwersja na format dziesiętny
                mean_return = returns.mean()
                std_return = returns.std()
                risk_free_rate = 0.01 / 365  # Założenie: 1% rocznie
                
                sharpe_ratio = (mean_return - risk_free_rate) / std_return * np.sqrt(365) if std_return != 0 else 0
            else:
                sharpe_ratio = 0
            
            # Przygotowanie statystyk
            stats = {
                'total_trades': total_trades,
                'winning_trades': winning_count,
                'losing_trades': total_trades - winning_count,
                'win_rate': win_rate,
                'avg_win_pct': avg_win,
                'avg_loss_pct': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown_pct': max_drawdown * 100,
                'total_return_pct': total_return,
                'sharpe_ratio': sharpe_ratio,
                'final_balance': balance
            }
    
    # Jeśli nie było żadnych transakcji
    if not stats:
        stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_win_pct': 0,
            'avg_loss_pct': 0,
            'profit_factor': 0,
            'max_drawdown_pct': 0,
            'total_return_pct': 0,
            'sharpe_ratio': 0,
            'final_balance': balance
        }
    
    # Przygotowanie wyniku
    backtest_results = {
        'stats': stats,
        'trades': trades_df.to_dict('records') if not trades_df.empty else [],
        'balance_history': balance_history_df.to_dict('records')
    }
    
    return backtest_results

def plot_backtest_results(backtest_results, save_path=None):
    """
    Generowanie wykresu wyników backtestingu.
    
    Args:
        backtest_results (dict): Wyniki backtestingu
        save_path (str, optional): Ścieżka do zapisu wykresu
    """
    # Konwersja danych do DataFrame
    balance_history = pd.DataFrame(backtest_results['balance_history'])
    balance_history['timestamp'] = pd.to_datetime(balance_history['timestamp'])
    
    # Tworzenie figury
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # 1. Wykres balansu
    ax1.plot(balance_history['timestamp'], balance_history['balance'], 
            label='Balance', color='blue', linewidth=2)
    
    ax1.set_title('Backtest Results', fontsize=14)
    ax1.set_ylabel('Balance', fontsize=12)
    ax1.grid(True)
    
    # Dodanie transakcji, jeśli istnieją
    if 'trades' in backtest_results and backtest_results['trades']:
        trades_df = pd.DataFrame(backtest_results['trades'])
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Rysowanie punktów dla transakcji
        for i, trade in trades_df.iterrows():
            if 'type' in trade:
                if trade['type'].startswith('OPEN_LONG'):
                    ax1.scatter(trade['timestamp'], trade['balance'], 
                               s=50, marker='^', color='green', zorder=5)
                elif trade['type'].startswith('OPEN_SHORT'):
                    ax1.scatter(trade['timestamp'], trade['balance'], 
                               s=50, marker='v', color='red', zorder=5)
                elif trade['type'].startswith('CLOSE_LONG'):
                    color = 'green' if trade['profit_amount'] > 0 else 'red'
                    ax1.scatter(trade['timestamp'], trade['balance'], 
                               s=50, marker='o', color=color, zorder=5)
                elif trade['type'].startswith('CLOSE_SHORT'):
                    color = 'green' if trade['profit_amount'] > 0 else 'red'
                    ax1.scatter(trade['timestamp'], trade['balance'], 
                               s=50, marker='o', color=color, zorder=5)
    
    # 2. Wykres drawdownu
    if 'peak' not in balance_history.columns:
        balance_history['peak'] = balance_history['balance'].cummax()
    if 'drawdown' not in balance_history.columns:
        balance_history['drawdown'] = (balance_history['peak'] - balance_history['balance']) / balance_history['peak'] * 100
    
    ax2.fill_between(balance_history['timestamp'], 0, balance_history['drawdown'], 
                     color='red', alpha=0.3)
    ax2.plot(balance_history['timestamp'], balance_history['drawdown'], 
            color='red', linewidth=1)
    
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.invert_yaxis()  # Odwracamy oś y, żeby drawdown był rysowany w dół
    ax2.grid(True)
    
    # Dodanie podsumowania wyników
    stats = backtest_results['stats']
    
    stats_text = (
        f"Total Return: {stats['total_return_pct']:.2f}%\n"
        f"Win Rate: {stats['win_rate']*100:.2f}%\n"
        f"Profit Factor: {stats['profit_factor']:.2f}\n"
        f"Max Drawdown: {stats['max_drawdown_pct']:.2f}%\n"
        f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}\n"
        f"Trades: {stats['total_trades']}"
    )
    
    # Dodanie tekstu do wykresu
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, 
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Zapisanie lub wyświetlenie wykresu
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Wykres zapisany: {save_path}")
    else:
        plt.show()

def save_backtest_results(backtest_results, output_path=None):
    """
    Zapisanie wyników backtestingu do plików.
    
    Args:
        backtest_results (dict): Wyniki backtestingu
        output_path (str, optional): Ścieżka wyjściowa
    """
    if output_path is None:
        output_path = config.RESULTS_DIR
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Zapisanie statystyk
    stats_file = os.path.join(output_path, f"backtest_stats_{timestamp}.csv")
    pd.DataFrame([backtest_results['stats']]).to_csv(stats_file, index=False)
    
    # Zapisanie transakcji
    if backtest_results['trades']:
        trades_file = os.path.join(output_path, f"backtest_trades_{timestamp}.csv")
        pd.DataFrame(backtest_results['trades']).to_csv(trades_file, index=False)
    
    # Zapisanie historii balansu
    balance_file = os.path.join(output_path, f"backtest_balance_{timestamp}.csv")
    pd.DataFrame(backtest_results['balance_history']).to_csv(balance_file, index=False)
    
    print(f"Wyniki backtestingu zapisane w katalogu: {output_path}")

def print_backtest_summary(backtest_results):
    """
    Wyświetlenie podsumowania backtestingu.
    
    Args:
        backtest_results (dict): Wyniki backtestingu
    """
    stats = backtest_results['stats']
    
    print("\n" + "="*50)
    print("WYNIKI BACKTESTINGU")
    print("="*50)
    print(f"Całkowity zwrot:      {stats['total_return_pct']:.2f}%")
    print(f"Liczba transakcji:    {stats['total_trades']}")
    print(f"Wygrane transakcje:   {stats['winning_trades']} ({stats['win_rate']*100:.2f}%)")
    print(f"Przegrane transakcje: {stats['losing_trades']}")
    print(f"Średni zysk:          {stats['avg_win_pct']:.2f}%")
    print(f"Średnia strata:       {stats['avg_loss_pct']:.2f}%")
    print(f"Profit Factor:        {stats['profit_factor']:.2f}")
    print(f"Maksymalny drawdown:  {stats['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio:         {stats['sharpe_ratio']:.2f}")
    print(f"Końcowy kapitał:      ${stats['final_balance']:.2f}")
    print("="*50 + "\n")

def main():
    """Główna funkcja skryptu."""
    args = parse_arguments()
    
    # Przeprowadzenie backtestingu
    print(f"Rozpoczęcie backtestingu od {args.start_date} do {args.end_date}...")
    
    analyzer = MarketAnalyzer()
    
    # Załadowanie modelu
    if args.model:
        analyzer.load_model(args.model)
    
    backtest_results = run_backtest(
        analyzer=analyzer,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.initial_balance
    )
    
    if backtest_results is None:
        print("Błąd podczas backtestingu")
        return
    
    # Wyświetlenie podsumowania
    print_backtest_summary(backtest_results)
    
    # Zapisanie wyników
    save_backtest_results(backtest_results, args.output)
    
    # Generowanie wykresu
    if args.plot:
        plot_path = os.path.join(config.RESULTS_DIR, f"backtest_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plot_backtest_results(backtest_results, save_path=plot_path)

if __name__ == "__main__":
    print("Bitcoin Trading Bot - Backtesting")
    main()
