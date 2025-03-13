"""
Skrypt do trenowania modelu LSTM dla Bitcoin Trading Bot
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import logging
import argparse
from datetime import datetime
import pickle

import config
from analyzer import MarketAnalyzer

logger = logging.getLogger("train")

def parse_arguments():
    """Parsowanie argumentów wiersza poleceń."""
    parser = argparse.ArgumentParser(description='Trenowanie modelu LSTM dla Bitcoin Trading Bot')
    
    parser.add_argument('--epochs', type=int, default=config.MODEL['epochs'],
                        help='Liczba epok treningu')
    parser.add_argument('--batch_size', type=int, default=config.MODEL['batch_size'],
                        help='Rozmiar paczki danych')
    parser.add_argument('--start_date', type=str, default="2020-01-01",
                        help='Data początkowa dla danych treningowych')
    parser.add_argument('--validation_split', type=float, default=config.MODEL['validation_split'],
                        help='Procent danych użyty jako zbiór walidacyjny')
    parser.add_argument('--force', action='store_true',
                        help='Wymuś trening, nawet jeśli istnieje wcześniej wytrenowany model')
    
    return parser.parse_args()

def build_model(input_shape):
    """
    Budowa modelu LSTM.
    
    Args:
        input_shape (tuple): Kształt danych wejściowych
        
    Returns:
        model: Skompilowany model Keras
    """
    model = Sequential()
    
    # Pierwsza warstwa LSTM
    model.add(LSTM(
        units=config.MODEL['lstm_units'][0],
        return_sequences=len(config.MODEL['lstm_units']) > 1,
        input_shape=input_shape
    ))
    model.add(Dropout(config.MODEL['dropout_rate']))
    
    # Dodatkowe warstwy LSTM, jeśli skonfigurowano więcej niż jedną
    for i in range(1, len(config.MODEL['lstm_units'])):
        model.add(LSTM(
            units=config.MODEL['lstm_units'][i],
            return_sequences=i < len(config.MODEL['lstm_units']) - 1
        ))
        model.add(Dropout(config.MODEL['dropout_rate']))
    
    # Warstwa ukryta
    model.add(Dense(units=config.MODEL['lstm_units'][-1] // 2))
    
    # Warstwa wyjściowa
    model.add(Dense(units=1))
    
    # Kompilacja modelu
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.MODEL['learning_rate']),
        loss='mean_squared_error'
    )
    
    # Wyświetlenie podsumowania modelu
    model.summary()
    
    return model

def plot_learning_curves(history):
    """
    Generowanie wykresu krzywych uczenia.
    
    Args:
        history: Historia treningu modelu
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Loss (training)')
    plt.plot(history.history['val_loss'], label='Loss (validation)')
    plt.title('Learning Curves')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Zapisanie wykresu
    plt.savefig(os.path.join(config.RESULTS_DIR, f"learning_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    plt.close()

def train_model():
    """Główna funkcja trenująca model."""
    args = parse_arguments()
    
    # Sprawdzenie czy istnieje wcześniej wytrenowany model
    if not args.force and any(f.endswith('.h5') for f in os.listdir(config.MODELS_DIR)):
        print("Znaleziono istniejące modele. Użyj --force, aby wymusić ponowny trening.")
        response = input("Czy chcesz kontynuować trening mimo to? (t/N): ")
        if response.lower() != 't':
            print("Anulowano trening.")
            return False
    
    # Inicjalizacja analizatora danych
    analyzer = MarketAnalyzer()
    
    # Pobieranie danych historycznych
    print(f"Pobieranie danych historycznych od {args.start_date}...")
    df = analyzer.fetch_market_data(
        timeframe=config.TIMEFRAME,
        limit=100000  # Maksymalna liczba świec
    )
    
    if df.empty:
        print("Błąd: Nie udało się pobrać danych historycznych")
        return False
    
    print(f"Pobrano {len(df)} świec danych")
    
    # Przetwarzanie danych
    print("Przetwarzanie danych...")
    X, y = analyzer.preprocess_data(df)
    
    if X is None or y is None:
        print("Błąd podczas przetwarzania danych")
        return False
    
    # Podział na zbiór treningowy i walidacyjny
    split_idx = int(len(X) * (1 - args.validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Dane treningowe: {X_train.shape}, Dane walidacyjne: {X_val.shape}")
    
    # Budowa i trenowanie modelu
    print("Budowa modelu LSTM...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)
    
    # Konfiguracja callbacków
    callbacks = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    callbacks.append(early_stopping)
    
    # Model checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filepath = os.path.join(config.MODELS_DIR, f'lstm_model_{timestamp}.h5')
    
    model_checkpoint = ModelCheckpoint(
        model_filepath,
        monitor='val_loss',
        save_best_only=True
    )
    callbacks.append(model_checkpoint)
    
    # Trenowanie modelu
    print(f"Rozpoczęcie treningu modelu ({args.epochs} epok)...")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Zapisanie modelu
    model.save(model_filepath)
    print(f"Model zapisany: {model_filepath}")
    
    # Zapisanie scaler'ów
    price_scaler_path = os.path.join(config.MODELS_DIR, f"lstm_model_{timestamp}_price_scaler.pkl")
    features_scaler_path = os.path.join(config.MODELS_DIR, f"lstm_model_{timestamp}_features_scaler.pkl")
    
    with open(price_scaler_path, 'wb') as f:
        pickle.dump(analyzer.scaler_price, f)
    
    with open(features_scaler_path, 'wb') as f:
        pickle.dump(analyzer.scaler_features, f)
    
    print(f"Skalery zapisane: {price_scaler_path}, {features_scaler_path}")
    
    # Generowanie wykresu krzywych uczenia
    plot_learning_curves(history)
    
    # Ocena modelu na zbiorze walidacyjnym
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    print(f"Ocena modelu - MSE na zbiorze walidacyjnym: {val_loss:.6f}")
    
    return True

if __name__ == "__main__":
    print("Bitcoin Trading Bot - Trenowanie modelu LSTM")
    success = train_model()
    
    if success:
        print("Trening zakończony pomyślnie!")
    else:
        print("Trening zakończony niepowodzeniem.")
