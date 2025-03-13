"""
Moduł odpowiedzialny za wysyłanie powiadomień e-mail.
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import logging

import config

logger = logging.getLogger("email_notifier")

class EmailNotifier:
    """
    Klasa do wysyłania powiadomień email o transakcjach.
    """
    
    def __init__(self, sender_email=None, sender_password=None, smtp_server=None, smtp_port=None):
        """
        Inicjalizacja powiadomień email.
        
        Args:
            sender_email (str, optional): Email nadawcy
            sender_password (str, optional): Hasło do email nadawcy
            smtp_server (str, optional): Serwer SMTP
            smtp_port (int, optional): Port serwera SMTP
        """
        # Jeśli nie podano parametrów, używamy wartości z konfiguracji
        self.sender_email = sender_email or config.EMAIL_NOTIFICATION.get('sender_email')
        self.sender_password = sender_password or config.EMAIL_NOTIFICATION.get('sender_password')
        self.smtp_server = smtp_server or config.EMAIL_NOTIFICATION.get('smtp_server')
        self.smtp_port = smtp_port or config.EMAIL_NOTIFICATION.get('smtp_port', 587)
        self.recipient_email = config.EMAIL_NOTIFICATION.get('recipient_email', config.LIVE_TRADING.get('notification_email'))
    
    def send_trade_notification(self, trade_action, market_position, symbol, price=None, details=None):
        """
        Wysyła powiadomienie email o wykonanej transakcji.
        
        Args:
            trade_action (str): Rodzaj akcji (np. "OPEN_LONG", "CLOSE_SHORT")
            market_position (str): Pozycja rynkowa ("LONG", "SHORT", "FLAT")
            symbol (str): Symbol instrumentu
            price (float, optional): Cena transakcji
            details (dict, optional): Dodatkowe szczegóły transakcji
            
        Returns:
            bool: True jeśli wysłano pomyślnie, False w przypadku błędu
        """
        try:
            # Sprawdzenie czy konfiguracja email jest dostępna
            if not all([self.sender_email, self.sender_password, self.smtp_server, self.recipient_email]):
                logger.warning("Brak pełnej konfiguracji email. Powiadomienie nie zostanie wysłane.")
                return False
            
            # Przygotowanie tematu emaila
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            subject = f"Trading Alert: {market_position} position on {symbol} at {timestamp}"
            
            # Przygotowanie treści emaila
            body = f"""
            <html>
            <body>
                <h2>Bitcoin Trading Bot - Alert Transakcji</h2>
                <p><strong>Czas:</strong> {timestamp}</p>
                <p><strong>Akcja:</strong> {trade_action}</p>
                <p><strong>Pozycja:</strong> {market_position}</p>
                <p><strong>Symbol:</strong> {symbol}</p>
            """
            
            if price:
                body += f"<p><strong>Cena:</strong> {price:.2f} USD</p>"
            
            if details:
                body += "<h3>Szczegóły:</h3><ul>"
                for key, value in details.items():
                    if key not in ['timestamp', 'action', 'symbol']:
                        body += f"<li><strong>{key}:</strong> {value}</li>"
                body += "</ul>"
            
            body += """
                <p>---<br>
                Wiadomość wysłana automatycznie przez Bitcoin Trading Bot.</p>
            </body>
            </html>
            """
            
            # Przygotowanie wiadomości
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = self.recipient_email
            
            # Dodanie treści HTML
            html_part = MIMEText(body, "html")
            message.attach(html_part)
            
            # Wysłanie emaila
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.recipient_email, message.as_string())
            
            logger.info(f"Powiadomienie email wysłane do {self.recipient_email}: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Błąd podczas wysyłania powiadomienia email: {e}")
            return False
