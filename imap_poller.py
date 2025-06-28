import asyncio
import asyncpg
import email
import imaplib
import httpx
import socket
import ssl
import re
import time
import os
import base64
import uuid
import logging
from email.header import decode_header
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import hashlib
import json
import signal
import sys
import threading
from functools import wraps

# Using environment variables directly from the system

def retry_imap_operation(max_retries=3, initial_delay=1, backoff=2):
    """
    Decorator to retry IMAP operations with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff: Multiplier for delay between retries
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            current_delay = initial_delay
            
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except (imaplib.IMAP4.abort, ConnectionError, TimeoutError) as e:
                    retries += 1
                    if retries >= max_retries:
                        print(f"❌ Max retries ({max_retries}) reached for {func.__name__}. Last error: {str(e)}")
                        raise
                    
                    print(f"⚠️  Attempt {retries}/{max_retries} failed: {str(e)}. Retrying in {current_delay} seconds...")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
                except Exception as e:
                    print(f"❌ Unexpected error in {func.__name__}: {str(e)}")
                    raise
        return wrapper
    return decorator

# Constants
MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024  # 10MB
ATTACHMENTS_DIR = 'attachments'

# Create attachments directory if it doesn't exist
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)

# Print current working directory for debugging
print(f"Current working directory: {os.getcwd()}")

# Debug: Verify environment variables are set
print("\n🔍 Checking required environment variables:")
print(f"DJANGO_SECRET_KEY is set: {'✅' if os.getenv('DJANGO_SECRET_KEY') else '❌'}")
print(f"DATABASE_URL is set: {'✅' if os.getenv('DATABASE_URL') else '❌'}")


# Database configuration from environment variables
# Using DATABASE_URL as the primary configuration source
# Fallback to individual DB_* variables for backward compatibility

def get_db_config():
    """Get database configuration from environment variables."""
    database_url = os.getenv("DATABASE_URL")
    
    if database_url:
        # If DATABASE_URL is provided, parse it
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        
        # Parse the URL to extract components
        from urllib.parse import urlparse, parse_qs
        result = urlparse(database_url)
        
        # Extract query parameters
        query = parse_qs(result.query)
        
        # Build config from URL
        config = {
            "user": result.username or "postgres",
            "password": result.password or "",
            "database": result.path[1:] if result.path else "railway",  # Remove leading '/'
            "host": result.hostname or "localhost",
            "port": result.port or 5432,
            "min_size": int(os.getenv("DB_POOL_MIN_SIZE", "1")),
            "max_size": int(os.getenv("DB_POOL_MAX_SIZE", "5")),
            "max_queries": int(os.getenv("DB_MAX_QUERIES", "50000")),
            "max_inactive_connection_lifetime": int(os.getenv("DB_MAX_INACTIVE_CONN_LIFETIME", "300")),
            "timeout": float(os.getenv("DB_TIMEOUT", "10.0")),
            "command_timeout": int(os.getenv("DB_COMMAND_TIMEOUT", "10")),
            "dsn": database_url
        }
        
        # Add SSL mode if specified
        if 'sslmode' in query:
            config['ssl'] = query['sslmode'][0]
            
        return config
    else:
        # Fallback to individual variables for backward compatibility
        required_vars = ["DB_PASSWORD", "DB_HOST"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return {
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_NAME", "railway"),
            "host": os.getenv("DB_HOST"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "command_timeout": int(os.getenv("DB_COMMAND_TIMEOUT", "10")),
            "min_size": int(os.getenv("DB_POOL_MIN_SIZE", "1")),
            "max_size": int(os.getenv("DB_POOL_MAX_SIZE", "5")),
            "max_inactive_connection_lifetime": int(os.getenv("DB_MAX_INACTIVE_CONN_LIFETIME", "300")),
            "max_queries": int(os.getenv("DB_MAX_QUERIES", "50000")),
            "timeout": float(os.getenv("DB_TIMEOUT", "10.0")),
            "ssl": os.getenv("DB_SSL", "require")
        }

# Initialize db_config
db_config = get_db_config()

# Monitoring configuration
MONITORING = {
    'max_retention_days': 7,  # How long to keep processed message IDs
    'cleanup_interval': 3600,  # Clean up every hour
    'max_processing_time': 300,  # Max seconds to process a single email
    'metrics_interval': 300,  # Log metrics every 5 minutes
    'max_errors_per_minute': 10,  # Maximum errors per minute before throttling
    'max_emails_per_minute': 100  # Maximum emails to process per minute
}

# Database schema
DB_SCHEMA = """
-- Table to track processed messages
CREATE TABLE IF NOT EXISTS processed_messages (
    id BIGSERIAL PRIMARY KEY,
    message_id TEXT NOT NULL,
    email TEXT NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    spam_score INTEGER,
    folder TEXT,
    UNIQUE(message_id, email)
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_processed_messages_message_id ON processed_messages(message_id);
CREATE INDEX IF NOT EXISTS idx_processed_messages_email ON processed_messages(email);
CREATE INDEX IF NOT EXISTS idx_processed_messages_processed_at ON processed_messages(processed_at);

-- Table for error tracking
CREATE TABLE IF NOT EXISTS email_processing_errors (
    id BIGSERIAL PRIMARY KEY,
    email TEXT,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    message_id TEXT,
    folder TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Table for monitoring metrics
CREATE TABLE IF NOT EXISTS email_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_name TEXT NOT NULL,
    metric_value FLOAT NOT NULL,
    email TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

N8N_WEBHOOK_URL = "https://shri003.app.n8n.cloud/webhook/email-in/"  # Replace with actual

# Decode email subject safely
def decode_mime_words(s):
    decoded = decode_header(s)
    return ''.join([str(t[0], t[1] or 'utf-8') if isinstance(t[0], bytes) else t[0] for t in decoded])

def analyze_email_content(text: str) -> dict:
    """Analyze email content for spam indicators."""
    if not text:
        return {
            'word_count': 0,
            'spam_indicators': {},
            'spam_score': 0
        }
        
    text_lower = text.lower()
    
    # Common spam indicators
    spam_phrases = [
        'earn money', 'make money', 'work from home', 'risk-free',
        'buy now', 'order now', 'limited time', 'act now',
        'satisfaction guaranteed', 'click here', 'visit our website',
        'winner', 'congratulations', 'you have won', 'free',
        'credit card', 'password', 'account', 'verify', 'login',
        'urgent', 'immediate response', 'no cost', 'no fees',
        'dear friend', 'dear valued customer', 'dear account holder',
        'no medical exam', 'special promotion', 'one time',
        'this isn\'t spam', 'this is not spam', 'remove in subject',
        'viagra', 'cialis', 'pills', 'pharmacy', 'medication',
        'investment', 'stock alert', 'penny stock', 'make $', 'earn $',
        '$$$', '!!!', '???', '***', '___',
    ]
    
    # Count spam indicators
    spam_indicators = {
        'has_spam_phrases': any(phrase in text_lower for phrase in spam_phrases),
        'has_excessive_punctuation': sum(1 for c in text if c in '!?') > 5,
        'has_all_caps_ratio': sum(1 for c in text if c.isupper()) / max(1, len(text)) > 0.3,
        'has_suspicious_links': len(re.findall(r'https?://[^\s<>"\']+', text)) > 3,
        'has_short_body': len(text.split()) < 10,
        'has_suspicious_chars': bool(re.search(r'[\x80-\xFF]', text)),
        'has_js_events': bool(re.search(r'on(load|click|submit|mouseover)="', text, re.I)),
        'has_invisible_text': bool(re.search(r'display:\s*none|visibility:\s*hidden', text, re.I))
    }
    
    # Calculate spam score (0-100)
    spam_score = sum(10 for v in spam_indicators.values() if v)
    
    return {
        'word_count': len(text.split()),
        'char_count': len(text),
        'spam_indicators': spam_indicators,
        'spam_score': min(spam_score, 100)
    }

def is_spam(msg) -> dict:
    """Check if an email is likely spam based on headers, subject, and content."""
    # Extract headers
    headers = {}
    for key, value in msg.items():
        headers[key] = value
    
    # Check common spam headers
    spam_headers = {
        'X-Spam-Flag': lambda x: x.lower() == 'yes',
        'X-Spam-Status': lambda x: 'yes' in x.lower(),
        'X-Spam-Score': lambda x: float(x.split()[0]) > 5.0 if x.replace('.', '').isdigit() else False,
        'Precedence': lambda x: x.lower() in ['bulk', 'junk', 'list'],
        'X-Auto-Response-Suppress': lambda x: True,
        'Auto-Submitted': lambda x: x.lower() != 'no',
        'List-Unsubscribe': lambda x: True,
        'X-CMAE-Envelope': lambda x: True,
        'X-Failed-Recipients': lambda x: True,
        'X-Mailer': lambda x: 'mass' in x.lower() or 'bulk' in x.lower()
    }
    
    # Analyze email content
    body = extract_email_body(msg)
    content_analysis = analyze_email_content(body)
    
    # Check subject for common spam indicators
    subject = headers.get('Subject', '')
    spam_subject_indicators = [
        'earn money', 'make money', 'work from home', 'risk-free',
        'buy now', 'order now', 'limited time', 'act now',
        'satisfaction guaranteed', 'click here', 'visit our website',
        'winner', 'congratulations', 'you have won', 'free',
        'credit card', 'password', 'account', 'verify', 'login',
        'urgent', 'immediate response', 'no cost', 'no fees',
        'dear friend', 'dear valued customer', 'dear account holder',
        'no medical exam', 'special promotion', 'one time',
        'this isn\'t spam', 'this is not spam', 'remove in subject',
        'viagra', 'cialis', 'pills', 'pharmacy', 'medication',
        'investment', 'stock alert', 'penny stock', 'make $', 'earn $',
        '$$$', '!!!', '???', '***', '___',
    ]
    
    # Calculate header-based spam score
    header_score = 0
    for header, check in spam_headers.items():
        if header in headers and check(headers[header]):
            header_score += 10  # 10 points per spam header
    
    # Check subject for spam indicators
    subject_score = 0
    subject_lower = subject.lower()
    if any(indicator in subject_lower for indicator in spam_subject_indicators):
        subject_score = 30  # 30 points for spammy subject
    
    # Check for suspicious sender patterns
    sender_score = 0
    from_header = headers.get('From', '').lower()
    if not from_header or '@' not in from_header:
        sender_score += 20  # 20 points for missing or invalid From
    elif 'noreply' in from_header or 'no-reply' in from_header:
        sender_score += 10  # 10 points for no-reply addresses
    
    # Calculate total spam score (0-100)
    total_score = min(
        header_score + 
        subject_score + 
        sender_score + 
        content_analysis['spam_score'],
        100  # Cap at 100
    )
    
    # Prepare result
    result = {
        'is_spam': total_score >= 30,  # Threshold for considering as spam
        'score': total_score,
        'categories': {
            'headers': header_score,
            'subject': subject_score,
            'sender': sender_score,
            'content': content_analysis['spam_score']
        },
        'indicators': {
            'subject_indicators': [i for i in spam_subject_indicators if i in subject_lower],
            'content_indicators': [k for k, v in content_analysis['spam_indicators'].items() if v],
            'header_indicators': [k for k, v in spam_headers.items() 
                               if k in headers and v(headers[k])]
        },
        'analysis': {
            'subject': subject,
            'from': headers.get('From', ''),
            'to': headers.get('To', ''),
            'date': headers.get('Date', ''),
            'message_id': headers.get('Message-ID', ''),
            'content_analysis': content_analysis
        }
    }
    
    return result

def get_email_headers(msg) -> Dict[str, str]:
    """Extract all email headers into a dictionary."""
    headers = {}
    for key, value in msg.items():
        if isinstance(value, str):
            headers[key] = decode_mime_words(value)
    return headers

def extract_email_body(message) -> str:
    """Extract email body safely with improved handling of different content types."""
    body = ""
    
    if message.is_multipart():
        # Look for alternative parts first (plain text)
        for part in message.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get('Content-Disposition', '')).lower()
            
            # Skip attachments
            if 'attachment' in content_disposition:
                continue
                
            # Prefer plain text over HTML
            if content_type == 'text/plain':
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        body = payload.decode(charset, errors='replace')
                        break
                except Exception:
                    continue
        
        # If no plain text found, try HTML
        if not body:
            for part in message.walk():
                if part.get_content_type() == 'text/html':
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or 'utf-8'
                            html_content = payload.decode(charset, errors='replace')
                            # Simple HTML to text conversion
                            body = re.sub(r'<[^>]+>', ' ', html_content)
                            body = re.sub(r'\s+', ' ', body).strip()
                            break
                    except Exception:
                        continue
    else:
        # Not multipart, just get the payload
        try:
            payload = message.get_payload(decode=True)
            if payload:
                charset = message.get_content_charset() or 'utf-8'
                body = payload.decode(charset, errors='replace')
        except Exception:
            pass
    
    return body or "[No message body]"

async def get_db_pool():
    """Create a connection pool for database operations with proper configuration."""
    # Get database configuration
    config = db_config.copy()
    
    # Use DSN if available (from DATABASE_URL), otherwise use individual parameters
    dsn = config.pop('dsn', None)
    
    # Remove None values and convert to asyncpg expected format
    pool_config = {k: v for k, v in config.items() if v is not None}
    
    # Create connection pool
    pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=pool_config.get('min_size', 1),
        max_size=pool_config.get('max_size', 10),
        max_queries=pool_config.get('max_queries', 50000),
        max_inactive_connection_lifetime=pool_config.get('max_inactive_connection_lifetime', 300.0),
        command_timeout=pool_config.get('command_timeout'),
        ssl=pool_config.get('ssl')
    )
    
    # Initialize database schema
    async with pool.acquire() as conn:
        await conn.execute(DB_SCHEMA)
        
    return pool

async def get_last_processed_date(pool, email_address: str) -> Optional[datetime]:
    """Get the latest processed date for an email address from the database."""
    try:
        async with pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT MAX(processed_at) 
                FROM processed_messages 
                WHERE email = $1
                """,
                email_address
            )
            return result
    except Exception as e:
        print(f"Error getting last processed date: {e}")
        return None

async def is_duplicate(pool, message_id: str, email_address: str) -> bool:
    """Check if a message has already been processed."""
    if not message_id:
        return False
        
    async with pool.acquire() as conn:
        result = await conn.fetchval(
            """
            SELECT 1 FROM processed_messages 
            WHERE message_id = $1 AND email = $2
            """,
            message_id, email_address
        )
        return bool(result)

async def mark_as_processed(pool, message_id: str, email_address: str) -> None:
    """Mark a message as processed."""
    if not message_id:
        return
        
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO processed_messages (message_id, email, processed_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (message_id, email) DO NOTHING
            """,
            message_id, email_address
        )

async def log_error(pool, email: str, error_type: str, error_message: str, 
                  message_id: str = None, folder: str = None) -> None:
    """Log an error to the database."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO email_processing_errors 
                (email, error_type, error_message, message_id, folder)
                VALUES ($1, $2, $3, $4, $5)
                """,
                email, error_type, str(error_message)[:1000], message_id, folder
            )
    except Exception as e:
        print(f"Failed to log error: {e}")

async def log_metric(pool, metric_name: str, metric_value: float, email: str = None) -> None:
    """Log a metric to the database."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO email_metrics 
                (metric_name, metric_value, email)
                VALUES ($1, $2, $3)
                """,
                metric_name, float(metric_value), email
            )
    except Exception as e:
        print(f"Failed to log metric: {e}")

async def get_error_count(pool, email: str, minutes: int = 1) -> int:
    """Get the number of errors for an email in the last N minutes."""
    try:
        return await pool.fetchval(
            """
            SELECT COUNT(*) 
            FROM email_processing_errors 
            WHERE email = $1 AND created_at > NOW() - ($2 * INTERVAL '1 minute')
            """,
            email, minutes
        ) or 0
    except Exception as e:
        print(f"Failed to get error count: {e}")
        return 0

async def is_rate_limited(pool, email: str) -> bool:
    """Check if we should rate limit processing for an email."""
    try:
        # Check errors in the last minute
        recent_errors = await get_error_count(pool, email, 1)
        if recent_errors > MONITORING['max_errors_per_minute']:
            return True
            
        # Check email volume in the last minute
        recent_emails = await pool.fetchval(
            """
            SELECT COUNT(*) 
            FROM processed_messages 
            WHERE email = $1 AND processed_at > NOW() - INTERVAL '1 minute'
            """,
            email
        ) or 0
        
        return recent_emails > MONITORING['max_emails_per_minute']
        
    except Exception as e:
        print(f"Rate limit check failed: {e}")
        return False

async def cleanup_old_records(pool) -> None:
    """Remove old processed message records."""
    async with pool.acquire() as conn:
        await conn.execute(
            f"""
            DELETE FROM processed_messages 
            WHERE processed_at < NOW() - INTERVAL '{MONITORING['max_retention_days']} days'
            """
        )

def get_fernet_key():
    """
    Generate Fernet key using the same method as Django's default encryption.
    Uses DJANGO_SECRET_KEY from environment variables.
    """
    secret_key = os.getenv('DJANGO_SECRET_KEY')
    if not secret_key:
        raise ValueError("DJANGO_SECRET_KEY environment variable is not set")
    
    salt = secret_key[:16].encode()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(secret_key.encode()))
    return key

def decrypt_fernet(encrypted_data: str) -> str:
    """
    Decrypt a password that was encrypted using the same key generation method.
    
    Args:
        encrypted_data: The encrypted password string (starts with 'gAA' or 'ENC:')
        
    Returns:
        str: The decrypted password
        
    Raises:
        ValueError: If decryption fails or password is not encrypted
    """
    if not encrypted_data:
        raise ValueError("No password provided")
        
    # Remove 'ENC:' prefix if present
    if encrypted_data.startswith('ENC:'):
        encrypted_data = encrypted_data[4:]
    
    # If it doesn't look like an encrypted password, return as-is
    if not encrypted_data.startswith('gAA'):
        return encrypted_data
    
    try:
        f = Fernet(get_fernet_key())
        decrypted = f.decrypt(encrypted_data.encode())
        return decrypted.decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to decrypt password: {str(e)}")

async def fetch_credentials(pool):
    """Fetch email credentials from the credentials_email table with provider-specific settings.
    
    Expects password to be encrypted with Fernet in the format:
    - ENC:encrypted_data
    
    The DJANGO_SECRET_KEY environment variable must be set for decryption.
    """
    print("\n=== Fetching email credentials ===")
    try:
        # Get the Fernet key from DJANGO_SECRET_KEY
        fernet_key = get_fernet_key()
            
        # Query to get all email accounts with their provider settings
        # Using the actual column names from the database schema
        query = """
            SELECT 
                account_id as id,
                email,
                password_encrypted as password,
                provider_type as provider,
                imap_host,
                imap_port,
                imap_use_ssl,
                smtp_host,
                smtp_port,
                smtp_use_ssl,
                smtp_use_tls,
                is_active,
                created_at,
                updated_at
            FROM credentials_email
            WHERE password_encrypted IS NOT NULL
            AND password_encrypted != ''
            AND is_active = true
        """
        
        print("Executing query to fetch email accounts...")
        records = await pool.fetch(query)
        print(f"Found {len(records)} email accounts in the database")
        
        if not records:
            print("No active email accounts found in the database.")
            return []
            
        # Convert records to list of dicts
        credentials = []
        for record in records:
            try:
                # Get the password and check if it's encrypted
                password = record['password']
                is_encrypted = False
                
                # Check if password is encrypted (with or without 'ENC:' prefix)
                if isinstance(password, str):
                    if password.startswith('ENC:'):
                        # Handle legacy format with 'ENC:' prefix
                        print(f"🔑 Decrypting password for {record['email']}")
                        encrypted_data = password[4:]  # Remove 'ENC:' prefix
                        is_encrypted = True
                    elif password.startswith('gAAAA'):
                        # Handle encrypted password without prefix
                        print(f"🔑 Decrypting password for {record['email']}")
                        encrypted_data = password
                        is_encrypted = True
                
                if is_encrypted:
                    try:
                        # Get the Fernet key and decrypt the password
                        fernet_key = get_fernet_key()
                        print(f"   🔑 Attempting to decrypt password for {record['email']}")
                        print(f"   🔑 Encrypted data length: {len(encrypted_data)} characters")
                        print(f"   🔑 Encrypted data starts with: {encrypted_data[:10]}...")
                        password = decrypt_fernet(encrypted_data)
                        print(f"   ✅ Password decrypted successfully for {record['email']}")
                        
                    except Exception as e:
                        print(f"Error: Failed to decrypt password for {record['email']}: {type(e).__name__} - {str(e)}")
                        continue
                
                cred = {
                    'email': record['email'],
                    'password': password,
                    'imap': {
                        'host': record['imap_host'],
                        'port': record['imap_port'],
                        'use_ssl': record['imap_use_ssl'],
                        'secure': record['imap_secure']
                    },
                    'smtp': {
                        'host': record['smtp_host'],
                        'port': record['smtp_port'],
                        'use_ssl': record['smtp_use_ssl'],
                        'use_tls': record['smtp_use_tls']
                    },
                    'provider': record['provider']
                }
                
                # Add optional fields if they exist
                if 'name' in record and record['name'] is not None:
                    cred['name'] = record['name']
                if 'notes' in record and record['notes'] is not None:
                    cred['notes'] = record['notes']
                    
                credentials.append(cred)
                
            except Exception as e:
                print(f"Error processing credentials for {record.get('email', 'unknown')}: {str(e)}")
                continue
                
        return credentials
        
    except Exception as e:
        print(f"Error fetching credentials: {str(e)}")
        return []

async def process_email_folder(mail, folder: str, email_address: str, pool, max_age_days: int = 30) -> int:
    """Process emails in a specific folder.
    
{{ ... }}
    Args:
        mail: IMAP4_SSL connection
        folder: Email folder to process
        email_address: Email address being processed
        pool: Database connection pool
        max_age_days: Maximum age of emails to process (default: 30)
        
    Returns:
        int: Number of errors encountered
    """
    error_count = 0  # Initialize error counter
    
    try:
        # Get the last processed date from the database
        last_processed = await get_last_processed_date(pool, email_address)
        
        # Default to max_age_days if no last_processed date
        if last_processed:
            since_date = last_processed.strftime('%d-%b-%Y')
            print(f"   📅 Processing emails since last processed date: {since_date}")
        else:
            # Fallback to max_age_days if no last_processed date
            try:
                max_days = int(max_age_days)
            except (TypeError, ValueError):
                max_days = 30  # Default to 30 days if invalid
            since_date = (datetime.now(timezone.utc) - timedelta(days=max_days)).strftime('%d-%b-%Y')
            print(f"   ⏳ No previous processed date found, using last {max_days} days")
            
        # Select the folder
        print(f"   Checking {folder}...")
        try:
            status, _ = mail.select(folder, readonly=True)
            if status != 'OK':
                print(f"   ❌ Could not select {folder} (status: {status})")
                return error_count + 1
        except Exception as e:
            error_msg = f"Error selecting {folder}: {str(e)}"
            print(f"   ❌ {error_msg}")
            await log_error(pool, email_address, 'folder_error', error_msg, folder=folder)
            return error_count + 1
        
        try:
            # Search for unread messages since the last processed date
            search_criteria = f'(SINCE "{since_date}" UNSEEN)'
            print(f"   🔍 Search criteria: {search_criteria}")
            
            status, messages = mail.search(None, search_criteria)
            
            if status != 'OK' or not messages or not messages[0]:
                print(f"   No unread messages in {folder}")
                return
                
            mail_ids = messages[0].split()
            if not mail_ids:
                print(f"   No unread messages in {folder}")
                return
                
            print(f"   Found {len(mail_ids)} unread message(s) in {folder}")
            
            # Process each email
            for num in mail_ids:
                try:
                    await process_single_email(mail, num, email_address, folder, pool)
                except Exception as e:
                    error_count += 1
                    error_msg = f"Error processing message {num} in {folder}: {str(e)}"
                    print(f"   ❌ {error_msg}")
                    await log_error(pool, email_address, 'message_processing_error', error_msg, folder=folder)
                    
                    # If we hit too many errors, stop processing this folder
                    if error_count >= 5:  # Adjust this threshold as needed
                        print(f"   ⚠️  Too many errors ({error_count}), stopping processing of {folder}")
                        break
                
        except Exception as e:
            error_msg = f"Error processing messages in {folder}: {str(e)}"
            print(f"   ❌ {error_msg}")
            await log_error(pool, email_address, 'folder_processing_error', error_msg, folder=folder)
            
    except Exception as e:
        error_msg = f"Unexpected error in process_email_folder: {str(e)}"
        print(f"   ❌ {error_msg}")
        await log_error(pool, email_address, 'unexpected_error', error_msg, folder=folder)
    
    return error_count  # Return the error count for monitoring

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, socket.timeout, asyncio.TimeoutError))
)
async def send_to_webhook(client: httpx.AsyncClient, url: str, payload: dict) -> None:
    """Send data to webhook with retry logic."""
    response = await client.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    return response

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to be filesystem-safe."""
    # Remove non-printable and special characters
    filename = re.sub(r'[^\x20-\x7E]', '', str(filename))
    # Replace spaces and dots with underscores
    filename = re.sub(r'[\s.]+', '_', filename)
    # Remove any remaining unsafe characters
    return re.sub(r'[\\/:*?"<>|]', '', filename)

def save_attachments(msg, message_id: str) -> List[Dict[str, Any]]:
    """Save email attachments to disk and return metadata."""
    attachments = []
    
    # Sanitize message_id for use in filenames
    safe_message_id = sanitize_filename(message_id.strip('<>'))
    
    for part in msg.walk():
        if part.get_content_maintype() == 'multipart' or part.get('Content-Disposition') is None:
            continue
            
        filename = part.get_filename()
        if not filename:
            continue
            
        # Clean filename
        filename = sanitize_filename(filename)
        filepath = os.path.join(ATTACHMENTS_DIR, f"{safe_message_id}_{filename}")
        
        try:
            # Check attachment size
            payload = part.get_payload(decode=True)
            if len(payload) > MAX_ATTACHMENT_SIZE:
                print(f"   ⚠️ Attachment {filename} exceeds size limit, skipping")
                continue
                
            # Save attachment
            with open(filepath, 'wb') as f:
                f.write(payload)
                
            attachments.append({
                'filename': filename,
                'path': filepath,
                'size': len(payload),
                'content_type': part.get_content_type()
            })
            
        except Exception as e:
            print(f"   ❌ Error saving attachment {filename}: {str(e)}")
            continue
            
    return attachments

async def process_single_email(mail, num: str, email_address: str, folder: str, pool) -> None:
    """Process a single email message with enhanced error handling and retries.
    
    Args:
        mail: IMAP4_SSL connection
        num: Message number/ID
        email_address: Email address being processed
        folder: Folder containing the message
        pool: Database connection pool
    """
    start_time = time.time()  # Track processing start time
    error_count = 0  # Initialize error count for this message
    
    try:
        # Fetch the email with retry logic
        for attempt in range(3):
            try:
                status, msg_data = mail.fetch(num, '(RFC822 BODY.PEEK[] FLAGS)')
                if status == 'OK' and msg_data and msg_data[0]:
                    break
            except (imaplib.IMAP4.error, socket.timeout) as e:
                if attempt == 2:  # Last attempt
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        else:
            print(f"   ❌ Could not fetch message {num} after 3 attempts")
            return
            
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)
        
        # Extract headers and basic info
        headers = get_email_headers(msg)
        message_id = headers.get('Message-ID', str(uuid.uuid4()))
        sender = headers.get('From', 'Unknown Sender')
        subject = headers.get('Subject', '(No Subject)')
        date = headers.get('Date', datetime.now(timezone.utc).isoformat())
        
        # Check if this is spam and analyze content
        spam_analysis = is_spam(msg)
        is_spam_message = spam_analysis['is_spam']
        
        # Check if we've already processed this message
        if await is_duplicate(pool, message_id, email_address):
            print(f"   ⏩ Skipping duplicate message: {message_id}")
            return
        
        # Extract body and attachments
        body = extract_email_body(msg)
        attachments = save_attachments(msg, message_id)
        
        # Prepare payload for webhook
        current_time = datetime.now(timezone.utc)
        payload = {
            "user_email": email_address,
            "sender": sender,
            "subject": subject,
            "body": body,
            "timestamp": date,
            "is_spam": is_spam_message,
            "spam_analysis": spam_analysis,
            "folder": folder,
            "message_id": message_id,
            "attachments": [{
                'filename': a['filename'],
                'size': a['size'],
                'content_type': a['content_type']
            } for a in attachments],
            "headers": {k: v for k, v in headers.items() if not k.lower().startswith('x-')},
            "metadata": {
                "processing_time": current_time.isoformat(),
                "attachment_count": len(attachments),
                "processing_host": socket.gethostname(),
                "processing_duration": time.time() - start_time
            }
        }
        
        # Mark as processed in database
        await mark_as_processed(pool, message_id, email_address)
        
        # Only send non-spam emails to webhook
        if not is_spam_message:
            try:
                # Send to webhook with retry logic
                async with httpx.AsyncClient() as client:
                    await send_to_webhook(client, N8N_WEBHOOK_URL, payload)
                print(f"   ✅ Forwarded to webhook: {message_id}")
            except Exception as e:
                error_msg = f"Failed to send to webhook: {str(e)}"
                print(f"   ❌ {error_msg}")
                await log_error(pool, email_address, 'webhook_error', error_msg, message_id, folder)
        else:
            print(f"   ⏩ Skipped spam email (score: {spam_analysis['score']}): {message_id}")
            
        # Mark email as read after processing
        try:
            mail.store(num, '+FLAGS', '\\Seen')
            print(f"   ✅ Marked message as read: {message_id}")
        except Exception as e:
            error_msg = f"Could not mark message as read: {str(e)}"
            print(f"   ⚠️ {error_msg}")
            await log_error(pool, email_address, 'mark_read_error', error_msg, message_id, folder)
        
        print(f"   ✅ Processed message from {sender} "
              f"({'SPAM' if is_spam_message else 'Inbox'}, "
              f"{len(attachments)} attachments, score: {spam_analysis['score']})")
        
    except httpx.HTTPStatusError as e:
        print(f"   ❌ HTTP error sending to webhook: {e.response.status_code} - {e.response.text}")
    except (socket.timeout, asyncio.TimeoutError):
        print("   ⏱️  Request timed out")
    except imaplib.IMAP4.error as e:
        error_msg = f"IMAP Error: {str(e)}"
        print(f"   ❌ {error_msg}")
        await log_error(pool, email_address, 'imap_error', error_msg)
        error_count += 1
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"   ❌ {error_msg}")
        await log_error(pool, email_address, 'unexpected_error', error_msg)
        error_count += 1
    finally:
        # Clean up any partial downloads
        pass

@retry_imap_operation(max_retries=3, initial_delay=1, backoff=2)
async def process_inbox(cred, pool):
    """Process emails for a single account with database connection pool.
    
    Args:
        cred: Dictionary containing email credentials and connection details
        pool: Database connection pool
    """
    email_address = cred.get('email')
    if not email_address:
        logging.error("No email address provided in credentials")
        return 1
        
    error_count = 0
    
    # Check IMAP configuration
    if not cred.get('imap_host') or not cred.get('imap_port'):
        logging.error(f"Missing IMAP configuration for {email_address}")
        await log_error(pool, email_address, 'configuration_error', 
                       'Missing IMAP host or port in configuration')
        return 1
    
    # Check if we should skip due to rate limiting
    if await is_rate_limited(pool, email_address):
        logging.warning(f"Rate limiting active for {email_address}. Skipping this cycle.")
        return 0

    mail = None
    start_time = time.time()
    processed_count = 0
    error_count = 0
    last_metrics_time = time.time()
    
    # Log the connection attempt
    print(f"\n🔍 Processing email: {email_address}")
    print(f"   Connecting to {host}:{port} (SSL: {use_ssl})...")
    
    # Check rate limiting
    if await is_rate_limited(pool, email_address):
        print(f"   ⚠️  Rate limited for {email_address}, skipping this cycle")
        await log_metric(pool, 'rate_limit_hit', 1, email_address)
        return

    try:
        print("\n🔍 Processing email:")
        print(email_address)
        print(f"   Connecting to {host}:{port}...")
        
        # Connect to IMAP server with enhanced error handling
        try:
            # Create appropriate IMAP connection based on SSL setting
            if use_ssl:
                # Create SSL context with modern security settings
                ssl_context = ssl.create_default_context()
                ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
                ssl_context.set_ciphers('DEFAULT@SECLEVEL=2')
                mail = imaplib.IMAP4_SSL(host, port, timeout=15, ssl_context=ssl_context)
            else:
                # For non-SSL connections (not recommended)
                mail = imaplib.IMAP4(host, port)
                # Upgrade to SSL/TLS if server supports STARTTLS
                mail.starttls(ssl_context=ssl.create_default_context())
            
            print(f"   ✅ Connected to {host}:{port} (SSL: {use_ssl})")
            
            # Log server capabilities for debugging
            try:
                status, capabilities = mail.capability()
                if status == 'OK' and capabilities:
                    # Handle both string and bytes in capabilities
                    if isinstance(capabilities, list) and capabilities and isinstance(capabilities[0], bytes):
                        capabilities = [cap.decode('utf-8', errors='replace') for cap in capabilities]
                    print(f"   🔧 Server capabilities: {' '.join(capabilities) if capabilities else 'None'}")
            except Exception as cap_error:
                print(f"   ⚠️  Could not retrieve server capabilities: {str(cap_error)}")
                
            # Attempt login with better error reporting
            print(f"   🔐 Attempting login as {email_address}...")
            
            # Ensure email and password are strings, not bytes
            if isinstance(email_address, bytes):
                email_address = email_address.decode('utf-8')
                
            if isinstance(password, bytes):
                password = password.decode('utf-8')
            
            # Ensure we have a valid password
            if not password:
                error_msg = "No password provided for login"
                print(f"❌ {error_msg}")
                await log_error(pool, email_address, 'login_error', error_msg)
                return False
                
            # Get the password from credentials
            password = cred['password']
            
            # Check if password is still encrypted (shouldn't happen)
            if isinstance(password, str) and (password.startswith('ENC:') or password.startswith('gAA')):
                error_msg = "Password appears to be encrypted but should be decrypted by now"
                print(f"⚠️  {error_msg}")
                await log_error(pool, email_address, 'login_error', error_msg)
                return False
            
            # Minimal debug output
            print(f"   Connecting to {cred['imap']['host']}:{cred['imap']['port']} (SSL: {cred['imap'].get('use_ssl', True)})")
            try:
                mail.login(email_address, password.strip())
                print("   Login successful")
                
                # Only proceed with email processing if login was successful
                try:
                    # List all mailboxes (for debugging)
                    print("\nAvailable mailboxes:")
                    status, mailboxes = mail.list()
                    if status == 'OK':
                        for mailbox in mailboxes[:5]:  # Show first 5 mailboxes
                            print(f"   - {mailbox.decode()}")
                    
                    # Select the INBOX
                    status, messages = mail.select('INBOX')
                    if status != 'OK':
                        print(f"   Cannot access INBOX: {messages}")
                        return False
                        
                    print(f"   Selected INBOX, {messages[0].decode()} messages")
                    
                    # Process the INBOX
                    error_count = 0
                    try:
                        error_count = await process_email_folder(mail, 'INBOX', email_address, pool)
                        if error_count and error_count > 0:
                            print(f"   ⚠️  Encountered {error_count} errors while processing emails")
                    except Exception as e:
                        error_count = (error_count or 0) + 1
                        print(f"   ❌ Error processing emails: {str(e)}")
                        await log_error(pool, email_address, 'process_folder_error', str(e), folder='INBOX')
                    
                    return True
                    
                except Exception as process_error:
                    print(f"   Error processing emails: {str(process_error)}")
                    import traceback
                    traceback.print_exc()
                    return False
                
            except imaplib.IMAP4.error as login_error:
                error_msg = str(login_error).lower()
                print(f"   Login failed: {error_msg}")
                if 'invalid credentials' in error_msg or 'bad username or password' in error_msg:
                    print("   Please verify your email and password are correct.")
                    print("   If using Gmail, ensure you've generated an App Password if 2FA is enabled.")
                
                # Log the error with the masked password for security
                masked_password = f"{password[:2]}{'*' * (len(password)-4)}{password[-2:]}" if password and len(password) > 4 else "*" * (len(password) if password else 0)
                error_details = f"{error_msg} (password: {masked_password})"
                await log_error(pool, email_address, 'login_error', error_details)
                return False
                
                if 'auth' in error_msg or 'oauth' in error_msg.lower():
                    print("   ❌ Authentication failed")
                    print("      This account may require OAuth2 authentication")
                    print("      Consider using OAuth2 for authentication")
                else:
                    print(f"   ❌ Login failed: {str(login_error)}")
                raise
                
        except Exception as e:
            print(f"   ❌ Connection error: {str(e)}")
            print("      Please check:")
            print(f"      1. Server {host} is reachable")
            print(f"      2. Port {port} is open and accepting connections")
            print("      3. Your network connection is stable")
            raise
                
        except ssl.SSLError as e:
            print(f"   ❌ SSL/TLS Error: {str(e)}")
            print("      This might be due to:")
            print("      1. Outdated SSL/TLS configuration on the server")
            print("      2. Missing or invalid SSL certificates")
            print("      3. Server not supporting secure connections")
            raise
        except socket.gaierror:
            print(f"   ❌ Network Error: Could not resolve host '{host}'")
            print(f"      Please verify the IMAP server address is correct")
            raise
        except socket.timeout:
            print(f"   ❌ Connection timed out while connecting to {host}:{port}")
            print("      Possible causes:")
            print("      1. Network connectivity issues")
            print(f"      2. Firewall blocking port {port}")
            print("      3. Server not responding")
            raise
        except ConnectionRefusedError:
            print(f"   ❌ Connection refused by {host}:{port}")
            print("      The server is not accepting connections on this port")
            print(f"      Verify that {host} is the correct IMAP server and port {port} is open")
            raise
        except Exception as e:
            print(f"   ❌ Connection error: {str(e)}")
            print("      Please check:")
            print(f"      1. Server {host} is reachable")
            print(f"      2. Port {port} is open and accepting connections")
            print("      3. Your network connection is stable")
            raise
        
        # Process Gmail folders with priority
        folders_to_check = [
            'INBOX',  # Primary inbox
            '[Gmail]/Spam',  # Gmail spam
            '[Gmail]/Trash'  # Gmail trash
        ]
        
        # Process each folder with timeout
        try:
            for folder in folders_to_check:
                if time.time() - start_time > MONITORING['max_processing_time']:
                    print(f"   ⏱️  Processing time limit reached for {email_address}")
                    break
                    
                try:
                    try:
                        # For Gmail, we'll directly process the known folders
                        if folder.startswith('[Gmail]'):
                            # For Gmail special folders, we need to use the literal name
                            folder_name = folder
                        else:
                            # For standard folders like INBOX
                            folder_name = folder
                        
                        print(f"   📂 Processing folder: {folder_name}")
                        folder_errors = await process_email_folder(mail, folder_name, email_address, pool)
                        
                        if folder_errors is not None:  # Only process if we got a valid error count
                            if folder_errors == 0:
                                processed_count += 1
                            else:
                                error_count += folder_errors
                                # If we hit too many errors, stop processing this email account
                                if error_count >= 10:  # Adjust this threshold as needed
                                    print(f"   ⚠️  Too many errors ({error_count}), stopping processing for {email_address}")
                                    raise Exception("Too many errors in folder processing")
                    except Exception as e:
                        error_count += 1
                        print(f"   ❌ Error processing folder {folder}: {str(e)}")
                        if error_count >= 10:
                            print(f"   ⚠️  Too many errors ({error_count}), stopping processing for {email_address}")
                            raise
                            
                except asyncio.TimeoutError:
                    print(f"   ⏱️  Timeout listing folders for {email_address}")
                    continue
                except Exception as e:
                    print(f"   ❌ Error checking folder {folder}: {str(e)}")
                    error_count += 1
                    continue
                    
        except Exception as e:
            print(f"   ❌ Unexpected error processing {email_address}: {str(e)}")
            error_count += 1
            
        # Log metrics periodically
        current_time = time.time()
        if current_time - last_metrics_time > MONITORING['metrics_interval']:
            duration = current_time - start_time
            print(f"   📊 Metrics for {email_address} - "
                  f"Processed: {processed_count}, "
                  f"Errors: {error_count}, "
                  f"Duration: {duration:.1f}s")
            
            # Log metrics to database
            await log_metric(pool, 'emails_processed', processed_count, email_address)
            await log_metric(pool, 'processing_errors', error_count, email_address)
            await log_metric(pool, 'processing_duration', duration, email_address)
            
            last_metrics_time = current_time
            
            # Reset counters for next interval
            processed_count = 0
            error_count = 0

    except imaplib.IMAP4.error as e:
        error_msg = f"IMAP Error: {str(e)}"
        print(f"   ❌ {error_msg}")
        await log_error(pool, email_address, 'imap_error', error_msg)
    except socket.gaierror as e:
        print(f"   ❌ Network Error: Could not resolve host '{host}'. Please check the IMAP server address.")
    except socket.timeout:
        print(f"   ❌ Connection timed out while connecting to {host}:{port}")
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"   ❌ {error_msg}")
        await log_error(pool, email_address, 'unexpected_error', error_msg)
    finally:
        # Logout and close connection
        if mail:
            try:
                mail.logout()
            except Exception as e:
                error_msg = f"Error during logout: {str(e)}"
                print(f"   ⚠️  {error_msg}")
                await log_error(pool, email_address, 'logout_error', error_msg)

def setup_graceful_shutdown(loop):
    """Setup handlers for graceful shutdown."""
    shutdown_requested = False
    
    def shutdown_handler(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            print("\n🛑 Force shutdown requested...")
            os._exit(1)
            
        print("\n🛑 Received shutdown signal. Cleaning up... (Press Ctrl+C again to force)")
        shutdown_requested = True
        
        # Cancel all running tasks except the current one
        for task in asyncio.all_tasks(loop=loop):
            if task is not asyncio.current_task(loop=loop):
                task.cancel()

    # Register signal handlers only in the main thread
    if threading.current_thread() is threading.main_thread():
        if sys.platform != 'win32':
            # Unix systems
            try:
                loop.add_signal_handler(signal.SIGTERM, shutdown_handler, signal.SIGTERM, None)
            except (NotImplementedError, RuntimeError):
                pass  # Not supported on this platform
        signal.signal(signal.SIGINT, shutdown_handler)
    else:
        print("⚠️  Not in main thread, skipping signal handler registration")

async def main():
    """Main entry point with database connection management."""
    pool = None
    
    try:
        # Setup graceful shutdown
        loop = asyncio.get_running_loop()
        setup_graceful_shutdown(loop)
        
        # Initialize database connection pool
        pool = await get_db_pool()
        print("✅ Database connection pool created")
        
        # Ensure database schema exists
        try:
            async with pool.acquire() as conn:
                await conn.execute(DB_SCHEMA)
                print("✅ Database schema verified/created")
        except Exception as e:
            print(f"❌ Error initializing database schema: {str(e)}")
            raise
        
        # Initial cleanup of old records
        await cleanup_old_records(pool)
        
        while True:
            cycle_start = time.time()
            print("\nStarting new poll cycle at")
            print(datetime.now().isoformat())
            
            try:
                # Fetch credentials from database
                credentials = await fetch_credentials(pool)
                
                if not credentials:
                    print("No active email accounts found. Waiting 60 seconds...")
                    await asyncio.sleep(60)
                    continue
                
                print(f"Found {len(credentials)} active email accounts")
                
                # Process each account
                tasks = []
                for cred in credentials:
                    try:
                        task = asyncio.create_task(process_inbox(cred, pool))
                        tasks.append(task)
                    except Exception as e:
                        print(f"⚠️  Error creating task for {cred.get('email', 'unknown')}: {str(e)}")
                
                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log any errors from tasks
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        email = credentials[i].get('email', 'unknown')
                        print(f"⚠️  Error processing {email}: {str(result)}")
                
                # Log metrics
                cycle_duration = time.time() - cycle_start
                await log_metric(pool, 'poll_cycle_duration', cycle_duration)
                await log_metric(pool, 'poll_cycle_complete', 1)
                
                # Calculate sleep time (5 minutes between cycles)
                poll_interval = 300  # 5 minutes in seconds
                cycle_duration = time.time() - cycle_start
                sleep_time = max(10, poll_interval - cycle_duration)  # At least 10 seconds between cycles
                
                print(f"✅ Poll cycle completed in {cycle_duration:.2f} seconds")
                print(f"⏳ Next poll in {sleep_time/60:.1f} minutes...")
                
                # Sleep before next poll
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                error_msg = f"Error in main loop: {str(e)}"
                print("\n⚠️  ")
                print(error_msg)
                await log_error(pool, 'system', 'main_loop_error', error_msg)
                await asyncio.sleep(60)  # Wait a bit before retrying
                
    except Exception as e:
        error_msg = f"Fatal error: {str(e)}"
        print("\n❌ ")
        print(error_msg)
        if pool:
            await log_error(pool, 'system', 'fatal_error', error_msg)
        raise
        
    finally:
        # Clean up resources
        if pool:
            try:
                await pool.close()
                print("\n✅ Database connection pool closed.")
            except Exception as e:
                print("\n⚠️  Error closing database pool:")
                print(str(e))

async def shutdown(loop):
    """Cleanup tasks tied to the service's shutdown."""
    print("\n🧹 Cleaning up resources...")
    
    # Cancel all running tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    
    # Wait for all tasks to be cancelled
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Close the loop
    loop.stop()
    print("✅ Cleanup complete")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\n👋 Shutdown requested. Exiting...")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
    finally:
        loop.run_until_complete(shutdown(loop))
        loop.close()
        print("👋 Goodbye!")
