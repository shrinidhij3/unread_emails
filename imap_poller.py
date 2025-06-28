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
from email.header import decode_header
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import hashlib
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants
MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024  # 10MB
ATTACHMENTS_DIR = 'attachments'

# Create attachments directory if it doesn't exist
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)

# Load environment variables
from dotenv import load_dotenv
import os

# Print current working directory for debugging
print(f"Current working directory: {os.getcwd()}")

# Load environment variables from .env file
print("🔍 Loading environment variables...")
env_path = os.path.join(os.path.dirname(__file__), '.env')
print(f"Looking for .env file at: {env_path}")

if os.path.exists(env_path):
    print("✅ .env file found")
    with open(env_path, 'r') as f:
        print("Contents of .env file:", f.read()[:100] + "..." if os.path.getsize(env_path) > 100 else f.read())
else:
    print("❌ .env file not found")

# Load environment variables
load_dotenv(env_path, override=True)

# Debug: Verify environment variables
print("\nEnvironment variables:")
print(f"DJANGO_SECRET_KEY is set: {'Yes' if os.getenv('DJANGO_SECRET_KEY') else 'No'}")
print(f"DATABASE_URL is set: {'Yes' if os.getenv('DATABASE_URL') else 'No'}")


# Database configuration from environment variables
DB_CONFIG = {
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME", "railway"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "command_timeout": int(os.getenv("DB_COMMAND_TIMEOUT", "10")),  # seconds
    "min_size": int(os.getenv("DB_POOL_MIN_SIZE", "1")),  # Minimum connections in pool
    "max_size": int(os.getenv("DB_POOL_MAX_SIZE", "5")),  # Maximum connections in pool
    "max_inactive_connection_lifetime": int(os.getenv("DB_MAX_INACTIVE_CONN_LIFETIME", "300")),  # 5 minutes
    "max_queries": int(os.getenv("DB_MAX_QUERIES", "50000")),  # Max queries before connection is replaced
    "timeout": float(os.getenv("DB_TIMEOUT", "10.0")),  # Connection timeout in seconds
    "ssl": os.getenv("DB_SSL", "require")  # SSL mode
}

# Validate required environment variables
required_vars = ["DB_PASSWORD", "DB_HOST"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

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
    # Remove None values from config
    pool_config = {k: v for k, v in DB_CONFIG.items() if v is not None}
    
    # Create connection pool with explicit parameters for better error reporting
    try:
        pool = await asyncpg.create_pool(
            host=pool_config.get('host'),
            port=pool_config.get('port'),
            user=pool_config.get('user'),
            password=pool_config.get('password'),
            database=pool_config.get('database'),
            min_size=pool_config.get('min_size', 1),
            max_size=pool_config.get('max_size', 5),
            max_inactive_connection_lifetime=pool_config.get('max_inactive_connection_lifetime', 300.0),
            max_queries=pool_config.get('max_queries', 50000),
            command_timeout=pool_config.get('command_timeout'),
            ssl=pool_config.get('ssl')
        )
        
        # Initialize database schema
        async with pool.acquire() as conn:
            await conn.execute(DB_SCHEMA)
            
        return pool
        
    except Exception as e:
        print(f"❌ Error creating database connection pool: {str(e)}")
        # Print the connection details (without password) for debugging
        debug_config = pool_config.copy()
        if 'password' in debug_config:
            debug_config['password'] = '***'
        print(f"Connection config: {debug_config}")
        raise

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
        print(f"❌ Decryption error: {e}")
        print(f"   Input length: {len(encrypted_data)}")
        print(f"   Input starts with: {encrypted_data[:10]}..." if len(encrypted_data) > 10 else "   Input too short")
        raise ValueError(f"Failed to decrypt password: {e}")

async def fetch_credentials(pool):
    """Fetch email credentials from the credentials_email table with provider-specific settings.
    
    Expects password to be encrypted with Fernet in the format:
    - ENC:encrypted_data
    
    The DJANGO_SECRET_KEY environment variable must be set for decryption.
    """
    try:
        print("🔍 Fetching email credentials from database...")
        
        # First, check if the table exists
        table_check = await pool.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'credentials_email'
            )
            """
        )
        
        if not table_check:
            print("❌ Error: credentials_email table does not exist")
            return []
            
        print("✅ Found credentials_email table")
        
        # Get all column names from the table
        columns_result = await pool.fetch(
            """
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'credentials_email'
            """
        )
        
        # Log column information for debugging
        column_info = {row['column_name']: row['data_type'] for row in columns_result}
        print(f"ℹ️ Table columns and types: {column_info}")
        
        # Build the query based on available columns
        select_fields = []
        if 'account_id' in column_info:
            select_fields.append('account_id')
        if 'email' in column_info:
            select_fields.append('email')
        if 'password_encrypted' in column_info:
            select_fields.append('password_encrypted as password')
        elif 'password' in column_info:
            select_fields.append('password')
        if 'provider_type' in column_info:
            select_fields.append('provider_type')
        if 'imap_host' in column_info:
            select_fields.append('imap_host')
        if 'imap_port' in column_info:
            select_fields.append('imap_port')
        if 'imap_use_ssl' in column_info:
            select_fields.append('imap_use_ssl')
        if 'smtp_host' in column_info:
            select_fields.append('smtp_host')
        if 'smtp_port' in column_info:
            select_fields.append('smtp_port')
        if 'smtp_use_ssl' in column_info:
            select_fields.append('smtp_use_ssl')
        if 'smtp_use_tls' in column_info:
            select_fields.append('smtp_use_tls')
        if 'is_active' in column_info:
            select_fields.append('is_active')
            
        if not select_fields:
            print("❌ Error: No valid columns found in credentials_email table")
            return []
            
        # Build the WHERE clause
        where_clause = "WHERE 1=1"
        if 'is_active' in column_info:
            where_clause += " AND is_active = TRUE"
            
        query = f"""
            SELECT {', '.join(select_fields)}
            FROM credentials_email
            {where_clause}
        """
        
        print(f"🔍 Executing query: {query}")
        
        # Execute the query
        records = await pool.fetch(query)
        
        if not records:
            print("ℹ️ No email accounts found in the database")
            return []
            
        print(f"✅ Found {len(records)} email accounts in the database")
        
        credentials = []
        for record in records:
            try:
                cred = dict(record)
                email = cred.get('email', 'unknown')
                print(f"\n🔍 Processing account: {email}")
                
                # Skip if no email or password
                if not email or not (cred.get('password') or cred.get('password_encrypted')):
                    print(f"⚠️ Skipping account - missing email or password: {email}")
                    continue
                
                # Handle password decryption if needed
                password = cred.get('password') or cred.get('password_encrypted')
                if password:
                    try:
                        # Handle string passwords
                        if isinstance(password, str):
                            # Check for ENC: prefix or Fernet token format (starts with gAAAAA)
                            if password.startswith('ENC:'):
                                print("   🔑 Detected ENC: prefixed password")
                                password = password[4:]  # Remove 'ENC:' prefix
                                password = decrypt_fernet(password)
                                print(f"   ✅ Successfully decrypted password")
                            elif password.startswith('gAAAAA'):
                                print("   🔑 Detected Fernet-encrypted password")
                                password = decrypt_fernet(password)
                                print(f"   ✅ Successfully decrypted Fernet password")
                            else:
                                print("   ℹ️ Using plaintext password")
                        # Handle bytes passwords
                        elif isinstance(password, bytes):
                            try:
                                password_str = password.decode('utf-8')
                                if password_str.startswith(('ENC:', 'gAAAAA')):
                                    password = decrypt_fernet(password_str[4:] if password_str.startswith('ENC:') else password_str)
                                    print(f"   ✅ Successfully decrypted password from bytes")
                                else:
                                    password = password_str
                                    print("   ℹ️ Using plaintext password from bytes")
                            except Exception as e:
                                print(f"   ⚠️ Could not process password bytes: {str(e)}")
                                continue
                    except Exception as e:
                        print(f"   ❌ Failed to process password: {str(e)}")
                        continue
                
                # Set default provider settings if not specified
                provider_type = (cred.get('provider_type') or 'custom').lower()
                print(f"Provider type: {provider_type}")
                
                # Set default IMAP/SMTP settings based on provider type if not provided
                if not cred.get('imap_host'):
                    if 'gmail' in provider_type:
                        print("🔧 Using Gmail default settings")
                        cred.update({
                            'imap_host': 'imap.gmail.com',
                            'imap_port': 993,
                            'imap_use_ssl': True,
                            'smtp_host': 'smtp.gmail.com',
                            'smtp_port': 465,
                            'smtp_use_ssl': True,
                            'smtp_use_tls': False
                        })
                    elif 'outlook' in provider_type or 'hotmail' in provider_type or 'office365' in provider_type:
                        print("🔧 Using Outlook/Office 365 default settings")
                        cred.update({
                            'imap_host': 'outlook.office365.com',
                            'imap_port': 993,
                            'imap_use_ssl': True,
                            'smtp_host': 'smtp.office365.com',
                            'smtp_port': 587,
                            'smtp_use_ssl': False,
                            'smtp_use_tls': True
                        })
                
                # Format the credentials with the expected structure
                formatted_cred = {
                    'email': email,
                    'password': password,
                    'imap': {
                        'host': cred.get('imap_host'),
                        'port': int(cred.get('imap_port', 993)),
                        'use_ssl': bool(cred.get('imap_use_ssl', True)),
                        'secure': bool(cred.get('secure', True))
                    },
                    'smtp': {
                        'host': cred.get('smtp_host'),
                        'port': int(cred.get('smtp_port', 587)),
                        'use_ssl': bool(cred.get('smtp_use_ssl', False)),
                        'use_tls': bool(cred.get('smtp_use_tls', True))
                    },
                    'provider': cred.get('provider_type', 'custom')
                }
                
                # Add any additional fields
                if 'name' in cred:
                    formatted_cred['name'] = cred['name']
                if 'notes' in cred:
                    formatted_cred['notes'] = cred['notes']
                
                # Add to the list of credentials
                credentials.append(formatted_cred)
                print(f"✅ Successfully processed account: {email}")
                print(f"   IMAP: {formatted_cred['imap']}")
                print(f"   SMTP: {formatted_cred['smtp']}")
                
            except Exception as e:
                print(f"❌ Error processing account: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
                
        print(f"\n✅ Successfully loaded {len(credentials)} email accounts")
        return credentials
        
    except Exception as e:
        print(f"❌ Error in fetch_credentials: {str(e)}")
        import traceback
        traceback.print_exc()
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
                return error_count  # Return current error count (0 if no errors yet)
                
            mail_ids = messages[0].split()
            if not mail_ids:
                print(f"   No unread messages in {folder}")
                return error_count  # Return current error count (0 if no errors yet)
                
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

async def process_inbox(cred, pool):
    """Process emails for a single account with database connection pool.
    
    Args:
        cred: Dictionary containing email credentials and connection details
        pool: Database connection pool
    """
    email_address = cred['email']
    password = cred['password']
    imap_config = cred['imap']
    host = imap_config['host']
    port = imap_config['port']
    use_ssl = imap_config['use_ssl']
    
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
            print(f"   Debug - Password length: {len(password) if password else 0}, starts with: {password[:2] if password else 'N/A'}...")
            
            # Ensure email and password are strings, not bytes
            print("\n🔍 DEBUG - Raw credentials:")
            print(f"   Email (type: {type(email_address).__name__}): {email_address}")
            print(f"   Password (type: {type(password).__name__}): {password[:10]}..." if password and len(password) > 10 else f"   Password: {password}")
            
            if isinstance(email_address, bytes):
                email_address = email_address.decode('utf-8')
                print("   Decoded email from bytes to string")
                
            if isinstance(password, bytes):
                password = password.decode('utf-8')
                print("   Decoded password from bytes to string")
            
            print("\n🔍 DEBUG - Processing credentials:")
            print(f"   Email: {email_address}")
            print(f"   Password type: {type(password).__name__}")
            print(f"   Password starts with: {password[:4]}..." if password else "   No password")
            
            # Ensure we have a valid password
            if not password:
                print("❌ Error: No password provided")
                return False
                
            # Get the password from credentials
            password = cred['password']
            
            # Debug output for password
            print("\n" + "="*50)
            print("🔑 PASSWORD DEBUGGING INFORMATION")
            print("="*50)
            print(f"\n[1/3] Password retrieved from credentials:")
            print(f"   Type: {type(password).__name__}")
            print(f"   Length: {len(password) if password else 0}")
            
            # Show masked password for security
            if password and len(password) > 4:
                masked = f"{password[:2]}{'*' * (len(password) - 4)}{password[-2:]}"
            else:
                masked = "*" * (len(password) if password else 0)
            
            print(f"   Masked password: {masked}")
            
            # Check if password looks encrypted (starts with 'ENC:')
            is_encrypted = isinstance(password, str) and password.startswith('ENC:')
            print(f"\n[2/3] Password analysis:")
            print(f"   Appears encrypted: {'Yes' if is_encrypted else 'No'}")
            if password:
                print(f"   First 10 chars: {password[:10]}...")
            
            # Verify we have a password
            if not password:
                print("\n❌ ERROR: No password provided for login")
                await log_error(pool, email_address, 'login_error', 'No password provided')
                return False
                
            if is_encrypted:
                print("\n⚠️  WARNING: Password appears to be encrypted but should be decrypted by now!")
                print("   The password should be decrypted in fetch_credentials() before reaching this point.")
                print("   This suggests an issue with the decryption process.")
            
            # Try login with the provided password
            print(f"\n[3/3] Attempting IMAP login:")
            print(f"   Email: {email_address}")
            print(f"   IMAP Server: {cred['imap']['host']}:{cred['imap']['port']}")
            print(f"   Using SSL: {cred['imap'].get('use_ssl', True)}")
            print(f"   Password length: {len(password)}")
            print("-"*50 + "\n")
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
                    error_count = await process_email_folder(mail, 'INBOX', email_address, pool)
                    if error_count > 0:
                        print(f"   Encountered {error_count} errors while processing emails")
                    
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

async def main():
    """Main entry point with database connection management."""
    pool = None
    
    try:
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
                
                # Calculate sleep time (minimum 30 seconds between cycles)
                cycle_duration = time.time() - cycle_start
                sleep_time = max(30, 30 - cycle_duration)  # At least 30 seconds between cycles
                
                print(f"✅ Poll cycle completed in {cycle_duration:.2f} seconds")
                print(f"⏳ Next poll in {sleep_time:.0f} seconds...")
                
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

if __name__ == "__main__":
    asyncio.run(main())
