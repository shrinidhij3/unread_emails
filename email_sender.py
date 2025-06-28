from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dotenv import load_dotenv
import base64
import os
import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import email.utils as email_utils
from email.utils import formataddr, formatdate, make_msgid
from typing import Optional, Dict, Any
import logging
from pprint import pformat
import re
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Email Sender API",
    description="API for sending emails with support for encrypted credentials",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/healthz")
async def healthz():
    """Health check endpoint for Render (matches their default path)"""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# Load environment variables from .env file
load_dotenv()

# Verify required environment variables are set
if not os.getenv('DJANGO_SECRET_KEY'):
    raise ValueError("DJANGO_SECRET_KEY environment variable not found in .env file")

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_formatter = logging.Formatter(log_format)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# Debug file handler
debug_handler = logging.FileHandler('logs/debug.log')
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(log_formatter)

# Error file handler
error_handler = logging.FileHandler('logs/error.log')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(log_formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(debug_handler)
logger.addHandler(error_handler)

# Disable overly verbose logs
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('email').setLevel(logging.WARNING)

# Log startup message
logger.info("=" * 50)
logger.info("EMAIL SENDER SERVICE STARTING")
logger.info("=" * 50)
logger.info("Logging configured. Log level: %s", logging.getLevelName(logger.getEffectiveLevel()))

def get_fernet_key():
    """
    Generate Fernet key using the same method as Django's default encryption.
    Uses DJANGO_SECRET_KEY from environment variables.
    """
    # Get the secret key from environment variables
    secret_key = os.getenv('DJANGO_SECRET_KEY')
    if not secret_key:
        raise ValueError("DJANGO_SECRET_KEY environment variable not set")
    
    # Use the first 16 bytes of the secret key as salt (matching IMAP poller)
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
        encrypted_data: The encrypted password string (starts with 'ENC:' or 'gAA')
        
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
        error_msg = f"Failed to decrypt password: {e}"
        logger.error(error_msg)
        logger.error(f"Input length: {len(encrypted_data)}")
        if len(encrypted_data) > 10:
            logger.error(f"Input starts with: {encrypted_data[:10]}...")
        else:
            logger.error("Input too short")
        raise ValueError(error_msg)

# Models
class MailingAddress(BaseModel):
    name: str = "Thorsignia"
    address_line1: str = "123 Business Street"
    address_line2: Optional[str] = None
    city: str = "Bangalore"
    state: str = "Karnataka"
    postal_code: str = "560001"
    country: str = "India"

class EmailRequest(BaseModel):
    sender: EmailStr
    password: str
    recipient: EmailStr
    subject: str
    message: str = ""
    html: Optional[str] = None
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    reply_to: Optional[EmailStr] = None
    mailing_address: Optional[MailingAddress] = None

def format_mailing_address(address: Optional[MailingAddress] = None) -> str:
    """Format a mailing address from a MailingAddress model."""
    if not address:
        address = MailingAddress()
    
    lines = [address.name]
    if address.address_line1:
        lines.append(address.address_line1)
    if address.address_line2:
        lines.append(address.address_line2)
    
    city_line = []
    if address.city:
        city_line.append(address.city)
    if address.state:
        city_line.append(address.state)
    if address.postal_code:
        city_line.append(address.postal_code)
    
    lines.append(", ".join(city_line))
    if address.country:
        lines.append(address.country)
    
    return "\n".join(lines)

@app.post("/send-email")
async def send_email(email_data: EmailRequest, request: Request):
    try:
        # Log request details (excluding sensitive data)
        request_data = email_data.dict()
        if 'password' in request_data:
            request_data['password'] = '***REDACTED***' if request_data['password'] else None
            
        logger.info("\n" + "="*50 + " NEW REQUEST " + "="*50)
        logger.info("Incoming request from: %s", request.client.host if request.client else "unknown")
        logger.info("Request data: %s", pformat(request_data, width=100))
        logger.debug("Full headers: %s", dict(request.headers))
        
        # Log the actual password (masked) for debugging
        if hasattr(email_data, 'password'):
            pwd = email_data.password
            logger.debug("Password received (masked): %s...%s (length: %d)", 
                       pwd[:2] if pwd else '', 
                       pwd[-2:] if pwd and len(pwd) > 4 else '',
                       len(pwd) if pwd else 0)
            
    except Exception as e:
        logger.critical("Error in request processing: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail={
            "status": "error",
            "error": f"Request processing failed: {str(e)}",
            "type": type(e).__name__,
            "details": str(e)
        })
    logger.info("\n" + "="*50 + " NEW REQUEST " + "="*50)
    logger.info(f"Processing request from {email_data.sender} to {email_data.recipient}")
    logger.info(f"Subject: {email_data.subject}")
    logger.info(f"SMTP Host: {email_data.smtp_host}:{email_data.smtp_port}")
    logger.info("="*108 + "\n")
    
    # Log request details (without sensitive data)
    request_data = email_data.dict()
    if 'password' in request_data:
        request_data['password'] = '***REDACTED***' if request_data['password'] else None
    logger.debug(f"Request data: {request_data}")
    """
    Send an email using SMTP
    
    Required fields:
    - sender: Sender's email address
    - password: SMTP password/app password
    - recipient: Recipient's email address
    - subject: Email subject
    - message: Plain text message (or html)
    """
    try:
        # Log email preparation
        logger.info("Preparing email from %s to %s", email_data.sender, email_data.recipient)
        logger.debug("Subject: %s", email_data.subject)
        
        # Use reply_to if provided, otherwise use sender
        reply_to = email_data.reply_to or email_data.sender
        logger.debug("Using reply-to address: %s", reply_to)
        
        # Create email message
        logger.debug("Creating MIME message")
        msg = MIMEMultipart('alternative')
        message_id = email_utils.make_msgid(domain=email_data.sender.split('@')[-1])
        logger.debug("Generated Message-ID: %s", message_id)
        
        # Set basic headers
        msg['Subject'] = email_data.subject
        msg['From'] = email_utils.formataddr(('Thorsignia Support', email_data.sender))
        msg['To'] = email_data.recipient
        msg['Reply-To'] = email_utils.formataddr(('Thorsignia Support', reply_to))
        msg['Date'] = email_utils.formatdate(localtime=True)
        msg['Message-ID'] = email_utils.make_msgid(domain=email_data.sender.split('@')[-1])
        
        # Add unsubscribe header
        msg['List-Unsubscribe'] = f'<mailto:{reply_to}?subject=Unsubscribe>'
        
        # Handle plain text content
        message_body = email_data.message
        if not message_body and email_data.html:
            # Create plain text from HTML if no plain text provided
            message_body = re.sub(r'<[^>]+>', ' ', email_data.html)
            message_body = re.sub(r'\s+', ' ', message_body).strip()
        
        if not message_body:
            message_body = "Please enable HTML to view this email."
        
        # Add plain text part
        text_part = MIMEText(message_body, 'plain', 'utf-8')
        msg.attach(text_part)
        
        # Add HTML part if provided
        if email_data.html:
            formatted_address = format_mailing_address(email_data.mailing_address)
            reply_to = email_data.reply_to or email_data.sender
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
                <title>{email_data.subject}</title>
            </head>
            <body style="margin: 0; padding: 20px; font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto;">
                    {email_data.html}
                    <p style="margin-top: 30px; font-size: 12px; color: #777; border-top: 1px solid #eee; padding-top: 10px;">
                        <a href="mailto:{reply_to}?subject=Unsubscribe" style="color: #0066cc; text-decoration: none;">Unsubscribe</a>
                        <span style="color: #ddd; margin: 0 10px;">|</span>
                        {formatted_address.replace(chr(10), '<br>')}
                    </p>
                </div>
            </body>
            </html>
            """
            
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
        
        # Set priority headers
        msg['X-Priority'] = '1'  # High priority
        msg['Importance'] = 'high'
        
        # Clean email addresses
        def clean_email(email_str: str) -> str:
            if not email_str:
                return ""
            cleaned = ''.join(c for c in email_str if ord(c) < 128).strip()
            if cleaned != email_str:
                logger.debug("Cleaned email: %s -> %s", email_str, cleaned)
            return cleaned
            
        try:
            clean_recipient = clean_email(email_data.recipient)
            clean_sender = clean_email(email_data.sender)
            logger.debug("Cleaned sender: %s -> %s", email_data.sender, clean_sender)
            logger.debug("Cleaned recipient: %s -> %s", email_data.recipient, clean_recipient)
        except Exception as e:
            logger.error("Error cleaning email addresses: %s", str(e), exc_info=True)
            clean_recipient = email_data.recipient
            clean_sender = email_data.sender
        
        # Decrypt the password if it's encrypted
        password = email_data.password
        logger.debug("Password received (first 5 chars): %s", 
                   password[:5] + '...' if password and len(password) > 5 else str(password))
        
        if password and (password.startswith('ENC:') or password.startswith('gAA') or len(password) > 50):
            try:
                logger.info("Password appears to be encrypted, attempting to decrypt...")
                password = decrypt_fernet(password)
                logger.info("Successfully decrypted password")
                logger.debug("Decrypted password length: %d", len(password) if password else 0)
                logger.debug("First 2 chars of decrypted password: %s", 
                            password[:2] + '...' if password else 'None')
            except Exception as e:
                error_msg = f"Failed to decrypt password: {str(e)}"
                logger.error(error_msg, exc_info=True)
                logger.error("Make sure DJANGO_SECRET_KEY in .env matches the one used for encryption")
                secret_key = os.getenv('DJANGO_SECRET_KEY')
                if secret_key:
                    logger.error("Current DJANGO_SECRET_KEY starts with: %s...", secret_key[:5])
                    logger.debug("Full DJANGO_SECRET_KEY: %s", secret_key)
                else:
                    logger.error("DJANGO_SECRET_KEY not set in environment variables")
                raise ValueError(error_msg) from e
        else:
            logger.info("Password doesn't appear to be encrypted, using as-is")
        
        # Send email using SMTP
        logger.info("Initiating SMTP connection to %s:%s", 
                   email_data.smtp_host, email_data.smtp_port)
        logger.debug("Authenticating as: %s", clean_sender)
        logger.debug("Using password: %s", 
                   '***' + password[-2:] if password and len(password) > 2 else '***')
        
        try:
            logger.debug("Creating SMTP connection...")
            with smtplib.SMTP(email_data.smtp_host, email_data.smtp_port, timeout=30) as server:
                logger.info("SMTP connection established")
                
                # EHLO/HELO
                logger.debug("Sending EHLO...")
                ehlo_resp = server.ehlo()
                logger.debug("EHLO response: %s", ehlo_resp[0] if ehlo_resp else "No response")
                
                # STARTTLS if available
                if server.has_extn('STARTTLS'):
                    logger.info("Server supports STARTTLS, upgrading connection...")
                    starttls_resp = server.starttls()
                    logger.debug("STARTTLS response: %s", starttls_resp)
                    ehlo_resp = server.ehlo()
                    logger.debug("EHLO after STARTTLS: %s", ehlo_resp[0] if ehlo_resp else "No response")
                
                # Login
                logger.info("Attempting to login to SMTP server...")
                logger.debug("SMTP Login - Email: %s", clean_sender)
                logger.debug("SMTP Login - Password length: %d", len(password) if password else 0)
                
                login_resp = server.login(clean_sender, password)
                logger.info("Successfully logged in to SMTP server")
                logger.debug("Login response: %s", login_resp)
                
                # Update headers with cleaned emails
                try:
                    logger.debug("Updating email headers...")
                    if 'From' in msg:
                        from_header = email_utils.formataddr(('Thorsignia Support', clean_sender))
                        msg.replace_header('From', from_header)
                        logger.debug("Updated From header: %s", from_header)
                    if 'To' in msg:
                        msg.replace_header('To', clean_recipient)
                        logger.debug("Updated To header: %s", clean_recipient)
                except Exception as e:
                    logger.warning("Failed to update email headers: %s", str(e), exc_info=True)
                
                # Send email
                logger.info("Sending email...")
                raw_message = msg.as_string()
                logger.debug("Message size: %d bytes", len(raw_message))
                
                send_response = server.sendmail(clean_sender, [clean_recipient], raw_message)
                if not send_response:
                    logger.info("Email sent successfully to %s", clean_recipient)
                    logger.debug("Server response: %s", send_response)
                else:
                    logger.warning("Received non-empty response from server: %s", send_response)
                
                # Log success
                logger.info("="*50 + " EMAIL SENT SUCCESSFULLY " + "="*50)
                logger.info("From: %s", clean_sender)
                logger.info("To: %s", clean_recipient)
                logger.info("Subject: %s", email_data.subject)
                logger.info("Message ID: %s", msg['Message-ID'])
                logger.info("="*108)
                
                return {
                    "status": "success",
                    "message": "Email sent successfully",
                    "to": clean_recipient,
                    "subject": email_data.subject
                }
                
        except smtplib.SMTPAuthenticationError as e:
            logger.critical("SMTP Authentication failed: %s", str(e), exc_info=True)
            logger.critical("Please check your email credentials and ensure 'Less secure app access' is enabled")
            raise
        except smtplib.SMTPException as e:
            logger.critical("SMTP Error: %s", str(e), exc_info=True)
            logger.critical("SMTP server response: %s", getattr(e, 'smtp_error', 'No response'))
            logger.critical("SMTP code: %s", getattr(e, 'smtp_code', 'Unknown'))
            logger.critical("SMTP enhanced status: %s", getattr(e, 'smtp_error', {}).get('enhanced_status_code', 'None'))
            raise
        except Exception as e:
            logger.critical("Unexpected error during SMTP communication: %s", str(e), exc_info=True)
            raise
            
    except Exception as e:
        import traceback
        import sys
        from typing import Type, Any
        
        # Get the full traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        # Format the traceback
        tb_list = traceback.format_exception(exc_type, exc_value, exc_traceback)
        formatted_traceback = ''.join(tb_list)
        
        # Log the full error with traceback
        logger.critical("\n" + "="*50 + " CRITICAL ERROR " + "="*50)
        logger.critical("Error Type: %s", type(e).__name__)
        logger.critical("Error Message: %s", str(e))
        logger.critical("\nTraceback:\n%s", formatted_traceback)
        
        # Log additional context
        logger.critical("\nError Context:")
        logger.critical("- SMTP Host: %s", email_data.smtp_host if 'email_data' in locals() else 'N/A')
        logger.critical("- Sender: %s", email_data.sender if 'email_data' in locals() else 'N/A')
        logger.critical("- Recipient: %s", email_data.recipient if 'email_data' in locals() else 'N/A')
        
        logger.critical("="*108 + "\n")
        
        # Also print to stderr for immediate visibility
        print("\n" + "="*50 + " CRITICAL ERROR " + "="*50, file=sys.stderr)
        print(f"Error Type: {type(e).__name__}", file=sys.stderr)
        print(f"Error Message: {str(e)}", file=sys.stderr)
        print("\nTraceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("\n" + "="*108, file=sys.stderr)
        
        # Return detailed error in response
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "error": f"Failed to send email: {str(e)}",
                "type": error_type,
                "details": str(e)  # Include full error details
            }
        )

# This allows the app to be used with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    
    # Configure logging for the main process
    logger.info("\n" + "="*50 + " STARTING SERVER " + "="*50)
    logger.info("Starting Uvicorn server...")
    
    try:
        # Start the server with detailed logging
        uvicorn.run(
            "email_sender:app",
            host="0.0.0.0",
            port=8000,
            log_level="debug",
            reload=True,
            log_config={
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "default": {
                        "()": "uvicorn.logging.DefaultFormatter",
                        "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        "use_colors": True,
                    },
                },
                "handlers": {
                    "default": {
                        "formatter": "default",
                        "class": "logging.StreamHandler",
                        "stream": "ext://sys.stderr",
                    },
                    "file": {
                        "class": "logging.FileHandler",
                        "filename": "email_sender.log",
                        "formatter": "default",
                    },
                },
                "loggers": {
                    "": {"handlers": ["default", "file"], "level": "DEBUG"},
                    "uvicorn": {"handlers": ["default", "file"], "level": "INFO"},
                    "uvicorn.error": {"level": "INFO"},
                    "uvicorn.access": {
                        "handlers": ["default", "file"],
                        "level": "INFO",
                        "propagate": False,
                    },
                },
            },
        )
    except KeyboardInterrupt:
        logger.info("\n" + "="*50 + " SHUTTING DOWN " + "="*50)
        logger.info("Received keyboard interrupt. Shutting down gracefully...")
    except Exception as e:
        logger.critical("Fatal error in server: %s", str(e), exc_info=True)
        raise
    finally:
        logger.info("Server shutdown complete")
