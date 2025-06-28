import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, validator, HttpUrl
from enum import Enum
import asyncpg
import smtplib
import email.utils
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email import policy
import re
from typing import Optional, Dict, Any, List, Union
import threading
import time

from email_service import (
    ProviderType,
    EmailAccount,
    EmailAccountManager,
    EmailSyncService,
    CryptoService,
    provider_registry,
    connection_manager
)

# Global variables for poller
poller_thread = None
poller_running = False
last_run = None

def run_poller_loop():
    """Background function that runs the email poller in a loop"""
    global last_run
    while poller_running:
        try:
            logging.info("Starting email poll cycle")
            asyncio.run(run_poller())
            last_run = datetime.utcnow()
            logging.info(f"Poll cycle completed at {last_run.isoformat()}")
        except Exception as e:
            logging.error(f"Error in poller: {str(e)}")
        
        # Sleep for 5 minutes, checking every second if we should stop
        for _ in range(300):  # 300 seconds = 5 minutes
            if not poller_running:
                break
            time.sleep(1)

# Import the poller
from imap_poller import main as run_poller

# Global variables for poller
poller_thread = None
poller_running = False
last_run = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('email_api.log')
    ]
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Email Sync API",
    description="API for managing email accounts and synchronization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection pool
_db_pool = None

# Services
_crypto_service = None
_account_manager = None
_sync_service = None

# Background tasks
_sync_tasks: Dict[int, asyncio.Task] = {}

# Pydantic models
# Email sending related models
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
    message: Optional[str] = ""
    html: Optional[str] = None
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    reply_to: Optional[EmailStr] = None
    mailing_address: Optional[MailingAddress] = None

class ProviderTypeStr(str, Enum):
    GMAIL = 'gmail'
    OFFICE365 = 'office365'
    CPANEL = 'cpanel'
    ZOHO = 'zoho'
    CUSTOM = 'custom'
    AUTO = 'auto'

class EmailAccountCreate(BaseModel):
    email: EmailStr
    password: str
    provider_type: ProviderTypeStr = ProviderTypeStr.AUTO
    imap_host: Optional[str] = None
    imap_port: Optional[int] = None
    imap_use_ssl: bool = True
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_use_ssl: bool = True
    smtp_use_tls: bool = True

class EmailAccountResponse(BaseModel):
    id: int
    email: str
    provider_type: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_synced_at: Optional[datetime] = None
    sync_status: str = 'idle'
    error_count: int = 0
    last_error: Optional[str] = None

    class Config:
        orm_mode = True
        from_attributes = True

class SyncResponse(BaseModel):
    success: bool
    message: str
    task_id: Optional[str] = None

class SyncStatus(str, Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'

class SyncTaskResponse(BaseModel):
    task_id: str
    status: SyncStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

# Database connection
def get_db_pool():
    return _db_pool

# Service getters
async def get_crypto_service() -> CryptoService:
    global _crypto_service
    if _crypto_service is None:
        _crypto_service = CryptoService()
    return _crypto_service

async def get_account_manager() -> EmailAccountManager:
    global _account_manager, _db_pool, _crypto_service
    if _account_manager is None:
        if _crypto_service is None:
            _crypto_service = await get_crypto_service()
        _account_manager = EmailAccountManager(_db_pool, _crypto_service)
    return _account_manager

async def get_sync_service() -> EmailSyncService:
    global _sync_service, _db_pool, _account_manager
    if _sync_service is None:
        if _account_manager is None:
            _account_manager = await get_account_manager()
        _sync_service = EmailSyncService(_db_pool, account_manager=_account_manager)
    return _sync_service

# Startup and shutdown events
@app.on_event("startup")
async def startup():
    """Initialize database connection, services, and start the poller."""
    global poller_thread, poller_running, _db_pool
    
    try:
        # Using environment variables directly from the system
        # Initialize connection manager without config - will be configured when needed
        global connection_manager
        
        # Default IMAP configuration from environment variables
        DEFAULT_IMAP_CONFIG = {
            'imap_host': os.getenv('IMAP_HOST', ''),
            'imap_port': int(os.getenv('IMAP_PORT', 993)),
            'imap_use_ssl': os.getenv('IMAP_USE_SSL', 'true').lower() == 'true'
        }
        
        # Update connection manager config if we have IMAP settings
        if DEFAULT_IMAP_CONFIG['imap_host']:
            connection_manager.config.update(DEFAULT_IMAP_CONFIG)
            logging.info(f"Configured IMAP connection manager with host: {DEFAULT_IMAP_CONFIG['imap_host']}")
            
            # Test the connection
            try:
                conn = await connection_manager.connect()
                if conn:
                    logging.info("Successfully connected to IMAP server")
                    conn.logout()
                else:
                    logging.warning("Could not establish IMAP connection with current configuration")
            except Exception as e:
                logging.error(f"Error testing IMAP connection: {str(e)}")
        else:
            logging.warning("No default IMAP configuration found. Email synchronization will not work until an account is added with valid IMAP settings.")

        # Initialize services
        crypto_service = await get_crypto_service()
        account_manager = await get_account_manager()
        sync_service = await get_sync_service()

        # Verify database schema
        try:
            if not _db_pool:
                _db_pool = await asyncpg.create_pool(dsn=os.getenv('DATABASE_URL'))
                
            async with _db_pool.acquire() as conn:
                # Check if credentials_email table exists
                table_exists = await conn.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'credentials_email'
                    );
                    """
                )
                
                if not table_exists:
                    logging.warning("Database tables not found. Running migrations...")
                    # Run migrations here if needed
                    # await run_migrations()
                    logging.info("Migrations completed")
                
                # Check if there are any email accounts
                account_count = await conn.fetchval("SELECT COUNT(*) FROM credentials_email")
                logging.info(f"Found {account_count} email accounts in the database")
                
        except Exception as e:
            logging.error(f"Error verifying database schema: {str(e)}")
            raise

        # Start the poller in a separate thread
        if poller_thread is None or not poller_thread.is_alive():
            poller_running = True
            poller_thread = threading.Thread(target=run_poller_loop, daemon=True)
            poller_thread.start()
            logging.info("Email poller started")

        logging.info("Application startup complete")

    except Exception as e:
        logging.error(f"Startup error: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown():
    """Clean up resources on application shutdown."""
    global poller_running
    
    # Stop the poller
    poller_running = False
    if poller_thread and poller_thread.is_alive():
        poller_thread.join(timeout=10)
    
    # Close database connection
    await connection_manager.close()
    logging.info("Application shutdown complete")

# API Endpoints
@app.post("/api/accounts", response_model=EmailAccountResponse, status_code=status.HTTP_201_CREATED)
async def create_account(
    account_data: EmailAccountCreate,
    account_manager: EmailAccountManager = Depends(get_account_manager)
):
    """Create a new email account"""
    try:
        # Create the account
        account = await account_manager.create_account(
            email=account_data.email,
            password=account_data.password,
            provider_type=account_data.provider_type,
            imap_host=account_data.imap_host,
            imap_port=account_data.imap_port,
            imap_use_ssl=account_data.imap_use_ssl,
            smtp_host=account_data.smtp_host,
            smtp_port=account_data.smtp_port,
            smtp_use_ssl=account_data.smtp_use_ssl,
            smtp_use_tls=account_data.smtp_use_tls
        )
        
        # Convert to response model
        return EmailAccountResponse(
            id=account.account_id,
            email=account.email,
            provider_type=account.provider_type.value,
            is_active=account.is_active,
            created_at=account.created_at,
            updated_at=account.updated_at,
            last_synced_at=account.last_synced_at,
            sync_status=account.sync_status,
            error_count=account.error_count,
            last_error=account.last_error
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to create account: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/accounts", response_model=List[EmailAccountResponse])
async def list_accounts(
    is_active: Optional[bool] = None,
    provider_type: Optional[ProviderTypeStr] = None,
    account_manager: EmailAccountManager = Depends(get_account_manager)
):
    """List all email accounts"""
    try:
        accounts = await account_manager.list_accounts(
            is_active=is_active,
            provider_type=provider_type.value if provider_type else None
        )
        
        return [
            EmailAccountResponse(
                id=acc.account_id,
                email=acc.email,
                provider_type=acc.provider_type.value,
                is_active=acc.is_active,
                created_at=acc.created_at,
                updated_at=acc.updated_at,
                last_synced_at=acc.last_synced_at,
                sync_status=acc.sync_status,
                error_count=acc.error_count,
                last_error=acc.last_error
            )
            for acc in accounts
        ]
        
    except Exception as e:
        logger.exception(f"Failed to list accounts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/accounts/{account_id}", response_model=EmailAccountResponse)
async def get_account(
    account_id: int,
    account_manager: EmailAccountManager = Depends(get_account_manager)
):
    """Get account by ID"""
    try:
        account = await account_manager.get_account(account_id)
        if not account:
            raise HTTPException(status_code=404, detail="Account not found")
            
        return EmailAccountResponse(
            id=account.account_id,
            email=account.email,
            provider_type=account.provider_type.value,
            is_active=account.is_active,
            created_at=account.created_at,
            updated_at=account.updated_at,
            last_synced_at=account.last_synced_at,
            sync_status=account.sync_status,
            error_count=account.error_count,
            last_error=account.last_error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get account {account_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/accounts/{account_id}/sync", response_model=SyncResponse)
async def sync_account(
    account_id: int,
    background_tasks: BackgroundTasks,
    sync_service: EmailSyncService = Depends(get_sync_service)
):
    """Start a sync for a specific account"""
    try:
        # Check if sync is already running for this account
        if account_id in _sync_tasks and not _sync_tasks[account_id].done():
            return SyncResponse(
                success=False,
                message=f"Sync already in progress for account {account_id}"
            )
        
        # Start sync in background
        task = asyncio.create_task(sync_service.start_sync(account_id))
        _sync_tasks[account_id] = task
        
        # Add callback to clean up task when done
        def cleanup_task(fut):
            if account_id in _sync_tasks:
                del _sync_tasks[account_id]
        
        task.add_done_callback(cleanup_task)
        
        return SyncResponse(
            success=True,
            message=f"Sync started for account {account_id}",
            task_id=str(id(task))
        )
        
    except Exception as e:
        logger.exception(f"Failed to start sync for account {account_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start sync")

@app.post("/api/sync/all", response_model=SyncResponse)
async def sync_all_accounts(
    background_tasks: BackgroundTasks,
    sync_service: EmailSyncService = Depends(get_sync_service)
):
    """Start sync for all active accounts"""
    try:
        # Create a task ID for this batch
        task_id = str(id(object()))
        
        # Start sync in background
        task = asyncio.create_task(sync_service.sync_all_active_accounts())
        
        return SyncResponse(
            success=True,
            message="Sync started for all active accounts",
            task_id=task_id
        )
        
    except Exception as e:
        logger.exception(f"Failed to start sync for all accounts: {e}")
        raise HTTPException(status_code=500, detail="Failed to start sync")

@app.get("/api/providers", response_model=List[Dict[str, Any]])
async def list_providers():
    """List all available email providers"""
    providers = []
    for provider in provider_registry._providers.values():
        providers.append({
            'id': provider.type.value,
            'name': provider.name,
            'domains': provider.domains,
            'auth_types': [auth.value for auth in provider.auth_types],
            'app_password_required': provider.app_password_required,
            'imap': {
                'host': provider.imap_host,
                'port': provider.imap_port,
                'ssl': provider.imap_use_ssl
            },
            'smtp': {
                'host': provider.smtp_host,
                'port': provider.smtp_port,
                'ssl': provider.smtp_use_ssl,
                'tls': provider.smtp_use_tls
            } if provider.smtp_host else None
        })
    return providers

# Health check endpoints
@app.get("/")
async def root_check():
    """Root endpoint for Render health checks"""
    return {"status": "ok", "message": "Email API is running"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint for Render and other platforms
    
    Returns:
        dict: Health status including poller information
    """
    return {
        "status": "healthy",
        "poller": {
            "status": "running" if poller_running else "stopped",
            "last_run": last_run.isoformat() if last_run else None,
            "next_run_in": (
                f"{300 - (datetime.utcnow() - last_run).total_seconds():.0f}s" 
                if last_run and (datetime.utcnow() - last_run).total_seconds() < 300 
                else "now"
            ) if last_run else "pending"
        },
        "database": {
            "connected": connection_manager.pool is not None,
            "pool_size": connection_manager.pool.get_size() if connection_manager.pool else 0
        }
    }

@app.get("/healthz")
async def healthz_check():
    """Standard health check endpoint for Kubernetes and other platforms"""
    return {"status": "ok"}

@app.post("/trigger-polling")
async def trigger_polling():
    """
    Trigger manual email polling
    This endpoint can be called by an external cron job service
    """
    try:
        # Import the main function from imap_poller
        from imap_poller import main as run_imap_poller
        
        # Run the poller in a separate thread to avoid blocking
        import threading
        thread = threading.Thread(target=asyncio.run, args=(run_imap_poller(),))
        thread.daemon = True
        thread.start()
        
        return {"status": "success", "message": "Email polling started in background"}
    except Exception as e:
        logger.error(f"Error triggering email polling: {str(e)}")
        return {"status": "error", "message": str(e)}

def format_mailing_address(address: Optional[MailingAddress] = None) -> str:
    """Format a mailing address from a MailingAddress model as HTML."""
    if not address:
        return ""
    
    parts = []
    if address.name:
        parts.append(address.name)
    if address.address_line1:
        parts.append(address.address_line1)
    if address.address_line2:
        parts.append(address.address_line2)
    
    city_parts = []
    if address.city:
        city_parts.append(address.city)
    if address.state:
        city_parts.append(address.state)
    if address.postal_code:
        city_parts.append(address.postal_code)
    
    if city_parts:
        parts.append(", ".join(city_parts))
    if address.country:
        parts.append(address.country)
    
    # Join with HTML line breaks
    return "<br>".join(parts)

# Email sending endpoint
@app.post("/send-email")
async def send_email(email_data: EmailRequest):
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
        # Use reply_to if provided, otherwise use sender
        reply_to = email_data.reply_to or email_data.sender
        
        # Create email message
        msg = MIMEMultipart('alternative')
        
        # Set basic headers
        msg['Subject'] = email_data.subject
        msg['From'] = f'Thorsignia Support <{email_data.sender}>'
        msg['To'] = email_data.recipient
        msg['Reply-To'] = reply_to
        msg['Date'] = email.utils.formatdate(localtime=True)
        msg['Message-ID'] = email.utils.make_msgid(domain=email_data.sender.split('@')[-1])
        
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
                        {formatted_address}
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
        def clean_email(email_str):
            return ''.join(c for c in email_str if ord(c) < 128).strip()
            
        clean_recipient = clean_email(email_data.recipient)
        clean_sender = clean_email(email_data.sender)
        
        # Send email using SMTP
        with smtplib.SMTP(email_data.smtp_host, email_data.smtp_port, timeout=30) as server:
            server.ehlo()
            if server.has_extn('STARTTLS'):
                server.starttls()
                server.ehlo()
            server.login(clean_sender, email_data.password)
            
            # Update headers with cleaned emails
            msg.replace_header('From', clean_sender)
            msg.replace_header('To', clean_recipient)
            
            # Send email
            raw_message = msg.as_string()
            server.sendmail(clean_sender, [clean_recipient], raw_message)
            
            logger.info(f"Email sent successfully to {clean_recipient}")
            
            return {
                "status": "success",
                "message": "Email sent successfully",
                "to": clean_recipient,
                "subject": email_data.subject
            }
            
    except Exception as e:
        import traceback
        error_type = type(e).__name__
        error_msg = f"Error Type: {error_type}"
        error_msg += f"\nError Message: {str(e)}"
        logger.error(f"Email sending failed: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "error": f"Failed to send email: {str(e)}",
                "type": error_type
            }
        )

# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "email_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
