import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, validator
from enum import Enum
import asyncpg

from email_service import (
    ProviderType,
    EmailAccount,
    EmailAccountManager,
    EmailSyncService,
    CryptoService,
    provider_registry,
    connection_manager
)

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
        _sync_service = EmailSyncService(_db_pool, _account_manager=_account_manager)
    return _sync_service

# Startup and shutdown events
@app.on_event("startup")
async def startup():
    global _db_pool
    try:
        # Initialize database connection pool
        _db_pool = await asyncpg.create_pool(
            user="your_db_user",
            password="your_db_password",
            database="your_db_name",
            host="localhost",
            port=5432,
            min_size=1,
            max_size=10
        )
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database connection pool: {e}")
        raise

@app.on_event("shutdown")
async def shutdown():
    global _db_pool
    if _db_pool:
        await _db_pool.close()
        logger.info("Database connection pool closed")

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

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.utcnow()}

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
