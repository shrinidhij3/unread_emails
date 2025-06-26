import os
import imaplib
import ssl
import logging
import asyncio
import base64
import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any, Callable, TypeVar, Type, Union
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet, InvalidToken
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Type variables
T = TypeVar('T', bound='ProviderConfig')

# Enums
class ProviderType(str, Enum):
    GMAIL = 'gmail'
    OFFICE365 = 'office365'
    CPANEL = 'cpanel'
    ZOHO = 'zoho'
    CUSTOM = 'custom'

class AuthType(str, Enum):
    PASSWORD = 'password'
    OAUTH2 = 'oauth2'
    XOAUTH2 = 'xoauth2'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('email_service.log')
    ]
)
logger = logging.getLogger(__name__)

# Exception Classes
class EmailServiceError(Exception):
    """Base exception for all email service errors"""
    pass

class IMAPConnectionError(EmailServiceError):
    """Raised when IMAP connection or authentication fails"""
    pass

class EmailSyncError(EmailServiceError):
    """Raised when email synchronization fails"""
    def __init__(self, message: str, error_type: str = 'unknown', 
                 account_id: Optional[int] = None, email_id: Optional[str] = None):
        self.message = message
        self.error_type = error_type
        self.account_id = account_id
        self.email_id = email_id
        self.timestamp = datetime.now(timezone.utc)
        super().__init__(self.message)

# Crypto Service for secure password storage
class CryptoService:
    """Service for encrypting and decrypting sensitive data"""
    
    def __init__(self, secret_key: Optional[bytes] = None):
        self.secret_key = secret_key or os.urandom(32)
        
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive a key from password and salt"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=390000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def encrypt(self, data: str, password: str) -> Tuple[bytes, bytes]:
        """Encrypt data with password, returns (encrypted_data, salt)"""
        salt = os.urandom(16)
        key = self._derive_key(password, salt)
        f = Fernet(key)
        encrypted = f.encrypt(data.encode())
        return encrypted, salt

    def decrypt(self, encrypted_data: bytes, password: str, salt: bytes) -> str:
        """Decrypt data with password and salt"""
        key = self._derive_key(password, salt)
        f = Fernet(key)
        try:
            return f.decrypt(encrypted_data).decode()
        except InvalidToken:
            raise ValueError("Invalid password or corrupted data")

# Provider Configuration
@dataclass
class ProviderConfig:
    """Configuration for an email provider"""
    name: str
    type: ProviderType
    imap_host: str
    imap_port: int
    imap_use_ssl: bool = True
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_use_ssl: bool = True
    smtp_use_tls: bool = True
    auth_types: List[AuthType] = field(default_factory=lambda: [AuthType.PASSWORD])
    domains: List[str] = field(default_factory=list)
    oauth_scopes: List[str] = field(default_factory=list)
    app_password_required: bool = False
    custom_data: Dict[str, Any] = field(default_factory=dict)

    def format_imap_host(self, email: str) -> str:
        """Format IMAP host with domain if needed"""
        if '{domain}' in self.imap_host:
            domain = email.split('@')[-1] if '@' in email else ''
            return self.imap_host.format(domain=domain)
        return self.imap_host

class ProviderRegistry:
    """Registry for email provider configurations"""
    
    def __init__(self):
        self._providers: Dict[str, ProviderConfig] = {}
        self._domain_map: Dict[str, str] = {}
        self._init_default_providers()
    
    def _init_default_providers(self):
        """Initialize with common email providers"""
        default_providers = [
            ProviderConfig(
                name='Gmail',
                type=ProviderType.GMAIL,
                imap_host='imap.gmail.com',
                imap_port=993,
                smtp_host='smtp.gmail.com',
                smtp_port=587,
                smtp_use_ssl=True,
                smtp_use_tls=True,
                auth_types=[AuthType.PASSWORD, AuthType.OAUTH2],
                domains=['gmail.com', 'googlemail.com'],
                oauth_scopes=['https://mail.google.com/'],
                app_password_required=True
            ),
            ProviderConfig(
                name='Office 365',
                type=ProviderType.OFFICE365,
                imap_host='outlook.office365.com',
                imap_port=993,
                smtp_host='smtp.office365.com',
                smtp_port=587,
                smtp_use_ssl=True,
                smtp_use_tls=True,
                auth_types=[AuthType.PASSWORD, AuthType.OAUTH2],
                domains=['outlook.com', 'office365.com', 'microsoft.com'],
                oauth_scopes=['https://outlook.office.com/IMAP.AccessAsUser.All', 'offline_access']
            ),
            ProviderConfig(
                name='cPanel',
                type=ProviderType.CPANEL,
                imap_host='mail.{domain}',
                imap_port=993,
                smtp_host='mail.{domain}',
                smtp_port=465,
                smtp_use_ssl=True,
                smtp_use_tls=False,
                auth_types=[AuthType.PASSWORD],
                domains=[]  # Matches any domain not handled by other providers
            ),
            ProviderConfig(
                name='Zoho Mail',
                type=ProviderType.ZOHO,
                imap_host='imap.zoho.com',
                imap_port=993,
                smtp_host='smtp.zoho.com',
                smtp_port=465,
                smtp_use_ssl=True,
                smtp_use_tls=True,
                auth_types=[AuthType.PASSWORD],
                domains=['zoho.com', 'zohomail.com'],
                app_password_required=True
            )
        ]
        
        for provider in default_providers:
            self.register(provider)
    
    def register(self, provider: ProviderConfig):
        """Register a provider configuration"""
        self._providers[provider.type.value] = provider
        for domain in provider.domains:
            self._domain_map[domain.lower()] = provider.type.value
    
    def get_provider(self, provider_type: Union[str, ProviderType]) -> Optional[ProviderConfig]:
        """Get provider by type"""
        if isinstance(provider_type, ProviderType):
            provider_type = provider_type.value
        return self._providers.get(provider_type.lower())
    
    def detect_provider(self, email: str) -> Optional[ProviderConfig]:
        """Detect provider from email domain"""
        if not email or '@' not in email:
            return None
            
        domain = email.split('@')[-1].lower()
        
        # Try exact domain match first
        if domain in self._domain_map:
            return self.get_provider(self._domain_map[domain])
            
        # Try subdomain matches (e.g., sub.example.com -> example.com)
        domain_parts = domain.split('.')
        if len(domain_parts) > 2:
            parent_domain = '.'.join(domain_parts[-2:])
            if parent_domain in self._domain_map:
                return self.get_provider(self._domain_map[parent_domain])
        
        # Default to cPanel for custom domains
        return self.get_provider(ProviderType.CPANEL)

# Initialize global provider registry
provider_registry = ProviderRegistry()

# IMAP Connection Manager
class IMAPConnectionManager:
    """Manages IMAP connections with retry and error handling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f'{__name__}.IMAPConnectionManager')
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(IMAPConnectionError),
        reraise=True
    )
    async def connect(self) -> imaplib.IMAP4_SSL:
        """Establish an IMAP connection with retry logic"""
        try:
            context = self._create_ssl_context()
            
            if self.config.get('imap_use_ssl', True):
                conn = imaplib.IMAP4_SSL(
                    host=self.config['imap_host'],
                    port=self.config['imap_port'],
                    ssl_context=context
                )
            else:
                conn = imaplib.IMAP4(
                    host=self.config['imap_host'],
                    port=self.config['imap_port']
                )
                if self.config.get('starttls', False):
                    conn.starttls(ssl_context=context)
            
            # Set timeouts
            conn.timeout = self.config.get('timeout', 30)
            
            return conn
            
        except (imaplib.IMAP4.error, ssl.SSLError, OSError) as e:
            error_msg = f"IMAP connection failed to {self.config.get('imap_host')}:{self.config.get('imap_port')} - {str(e)}"
            self.logger.error(error_msg)
            raise IMAPConnectionError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error connecting to IMAP: {str(e)}"
            self.logger.exception(error_msg)
            raise IMAPConnectionError(error_msg)
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with appropriate settings"""
        context = ssl.create_default_context()
        
        if self.config.get('allow_insecure_ssl', False):
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        
        # Configure SSL/TLS versions
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        return context
    
    async def login(self, conn: imaplib.IMAP4_SSL, email: str, password: str) -> bool:
        """Authenticate with the IMAP server"""
        try:
            conn.login(email, password)
            self.logger.info(f"Successfully authenticated {email}")
            return True
        except imaplib.IMAP4.error as e:
            self.logger.error(f"IMAP login failed for {email}: {str(e)}")
            return False
    
    async def select_folder(self, conn: imaplib.IMAP4_SSL, folder: str = 'INBOX', 
                          readonly: bool = True) -> bool:
        """Select an IMAP folder"""
        try:
            status, _ = conn.select(folder, readonly=readonly)
            if status == 'OK':
                self.logger.debug(f"Selected folder: {folder}")
                return True
            return False
        except imaplib.IMAP4.error as e:
            self.logger.error(f"Failed to select folder {folder}: {str(e)}")
            return False
    
    async def close_connection(self, conn: Optional[imaplib.IMAP4_SSL]):
        """Close the IMAP connection"""
        if conn:
            try:
                conn.logout()
            except Exception as e:
                self.logger.warning(f"Error closing IMAP connection: {str(e)}")

# Initialize global connection manager
connection_manager = IMAPConnectionManager({})

# Database Models and Operations
class EmailAccount:
    """Represents an email account with provider-specific settings"""
    
    def __init__(self, 
                 email: str,
                 password_encrypted: bytes,
                 password_salt: bytes,
                 provider_type: Union[str, ProviderType],
                 imap_host: str,
                 imap_port: int,
                 imap_use_ssl: bool = True,
                 smtp_host: Optional[str] = None,
                 smtp_port: Optional[int] = None,
                 smtp_use_ssl: bool = True,
                 smtp_use_tls: bool = True,
                 is_active: bool = True,
                 account_id: Optional[int] = None,
                 created_at: Optional[datetime] = None,
                 updated_at: Optional[datetime] = None,
                 last_synced_at: Optional[datetime] = None,
                 sync_status: str = 'idle',
                 error_count: int = 0,
                 last_error: Optional[str] = None,
                 custom_data: Optional[Dict[str, Any]] = None):
        
        self.account_id = account_id
        self.email = email.lower()
        self.password_encrypted = password_encrypted
        self.password_salt = password_salt
        self.provider_type = ProviderType(provider_type) if isinstance(provider_type, str) else provider_type
        self.imap_host = imap_host
        self.imap_port = imap_port
        self.imap_use_ssl = imap_use_ssl
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_use_ssl = smtp_use_ssl
        self.smtp_use_tls = smtp_use_tls
        self.is_active = is_active
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)
        self.last_synced_at = last_synced_at
        self.sync_status = sync_status
        self.error_count = error_count
        self.last_error = last_error
        self.custom_data = custom_data or {}
    
    @classmethod
    def create(
        cls,
        email: str,
        password: str,
        provider_type: Union[str, ProviderType],
        crypto_service: Optional[CryptoService] = None
    ) -> 'EmailAccount':
        """Create a new email account with encrypted password"""
        crypto = crypto_service or CryptoService()
        password_encrypted, password_salt = crypto.encrypt(password, password)
        
        # Get provider config
        provider = provider_registry.get_provider(provider_type)
        if not provider:
            raise ValueError(f"Unsupported provider: {provider_type}")
        
        # Format IMAP host with domain if needed
        imap_host = provider.format_imap_host(email)
        
        return cls(
            email=email,
            password_encrypted=password_encrypted,
            password_salt=password_salt,
            provider_type=provider_type,
            imap_host=imap_host,
            imap_port=provider.imap_port,
            imap_use_ssl=provider.imap_use_ssl,
            smtp_host=provider.smtp_host,
            smtp_port=provider.smtp_port,
            smtp_use_ssl=provider.smtp_use_ssl,
            smtp_use_tls=provider.smtp_use_tls,
            custom_data={
                'provider_name': provider.name,
                'auth_types': [auth.value for auth in provider.auth_types],
                'app_password_required': provider.app_password_required
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'email': self.email,
            'password_encrypted': self.password_encrypted,
            'password_salt': self.password_salt,
            'provider_type': self.provider_type.value,
            'imap_host': self.imap_host,
            'imap_port': self.imap_port,
            'imap_use_ssl': self.imap_use_ssl,
            'smtp_host': self.smtp_host,
            'smtp_port': self.smtp_port,
            'smtp_use_ssl': self.smtp_use_ssl,
            'smtp_use_tls': self.smtp_use_tls,
            'is_active': self.is_active,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'last_synced_at': self.last_synced_at,
            'sync_status': self.sync_status,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'custom_data': self.custom_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmailAccount':
        """Create from database dictionary"""
        return cls(
            account_id=data.get('id'),
            email=data['email'],
            password_encrypted=data['password_encrypted'],
            password_salt=data['password_salt'],
            provider_type=data['provider_type'],
            imap_host=data['imap_host'],
            imap_port=data['imap_port'],
            imap_use_ssl=data.get('imap_use_ssl', True),
            smtp_host=data.get('smtp_host'),
            smtp_port=data.get('smtp_port'),
            smtp_use_ssl=data.get('smtp_use_ssl', True),
            smtp_use_tls=data.get('smtp_use_tls', True),
            is_active=data.get('is_active', True),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            last_synced_at=data.get('last_synced_at'),
            sync_status=data.get('sync_status', 'idle'),
            error_count=data.get('error_count', 0),
            last_error=data.get('last_error'),
            custom_data=data.get('custom_data', {})
        )
    
    def get_connection_config(self) -> Dict[str, Any]:
        """Get IMAP connection configuration"""
        return {
            'imap_host': self.imap_host,
            'imap_port': self.imap_port,
            'imap_use_ssl': self.imap_use_ssl,
            'timeout': 30,
            'allow_insecure_ssl': False
        }

class EmailAccountManager:
    """Manages email account operations with the database"""
    
    def __init__(self, db_pool, crypto_service: Optional[CryptoService] = None):
        self.db_pool = db_pool
        self.crypto_service = crypto_service or CryptoService()
        self.logger = logging.getLogger(f'{__name__}.EmailAccountManager')
    
    async def create_account(
        self,
        email: str,
        password: str,
        provider_type: Union[str, ProviderType],
        **kwargs
    ) -> EmailAccount:
        """Create and save a new email account"""
        # Validate email format
        if not self._validate_email(email):
            raise ValueError("Invalid email format")
        
        # Auto-detect provider if not specified
        if provider_type == 'auto':
            provider = provider_registry.detect_provider(email)
            if not provider:
                raise ValueError("Could not detect email provider")
            provider_type = provider.type
        
        # Create account
        account = EmailAccount.create(
            email=email,
            password=password,
            provider_type=provider_type,
            crypto_service=self.crypto_service
        )
        
        # Save to database
        return await self.save_account(account)
    
    async def save_account(self, account: EmailAccount) -> EmailAccount:
        """Save account to database"""
        data = account.to_dict()
        
        async with self.db_pool.acquire() as conn:
            if account.account_id is None:
                # Insert new account
                query = """
                    INSERT INTO email_accounts (
                        email, password_encrypted, password_salt, provider_type,
                        imap_host, imap_port, imap_use_ssl,
                        smtp_host, smtp_port, smtp_use_ssl, smtp_use_tls,
                        is_active, custom_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    RETURNING id, created_at, updated_at
                """
                
                result = await conn.fetchrow(
                    query,
                    data['email'],
                    data['password_encrypted'],
                    data['password_salt'],
                    data['provider_type'],
                    data['imap_host'],
                    data['imap_port'],
                    data['imap_use_ssl'],
                    data['smtp_host'],
                    data['smtp_port'],
                    data['smtp_use_ssl'],
                    data['smtp_use_tls'],
                    data['is_active'],
                    data['custom_data']
                )
                
                account.account_id = result['id']
                account.created_at = result['created_at']
                account.updated_at = result['updated_at']
                
            else:
                # Update existing account
                query = """
                    UPDATE email_accounts
                    SET
                        email = $2,
                        password_encrypted = $3,
                        password_salt = $4,
                        provider_type = $5,
                        imap_host = $6,
                        imap_port = $7,
                        imap_use_ssl = $8,
                        smtp_host = $9,
                        smtp_port = $10,
                        smtp_use_ssl = $11,
                        smtp_use_tls = $12,
                        is_active = $13,
                        updated_at = NOW(),
                        custom_data = $14
                    WHERE id = $1
                    RETURNING updated_at
                """
                
                result = await conn.fetchrow(
                    query,
                    account.account_id,
                    data['email'],
                    data['password_encrypted'],
                    data['password_salt'],
                    data['provider_type'],
                    data['imap_host'],
                    data['imap_port'],
                    data['imap_use_ssl'],
                    data['smtp_host'],
                    data['smtp_port'],
                    data['smtp_use_ssl'],
                    data['smtp_use_tls'],
                    data['is_active'],
                    data['custom_data']
                )
                
                account.updated_at = result['updated_at']
            
            return account
    
    async def get_account(self, account_id: int) -> Optional[EmailAccount]:
        """Get account by ID"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                'SELECT * FROM email_accounts WHERE id = $1',
                account_id
            )
            
            if not row:
                return None
                
            return EmailAccount.from_dict(dict(row))
    
    async def get_account_by_email(self, email: str) -> Optional[EmailAccount]:
        """Get account by email"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                'SELECT * FROM email_accounts WHERE email = $1',
                email.lower()
            )
            
            if not row:
                return None
                
            return EmailAccount.from_dict(dict(row))
    
    async def list_accounts(
        self,
        is_active: Optional[bool] = None,
        provider_type: Optional[Union[str, ProviderType]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[EmailAccount]:
        """List accounts with optional filters"""
        query = 'SELECT * FROM email_accounts WHERE 1=1'
        params = []
        
        if is_active is not None:
            query += f' AND is_active = ${len(params) + 1}'
            params.append(is_active)
            
        if provider_type is not None:
            provider_str = provider_type.value if isinstance(provider_type, ProviderType) else provider_type
            query += f' AND provider_type = ${len(params) + 1}'
            params.append(provider_str)
        
        query += f' ORDER BY created_at DESC LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}'
        params.extend([limit, offset])
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [EmailAccount.from_dict(dict(row)) for row in rows]
    
    async def delete_account(self, account_id: int) -> bool:
        """Delete an account"""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                'DELETE FROM email_accounts WHERE id = $1',
                account_id
            )
            return result == 'DELETE 1'
    
    async def update_sync_status(
        self,
        account_id: int,
        status: str,
        error: Optional[str] = None
    ) -> bool:
        """Update account sync status and error information"""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE email_accounts
                SET
                    sync_status = $1,
                    last_synced_at = CASE WHEN $1 = 'success' THEN NOW() ELSE last_synced_at END,
                    error_count = CASE WHEN $1 = 'error' THEN error_count + 1 ELSE 0 END,
                    last_error = $2,
                    updated_at = NOW()
                WHERE id = $3
                """,
                status,
                error,
                account_id
            )
            return result == 'UPDATE 1'
    
    @staticmethod
    def _validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        return bool(re.match(pattern, email))


class EmailSyncService:
    """Service for synchronizing emails from IMAP accounts"""
    
    def __init__(
        self,
        db_pool,
        crypto_service: Optional[CryptoService] = None,
        account_manager: Optional[EmailAccountManager] = None,
        connection_manager: Optional[IMAPConnectionManager] = None
    ):
        self.db_pool = db_pool
        self.crypto_service = crypto_service or CryptoService()
        self.account_manager = account_manager or EmailAccountManager(db_pool, self.crypto_service)
        self.connection_manager = connection_manager or connection_manager
        self.logger = logging.getLogger(f'{__name__}.EmailSyncService')
        self.running = False
    
    async def start_sync(self, account_id: int) -> bool:
        """Start syncing a single email account"""
        try:
            # Get the account
            account = await self.account_manager.get_account(account_id)
            if not account:
                self.logger.error(f"Account {account_id} not found")
                return False
            
            # Update sync status
            await self.account_manager.update_sync_status(account_id, 'syncing')
            
            # Get decrypted password
            password = self._decrypt_password(account)
            
            # Connect to IMAP server
            conn = await self._connect_to_imap(account, password)
            if not conn:
                await self.account_manager.update_sync_status(
                    account_id, 
                    'error', 
                    'Failed to connect to IMAP server'
                )
                return False
            
            try:
                # Process emails
                await self._process_emails(account, conn)
                
                # Update last sync time
                await self.account_manager.update_sync_status(account_id, 'success')
                return True
                
            except Exception as e:
                self.logger.exception(f"Error processing emails for account {account_id}")
                await self.account_manager.update_sync_status(
                    account_id, 
                    'error', 
                    str(e)
                )
                return False
                
            finally:
                # Always close the connection
                await self.connection_manager.close_connection(conn)
                
        except Exception as e:
            self.logger.exception(f"Unexpected error in sync for account {account_id}")
            return False
    
    async def sync_all_active_accounts(self) -> Dict[int, bool]:
        """Sync all active email accounts"""
        results = {}
        accounts = await self.account_manager.list_accounts(is_active=True)
        
        for account in accounts:
            if not account.account_id:
                continue
                
            success = await self.start_sync(account.account_id)
            results[account.account_id] = success
            
            # Add a small delay between account syncs
            await asyncio.sleep(1)
            
        return results
    
    async def _connect_to_imap(
        self, 
        account: EmailAccount, 
        password: str
    ) -> Optional[imaplib.IMAP4_SSL]:
        """Establish IMAP connection with retry logic"""
        try:
            # Get connection config
            config = account.get_connection_config()
            
            # Try to connect
            conn = await self.connection_manager.connect()
            
            # Login
            if not await self.connection_manager.login(conn, account.email, password):
                self.logger.error(f"Failed to login to {account.email}")
                return None
                
            # Select INBOX
            if not await self.connection_manager.select_folder(conn, 'INBOX'):
                self.logger.error(f"Failed to select INBOX for {account.email}")
                return None
                
            return conn
            
        except Exception as e:
            self.logger.exception(f"Error connecting to IMAP for {account.email}")
            return None
    
    async def _process_emails(self, account: EmailAccount, conn: imaplib.IMAP4_SSL):
        """Process emails for an account"""
        try:
            # Search for unseen messages
            status, messages = conn.search(None, 'UNSEEN')
            if status != 'OK':
                self.logger.error(f"Failed to search for unseen messages: {messages}")
                return
                
            message_nums = messages[0].split()
            if not message_nums:
                self.logger.info(f"No new messages for {account.email}")
                return
                
            self.logger.info(f"Found {len(message_nums)} new messages for {account.email}")
            
            # Process each message
            for num in message_nums:
                try:
                    await self._process_single_email(account, conn, num)
                except Exception as e:
                    self.logger.exception(f"Error processing message {num} for {account.email}")
        
        except Exception as e:
            self.logger.exception(f"Error processing emails for {account.email}")
            raise
    
    async def _process_single_email(
        self, 
        account: EmailAccount, 
        conn: imaplib.IMAP4_SSL, 
        message_num: bytes
    ):
        """Process a single email message"""
        try:
            # Fetch the email
            status, msg_data = conn.fetch(message_num, '(RFC822)')
            if status != 'OK' or not msg_data or not msg_data[0]:
                self.logger.error(f"Failed to fetch message {message_num}")
                return
                
            # Parse the email
            raw_email = msg_data[0][1]
            email_message = email.message_from_bytes(raw_email)
            
            # Extract email details
            msg_id = email_message.get('Message-ID', '').strip('<>')
            subject = email_message.get('Subject', 'No Subject')
            from_ = email_message.get('From', '')
            to = email_message.get('To', '')
            date = email_message.get('Date', '')
            
            self.logger.info(f"Processing email: {subject} (ID: {msg_id})")
            
            # Process attachments
            attachments = []
            for part in email_message.walk():
                if part.get_content_maintype() == 'multipart':
                    continue
                    
                if part.get('Content-Disposition') is None:
                    continue
                    
                filename = part.get_filename()
                if not filename:
                    continue
                    
                # Save attachment
                attachment_data = part.get_payload(decode=True)
                if attachment_data:
                    attachments.append({
                        'filename': filename,
                        'content_type': part.get_content_type(),
                        'size': len(attachment_data),
                        'data': attachment_data
                    })
            
            # Here you would typically process the email (save to DB, forward, etc.)
            # For now, just log the details
            self.logger.info(f"Processed email from {from_} to {to} with {len(attachments)} attachments")
            
            # Mark as read
            conn.store(message_num, '+FLAGS', '\\Seen')
            
        except Exception as e:
            self.logger.exception(f"Error processing message {message_num}")
            raise
    
    def _decrypt_password(self, account: EmailAccount) -> str:
        """Decrypt the account password"""
        try:
            return self.crypto_service.decrypt(
                account.password_encrypted,
                account.email,  # Using email as password for decryption
                account.password_salt
            )
        except Exception as e:
            self.logger.exception(f"Failed to decrypt password for {account.email}")
            raise ValueError("Invalid password or corrupted data")
