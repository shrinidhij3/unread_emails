import asyncio
import asyncpg
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def add_email_account():
    """Add a new email account to the credentials_email table."""
    conn = None
    try:
        # Get database URL from environment
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is not set")
            
        # Parse the database URL
        from urllib.parse import urlparse, parse_qs
        
        # Handle different URL formats
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
            
        result = urlparse(database_url)
        username = result.username
        password = result.password
        database = result.path[1:]  # Remove leading '/'
        hostname = result.hostname
        port = result.port or 5432
        
        # Parse query parameters for SSL
        query = parse_qs(result.query)
        ssl_mode = 'require' if query.get('sslmode', [''])[0] == 'require' else None
        
        db_config = {
            "user": username,
            "password": password,
            "database": database,
            "host": hostname,
            "port": port,
            "ssl": ssl_mode
        }
        if not db_config["host"]:
            raise ValueError("DB_HOST environment variable is not set")
        
        # Get email account details from user
        print("\n📧 Add New Email Account")
        print("=" * 50)
        
        email = input("Email address: ").strip()
        if not email or '@' not in email:
            print("❌ Invalid email address")
            return
            
        password = input("Password: ").strip()
        if not password:
            print("❌ Password cannot be empty")
            return
            
        # Optional: Get IMAP/SMTP settings or use auto-detection
        print("\nIMAP Settings (press Enter to use auto-detection):")
        imap_host = input(f"IMAP Host [mail.{email.split('@')[1]}]: ").strip()
        imap_port = input("IMAP Port [993]: ").strip() or "993"
        use_ssl = input("Use SSL? (y/n) [y]: ").strip().lower() != 'n'
        
        print("\nSMTP Settings (press Enter to use auto-detection):")
        smtp_host = input(f"SMTP Host [smtp.{email.split('@')[1]}]: ").strip()
        smtp_port = input("SMTP Port [587]: ").strip() or "587"
        smtp_use_ssl = input("Use SSL for SMTP? (y/n) [n]: ").strip().lower() == 'y'
        smtp_use_tls = not smtp_use_ssl  # Default to TLS if not using SSL
        
        # Connect to database
        print("\n🔌 Connecting to database...")
        conn = await asyncpg.connect(**db_config)
        
        # Check if email already exists
        exists = await conn.fetchval(
            'SELECT 1 FROM credentials_email WHERE email = $1',
            email
        )
        
        if exists:
            print(f"❌ Email {email} already exists in the database")
            return
        
        # Get provider from email domain if not provided
        provider = input("Provider (e.g., gmail, outlook, custom): ").strip().lower()
        if not provider:
            domain = email.split('@')[-1].split('.')[0]
            provider = domain if domain in ['gmail', 'outlook', 'yahoo', 'aol'] else 'custom'
        
        # Get name for the account
        name = input(f"Name for this account [{email}]: ").strip() or email
        
        # Insert new account with all required fields
        await conn.execute('''
            INSERT INTO credentials_email (
                name,
                email, 
                password,
                provider,
                imap_host, 
                imap_port, 
                use_ssl,
                secure,
                smtp_host, 
                smtp_port,
                is_processed,
                created_at, 
                updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        ''',
            name,
            email,
            password,  # In production, encrypt this password
            provider,
            imap_host if imap_host else None,
            int(imap_port) if imap_port else 993,
            use_ssl,  # use_ssl
            smtp_use_tls,  # secure (TLS)
            smtp_host if smtp_host else None,
            int(smtp_port) if smtp_port else 587,
            False,  # is_processed
            datetime.now(),
            datetime.now()
        )
        
        print("\n✅ Successfully added email account:")
        print(email)
        print("\nNote: The password is stored in plain text. In production, ensure to encrypt it.")
        
    except Exception as e:
        print("\n❌ Error:")
        print(e)
    finally:
        if conn:
            await conn.close()
            print("\n🔌 Database connection closed.")

if __name__ == "__main__":
    print("🚀 Email Account Management")
    print("=" * 50)
    asyncio.run(add_email_account())
