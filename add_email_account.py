import asyncio
import asyncpg
import os
from datetime import datetime, timezone

# Load environment variables from .env file

async def add_email_account():
    """Add a new email account to the credentials_email table."""
    conn = None
    try:
        # Get database URL from environment
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            # Fallback to individual variables if DATABASE_URL is not set
            db_config = {
                "user": os.getenv("DB_USER", "postgres"),
                "password": os.getenv("DB_PASSWORD"),
                "database": os.getenv("DB_NAME", "railway"),
                "host": os.getenv("DB_HOST"),
                "port": int(os.getenv("DB_PORT", "5432")),
                "command_timeout": int(os.getenv("DB_COMMAND_TIMEOUT", "10")),
                "ssl": os.getenv("DB_SSL", "require")
            }
            
            # Verify required environment variables are set
            if not db_config["password"]:
                raise ValueError("DB_PASSWORD environment variable is not set")
            if not db_config["host"]:
                raise ValueError("DB_HOST environment variable is not set")
                
            # Convert to connection string
            database_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            
        # Handle different URL formats
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
            
        # Connect using the database URL
        print("🔍 Connecting to the database...")
        conn = await asyncpg.connect(dsn=database_url)
        
        # First, check if the table exists and get its structure
        print("\n📋 Checking table structure...")
        table_exists = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'credentials_email')"
        )
        
        if not table_exists:
            print("❌ Error: 'credentials_email' table does not exist")
            return
        
        # Get column information
        columns = await conn.fetch('''
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = 'credentials_email'
            ORDER BY ordinal_position
        ''')
        
        if not columns:
            print("❌ Error: Could not retrieve column information")
            return
            
        print("\n📋 Table structure of 'credentials_email':")
        print("-" * 80)
        print(f"{'Column Name':<20} {'Data Type':<20} {'Nullable':<10} {'Default'}")
        print("-" * 80)
        for col in columns:
            print(f"{col['column_name']:<20} {col['data_type']:<20} {col['is_nullable']:<10} {col['column_default'] or ''}")
        
        # Get a sample row to understand the data format
        try:
            sample = await conn.fetchrow('SELECT * FROM credentials_email LIMIT 1')
            if sample:
                print("\n📋 Sample row data:")
                for k, v in sample.items():
                    print(f"{k}: {v}")
        except Exception as e:
            print("\n⚠️  Could not fetch sample row:")
            print(e)
        
        # Ask user if they want to add a new account
        print("\nWould you like to add a new email account? (y/n): ")
        if input().strip().lower() != 'y':
            return
            
        # Get email account details
        print("\nEnter email address: ")
        email = input().strip()
        print("Enter password: ")
        password = input().strip()
        
        # Get IMAP configuration
        print("\nIMAP Configuration:")
        print("IMAP Host (e.g., imap.gmail.com): ")
        imap_host = input().strip()
        print("IMAP Port (e.g., 993 for SSL, 143 for STARTTLS): ")
        imap_port = int(input().strip())
        print("Use SSL? (y/n, default: y): ")
        imap_use_ssl = input().strip().lower() != 'n'
        
        # Get SMTP configuration (optional)
        print("\nSMTP Configuration (press Enter to skip):")
        print("SMTP Host (e.g., smtp.gmail.com): ")
        smtp_host = input().strip() or None
        smtp_port = None
        if smtp_host:
            print("SMTP Port (e.g., 465 for SSL, 587 for STARTTLS): ")
            port_input = input().strip()
            smtp_port = int(port_input) if port_input else None
        
        # Try to insert the new account
        try:
            # First check if email already exists
            exists = await conn.fetchval(
                'SELECT 1 FROM credentials_email WHERE email = $1',
                email
            )
            
            if exists:
                print("\n❌ Account with email already exists:")
                print(email)
                return
            
            # Get the provider type from email domain
            provider_type = 'gmail' if 'gmail.com' in email.lower() else 'custom'
                
            # Insert new account with all required fields
            await conn.execute('''
                INSERT INTO credentials_email (
                    email, password_encrypted, password_salt, provider_type,
                    imap_host, imap_port, imap_use_ssl,
                    smtp_host, smtp_port, smtp_use_ssl, smtp_use_tls,
                    created_at, updated_at, is_active
                ) VALUES (
                    $1, $2, $3, $4,
                    $5, $6, $7,
                    $8, $9, $10, $11,
                    $12, $13, TRUE
                )
            ''', 
                email,  # email
                password.encode('utf-8'),  # password_encrypted (temporarily storing plaintext, will be encrypted later)
                os.urandom(16),  # password_salt (random salt)
                provider_type,  # provider_type
                imap_host,  # imap_host
                imap_port,  # imap_port
                imap_use_ssl,  # imap_use_ssl
                smtp_host,  # smtp_host
                smtp_port,  # smtp_port
                smtp_host is not None,  # smtp_use_ssl
                smtp_host is not None,  # smtp_use_tls
                datetime.now(timezone.utc),  # created_at
                datetime.now(timezone.utc)  # updated_at
            )
            
            print("\n✅ Successfully added email account:")
            print(email)
            
        except Exception as e:
            print("\n❌ Error adding email account:")
            print(e)
            
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
