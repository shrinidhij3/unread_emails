import asyncio
import asyncpg
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def add_email_account():
    conn = None
    try:
        # Database configuration from environment variables
        db_config = {
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_NAME", "railway"),
            "host": os.getenv("DB_HOST"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "command_timeout": int(os.getenv("DB_COMMAND_TIMEOUT", "10"))
        }
        
        print("🔍 Connecting to the database...")
        conn = await asyncpg.connect(**db_config)
        
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
                
            # Insert new account
            await conn.execute('''
                INSERT INTO credentials_email (email, password, created_at, updated_at)
                VALUES ($1, $2, $3, $4)
            ''', email, password, datetime.now(), datetime.now())
            
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
