import asyncio
import asyncpg
import os
from datetime import datetime, timezone

async def check_credentials_table():
    """Check the structure of the credentials_email table."""
    try:
        # Get database URL from environment
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            print("❌ DATABASE_URL environment variable is not set")
            return
            
        # Connect to the database
        print("🔍 Connecting to the database...")
        conn = await asyncpg.connect(dsn=database_url)
        
        # Get column information
        print("\n📋 Table structure of 'credentials_email':")
        print("-" * 80)
        print(f"{'Column Name':<25} {'Data Type':<20} {'Nullable':<10} {'Default'}")
        print("-" * 80)
        
        columns = await conn.fetch('''
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = 'credentials_email'
            ORDER BY ordinal_position
        ''')
        
        for col in columns:
            print(f"{col['column_name']:<25} {col['data_type']:<20} {col['is_nullable']:<10} {col['column_default'] or ''}")
        
        # Get a sample row to understand the data format
        print("\n📋 Sample rows from 'credentials_email':")
        print("-" * 80)
        rows = await conn.fetch('SELECT * FROM credentials_email LIMIT 2')
        for i, row in enumerate(rows, 1):
            print(f"\nRow {i}:")
            for k, v in row.items():
                if k.lower() == 'password_encrypted' and v:
                    print(f"  {k}: [REDACTED]")
                else:
                    print(f"  {k}: {v}")
        
        print("\n✅ Database check completed")
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
    finally:
        if 'conn' in locals() and not conn.is_closed():
            await conn.close()

if __name__ == "__main__":
    print("🔍 Checking database structure...")
    asyncio.run(check_credentials_table())
