#!/usr/bin/env python3
"""
Database migration script for the email auto-responder application.
"""
import os
import argparse
import asyncpg
import logging
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('migrations')

async def get_db_connection():
    """Get a database connection from environment variables."""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    
    # Parse the connection URL
    return await asyncpg.connect(database_url)

async def run_migration(conn, migration_file: Path):
    """Run a single migration file."""
    logger.info(f"Running migration: {migration_file.name}")
    
    try:
        async with conn.transaction():
            with open(migration_file, 'r', encoding='utf-8') as f:
                sql = f.read()
                await conn.execute(sql)
            logger.info(f"Successfully applied {migration_file.name}")
    except Exception as e:
        logger.error(f"Error running migration {migration_file.name}: {str(e)}")
        raise

async def get_applied_migrations(conn) -> List[str]:
    """Get the list of already applied migrations."""
    try:
        # Try to create the migrations table if it doesn't exist
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Get the list of applied migrations
        rows = await conn.fetch("SELECT name FROM _migrations ORDER BY name")
        return [row['name'] for row in rows]
    except Exception as e:
        logger.error(f"Error getting applied migrations: {str(e)}")
        return []

async def mark_migration_applied(conn, migration_name: str):
    """Mark a migration as applied."""
    await conn.execute(
        "INSERT INTO _migrations (name) VALUES ($1) ON CONFLICT DO NOTHING",
        migration_name
    )

async def run_all_migrations():
    """Run all pending migrations."""
    migrations_dir = Path(__file__).parent / 'migrations'
    if not migrations_dir.exists():
        logger.error(f"Migrations directory not found: {migrations_dir}")
        return False
    
    # Get list of migration files
    migration_files = sorted([f for f in migrations_dir.glob('*.sql') if f.is_file()])
    
    if not migration_files:
        logger.warning("No migration files found")
        return True
    
    conn = None
    try:
        conn = await get_db_connection()
        applied_migrations = await get_applied_migrations(conn)
        
        for migration_file in migration_files:
            if migration_file.name not in applied_migrations:
                await run_migration(conn, migration_file)
                await mark_migration_applied(conn, migration_file.name)
                logger.info(f"Successfully marked {migration_file.name} as applied")
            else:
                logger.info(f"Skipping already applied migration: {migration_file.name}")
        
        return True
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        return False
    finally:
        if conn:
            await conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run database migrations')
    parser.add_argument('--migrate', action='store_true', help='Run all pending migrations')
    parser.add_argument('--check', action='store_true', help='Check if migrations are needed')
    
    args = parser.parse_args()
    
    if args.migrate:
        import asyncio
        success = asyncio.get_event_loop().run_until_complete(run_all_migrations())
        exit(0 if success else 1)
    elif args.check:
        # Just check if migrations are needed
        logger.info("Checking if migrations are needed...")
        # This is a placeholder - in a real implementation, you would check
        # if there are any pending migrations that need to be applied
        logger.info("No pending migrations found")
        exit(0)
    else:
        parser.print_help()
        exit(1)
