import asyncpg
import asyncio
import os
import ssl
import socket
from urllib.parse import urlparse, urlunparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Build connection string from environment variables
def get_database_url():
    """Construct database URL from environment variables."""
    return "postgresql://{user}:{password}@{host}:{port}/{database}".format(
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "railway")
    )

# Get database URL
DATABASE_URL = get_database_url()

async def test_connection():
    print(f"\n🔍 Testing database connection at {datetime.now()}")
    print("=" * 50)
    
    # Parse the connection URL
    db_url = urlparse(DATABASE_URL)
    
    # Extract connection parameters
    db_params = {
        'host': db_url.hostname,
        'port': db_url.port,
        'user': db_url.username,
        'password': db_url.password,
        'database': db_url.path[1:],  # Remove leading slash
        'ssl': 'require',  # Force SSL
        'timeout': 10,  # 10 second timeout
    }
    
    print("\n🔗 Connection Details:")
    print(f"  - Host: {db_params['host']}")
    print(f"  - Port: {db_params['port']}")
    print(f"  - Database: {db_params['database']}")
    print(f"  - User: {db_params['user']}")
    
    # Test 1: Check if host is reachable
    print("\n🔍 Testing host reachability...")
    try:
        # Try to resolve the hostname
        ip_address = socket.gethostbyname(db_params['host'])
        print(f"  ✅ Host resolved to IP: {ip_address}")
        
        # Try to connect to the port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((ip_address, db_params['port']))
        sock.close()
        
        if result == 0:
            print(f"  ✅ Port {db_params['port']} is open")
        else:
            print(f"  ❌ Port {db_params['port']} is closed or blocked")
            print("  ℹ️  Please check if the database is running and accessible")
            return False
            
    except socket.gaierror:
        print("  ❌ Could not resolve hostname")
        return False
    except Exception as e:
        print(f"  ❌ Network error: {e}")
        return False
    
    # Test 2: Try to establish database connection
    print("\n🔌 Testing database connection...")
    try:
        print("  🔄 Attempting to connect...")
        conn = await asyncpg.connect(**db_params)
        print("  ✅ Successfully connected to the database")
        
        # Test 3: Get database version
        version = await conn.fetchval('SELECT version()')
        print(f"  ℹ️  Database version: {version}")
        
        # Test 4: Run a simple query
        result = await conn.fetchval('SELECT 1 as test_value')
        print(f"  ✅ Test query result: {result}")
        
        # Test 5: Check if we can access the database
        try:
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                LIMIT 5
            """)
            print(f"  ℹ️  Found {len(tables)} tables in the database")
            if tables:
                print(f"  ℹ️  Example tables: {', '.join([t['table_name'] for t in tables])}")
        except Exception as e:
            print(f"  ⚠️  Could not list tables (permission issue?): {e}")
        
        await conn.close()
        print("\n🎉 All tests completed successfully!")
        return True
        
    except asyncpg.InvalidPasswordError:
        print("  ❌ Authentication failed: Invalid username or password")
    except asyncpg.InvalidCatalogNameError:
        print(f"  ❌ Database '{db_params['database']}' does not exist")
    except asyncpg.InvalidAuthorizationSpecificationError:
        print("  ❌ Authentication failed: Check username and password")
    except asyncpg.TooManyConnectionsError:
        print("  ❌ Too many connections to the database")
    except asyncpg.ConnectionDoesNotExistError:
        print("  ❌ Connection to the database was lost")
    except asyncpg.PostgresConnectionError as e:
        print(f"  ❌ Database connection error: {e}")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
    
    return False

if __name__ == "__main__":
    print("🚀 Starting database connection test...")
    success = asyncio.run(test_connection())
    if not success:
        print("\n🔴 Database connection test failed. Please check the error messages above.")
        print("\n🔧 Troubleshooting tips:")
        print("1. Verify the database is running and accessible")
        print("2. Check if the hostname and port are correct")
        print("3. Verify the username and password")
        print("4. Check if the database name is correct")
        print("5. Ensure your IP is whitelisted in the database firewall")
        print("6. Check if the database server is configured to accept remote connections")
        exit(1)
    else:
        print("\n✅ Database connection test completed successfully!")
        exit(0)
