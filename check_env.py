import os
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if DJANGO_SECRET_KEY exists
    secret_key = os.getenv('DJANGO_SECRET_KEY')
    
    if secret_key:
        print("✅ DJANGO_SECRET_KEY is set in .env")
        print(f"Key length: {len(secret_key)} characters")
        print(f"Key starts with: {secret_key[:10]}...")
        
        # Check if the key looks like a Django secret key
        if len(secret_key) < 32:
            print("⚠️  Warning: Key is shorter than recommended (should be at least 50 characters)")
        elif len(secret_key) > 100:
            print("⚠️  Warning: Key is unusually long (typically 50-80 characters)")
            
        # Check for common issues
        if ' ' in secret_key:
            print("⚠️  Warning: Key contains spaces - make sure it's properly quoted in .env")
            
        if secret_key.startswith(('"', "'")) and secret_key.endswith(('"', "'")):
            print("⚠️  Warning: Key appears to be wrapped in quotes - this might cause issues")
    else:
        print("❌ DJANGO_SECRET_KEY is not set in .env")
        print("\nPlease make sure your .env file contains a line like:")
        print('DJANGO_SECRET_KEY="your-secret-key-here"')

if __name__ == "__main__":
    main()
