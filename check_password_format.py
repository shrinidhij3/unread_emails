import os
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def analyze_password(password):
    print("\n🔍 Password Analysis")
    print("=" * 50)
    
    # Basic info
    print(f"Password length: {len(password)}")
    print(f"First 10 chars: {password[:10]}")
    print(f"Last 10 chars: {password[-10:] if len(password) > 10 else password}")
    
    # Check common patterns
    print("\n🔎 Pattern Analysis:")
    
    # Check if it's a Fernet token (starts with gAAAAA and contains only base64url chars)
    is_fernet = bool(re.match(r'^gAAAAA[\w-]+={0,3}$', password))
    print(f"- Looks like a Fernet token: {'✅' if is_fernet else '❌'}")
    
    # Check if it's base64 encoded
    base64_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='
    is_base64 = all(c in base64_chars for c in password)
    print(f"- Contains only base64 chars: {'✅' if is_base64 else '❌'}")
    
    # Check if it's already in plaintext (contains letters, numbers, and possibly special chars)
    is_plaintext = any(c.isalpha() for c in password) and any(c.isdigit() for c in password)
    print(f"- Looks like plaintext: {'✅' if is_plaintext else '❌'}")
    
    # Check if it contains special characters
    has_special = bool(re.search(r'[^A-Za-z0-9+/=]', password))
    print(f"- Contains special chars: {'✅' if has_special else '❌'}")
    
    # Check if it's a URL-safe base64 string
    is_url_safe = bool(re.match(r'^[A-Za-z0-9-_]+$', password))
    print(f"- URL-safe base64: {'✅' if is_url_safe else '❌'}")
    
    # Check if it's a standard base64 string
    is_std_base64 = bool(re.match(r'^[A-Za-z0-9+/]+={0,2}$', password))
    print(f"- Standard base64: {'✅' if is_std_base64 else '❌'}")
    
    # Check for common encryption indicators
    print("\n🔒 Encryption Indicators:")
    print(f"- Starts with 'gAAAAA': {'✅' if password.startswith('gAAAAA') else '❌'}")
    print(f"- Starts with 'ENC:': {'✅' if password.startswith('ENC:') else '❌'}")
    print(f"- Contains '==': {'✅' if '==' in password else '❌'}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        password = sys.argv[1]
        analyze_password(password)
        
        # If it looks like a Fernet token but decryption is failing
        if password.startswith('gAAAAA'):
            print("\n🔑 Fernet Token Analysis:")
            print("This appears to be a Fernet token. Common decryption issues:")
            print("1. The encryption key (DJANGO_SECRET_KEY) doesn't match the one used for encryption")
            print("2. The token might be corrupted or incomplete")
            print("3. The token might be using a different encryption method")
    else:
        print("Usage: python check_password_format.py <password>")
        print("This will analyze the format of the provided password/encrypted string.")
