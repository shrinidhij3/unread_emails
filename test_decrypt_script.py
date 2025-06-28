import os
import sys
import base64
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Load environment variables
load_dotenv()

def get_fernet_key():
    """Generate Fernet key using the same method as Django's default encryption."""
    secret_key = os.getenv('DJANGO_SECRET_KEY')
    if not secret_key:
        raise ValueError("❌ ERROR: DJANGO_SECRET_KEY environment variable not found in .env file")
    
    print(f"✅ Found DJANGO_SECRET_KEY (length: {len(secret_key)})")
    
    # Validate the secret key format
    if len(secret_key) < 32:
        print("⚠️  WARNING: DJANGO_SECRET_KEY is shorter than recommended (should be at least 50 chars)")
    
    try:
        # Use the first 32 bytes of the hashed secret key
        salt = b'django_core_encryption'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        derived_key = kdf.derive(secret_key.encode())
        key = base64.urlsafe_b64encode(derived_key)
        print(f"✅ Successfully generated Fernet key")
        return key
    except Exception as e:
        raise ValueError(f"❌ Failed to generate Fernet key: {str(e)}")

def decrypt_fernet(encrypted_data: str) -> str:
    """Decrypt a password that was encrypted using the same key generation method."""
    try:
        print(f"\n🔐 Decrypting password (length: {len(encrypted_data)})")
        
        if not encrypted_data:
            raise ValueError("Empty encrypted data provided")
            
        # Strip 'ENC:' prefix if present
        if encrypted_data.startswith('ENC:'):
            encrypted_data = encrypted_data[4:]
            print(f"ℹ️  Removed 'ENC:' prefix")
        
        # Get the Fernet key
        print("🔑 Generating encryption key...")
        key = get_fernet_key()
        fernet = Fernet(key)
        
        print("🔓 Attempting to decrypt...")
        # Decrypt the data
        decrypted = fernet.decrypt(encrypted_data.encode())
        decrypted_str = decrypted.decode('utf-8')
        
        print(f"✅ Successfully decrypted password (length: {len(decrypted_str)})")
        return decrypted_str
        
    except Exception as e:
        error_msg = str(e)
        if 'Invalid token' in error_msg:
            error_msg = "Invalid token - This usually means the encryption key doesn't match"
        elif 'Invalid base64' in error_msg:
            error_msg = "Invalid base64 data - Check the encrypted password format"
        raise ValueError(f"❌ Decryption failed: {error_msg}")

def test_decrypt(encrypted_password):
    print("\n" + "="*60)
    print("🔍 Password Decryption Test")
    print("="*60)
    
    try:
        # First, verify we can access the environment variable
        secret_key = os.getenv('DJANGO_SECRET_KEY')
        if not secret_key:
            raise ValueError("❌ DJANGO_SECRET_KEY not found in environment variables")
            
        print(f"🔑 Using DJANGO_SECRET_KEY (first 10 chars): {secret_key[:10]}...")
        print(f"🔑 Encrypted password (first 10 chars): {encrypted_password[:10]}...")
        
        # Try to decrypt
        decrypted = decrypt_fernet(encrypted_password)
        
        # Show partial decrypted result (don't log full password)
        decrypted_display = f"{decrypted[:2]}...{decrypted[-2:]}" if len(decrypted) > 4 else "[too short]"
        print(f"\n✅ DECRYPTION SUCCESSFUL!")
        print(f"   Original length: {len(encrypted_password)}")
        print(f"   Decrypted length: {len(decrypted)}")
        print(f"   First/last chars: {decrypted_display}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DECRYPTION FAILED!")
        print(f"   Error: {str(e)}")
        print("\n🔧 Troubleshooting Tips:")
        print("1. Verify DJANGO_SECRET_KEY in .env matches the one used for encryption")
        print("2. Ensure the encrypted password is correct and complete")
        print("3. Check for any special characters that might need escaping")
        print("4. The encryption key might be different from what's expected")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test with the provided encrypted password
        test_decrypt(sys.argv[1])
    else:
        print("Usage: python test_decrypt_script.py <encrypted_password>")
        print("Make sure your .env file contains the correct DJANGO_SECRET_KEY")
