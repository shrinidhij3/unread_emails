import os
import base64
from dotenv import load_dotenv
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

def print_header(text):
    print("\n" + "="*60)
    print(f"🔍 {text}")
    print("="*60)

def test_environment():
    print_header("Environment Check")
    # Load .env file manually
    from pathlib import Path
    env_path = Path('.') / '.env'
    print(f"Looking for .env at: {env_path.absolute()}")
    print(f".env exists: {env_path.exists()}")
    
    # Try to load .env
    load_dotenv(dotenv_path=env_path, override=True)
    
    # Check DJANGO_SECRET_KEY
    secret_key = os.getenv('DJANGO_SECRET_KEY')
    if secret_key:
        print(f"✅ Found DJANGO_SECRET_KEY (length: {len(secret_key)})")
        print(f"   First 10 chars: {secret_key[:10]}...")
        print(f"   Last 10 chars: ...{secret_key[-10:] if len(secret_key) > 10 else ''}")
        return secret_key
    else:
        print("❌ DJANGO_SECRET_KEY not found in environment")
        return None

def test_decryption(secret_key, encrypted_password):
    print_header("Decryption Test")
    try:
        # Generate the Fernet key
        salt = b'django_core_encryption'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(secret_key.encode()))
        print(f"✅ Generated Fernet key")
        
        # Try to decrypt
        fernet = Fernet(key)
        print(f"🔐 Attempting to decrypt password...")
        decrypted = fernet.decrypt(encrypted_password.encode())
        
        print("\n🎉 DECRYPTION SUCCESSFUL!")
        print(f"Decrypted password: {decrypted.decode()}")
        return True
        
    except InvalidToken as e:
        print("❌ Invalid token - The key doesn't match or token is corrupted")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"❌ Decryption failed: {e}")
    
    return False

if __name__ == "__main__":
    # The encrypted password we're trying to decrypt
    ENCRYPTED_PASSWORD = "gAAAAABoXnsIYdE3OOL8K7amfBHMgX-4go2oK0prNEtBVCQqK8gudbbKm_PUd7iIbDLouQ96k0wElY08jWkF-SZVct7q1bv3-UYOxzNnvEiDwUjPeAT-5Zc="
    
    print_header("Fernet Password Decryption Debugger")
    
    # Test 1: Check environment
    secret_key = test_environment()
    
    if secret_key:
        # Test 2: Try decryption
        if not test_decryption(secret_key, ENCRYPTED_PASSWORD):
            print("\n🔧 Troubleshooting:")
            print("1. The DJANGO_SECRET_KEY in your .env file doesn't match the one used for encryption")
            print("2. The encrypted password might be corrupted or in a different format")
            print("3. The encryption method might be different")
            
            # Try with a test password to verify the encryption/decryption works
            try:
                print("\n🔧 Testing with a known password...")
                test_key = Fernet.generate_key()
                test_fernet = Fernet(test_key)
                test_password = "test-password-123"
                test_encrypted = test_fernet.encrypt(test_password.encode()).decode()
                test_decrypted = test_fernet.decrypt(test_encrypted.encode()).decode()
                
                print(f"✅ Test encryption/decryption successful!")
                print(f"   Original: {test_password}")
                print(f"   Encrypted: {test_encrypted}")
                print(f"   Decrypted: {test_decrypted}")
            except Exception as e:
                print(f"❌ Test encryption failed: {e}")
