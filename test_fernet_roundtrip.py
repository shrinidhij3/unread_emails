import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dotenv import load_dotenv

def get_fernet_key(secret_key):
    """Generate Fernet key using Django's key derivation"""
    salt = b'django_core_encryption'
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(secret_key.encode()))

def test_encrypt_decrypt(secret_key, password):
    """Test encryption and decryption with the given key"""
    try:
        # Generate Fernet key
        key = get_fernet_key(secret_key)
        fernet = Fernet(key)
        
        # Encrypt
        encrypted = fernet.encrypt(password.encode())
        print(f"✅ Encryption successful")
        print(f"   Original: {password}")
        print(f"   Encrypted: {encrypted.decode()}")
        
        # Decrypt
        decrypted = fernet.decrypt(encrypted).decode()
        print(f"✅ Decryption successful")
        print(f"   Decrypted: {decrypted}")
        
        # Verify
        if decrypted == password:
            print("✅ Verification: Original and decrypted match!")
        else:
            print("❌ Verification failed: Mismatch between original and decrypted")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Load environment
    load_dotenv()
    
    # Get DJANGO_SECRET_KEY
    secret_key = os.getenv('DJANGO_SECRET_KEY')
    if not secret_key:
        print("❌ DJANGO_SECRET_KEY not found in environment")
        exit(1)
    
    print(f"Using DJANGO_SECRET_KEY (first 10 chars): {secret_key[:10]}...")
    
    # Test with a known password
    test_password = "test-password-123"
    print(f"\n🔒 Testing with password: {test_password}")
    test_encrypt_decrypt(secret_key, test_password)
    
    # Try with the actual encrypted password
    encrypted_password = "gAAAAABoXnsIYdE3OOL8K7amfBHMgX-4go2oK0prNEtBVCQqK8gudbbKm_PUd7iIbDLouQ96k0wElY08jWkF-SZVct7q1bv3-UYOxzNnvEiDwUjPeAT-5Zc="
    print(f"\n🔓 Trying to decrypt the actual password...")
    
    try:
        key = get_fernet_key(secret_key)
        fernet = Fernet(key)
        decrypted = fernet.decrypt(encrypted_password.encode()).decode()
        print(f"✅ Successfully decrypted password")
        print(f"   Decrypted: {decrypted}")
    except Exception as e:
        print(f"❌ Failed to decrypt: {e}")
        print("\nThis suggests the DJANGO_SECRET_KEY doesn't match the one used for encryption.")
        print("Possible solutions:")
        print("1. Verify the correct DJANGO_SECRET_KEY is in your .env file")
        print("2. If this is a production environment, check your deployment settings")
        print("3. You may need to re-encrypt the password with the current key")
