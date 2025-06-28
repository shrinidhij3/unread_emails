import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

def get_fernet_key(secret_key):
    """Generate Fernet key using the same method as Django's default encryption."""
    salt = b'django_core_encryption'
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(secret_key.encode()))

def test_decrypt(encrypted_password, secret_key):
    try:
        print(f"\n🔐 Testing decryption with secret key: {secret_key[:10]}...")
        key = get_fernet_key(secret_key)
        fernet = Fernet(key)
        
        # Try to decrypt
        decrypted = fernet.decrypt(encrypted_password.encode())
        print(f"✅ Success! Decrypted password: {decrypted.decode()}")
        return True
    except Exception as e:
        print(f"❌ Failed to decrypt: {e}")
        return False

if __name__ == "__main__":
    encrypted_password = "gAAAAABoXnsIYdE3OOL8K7amfBHMgX-4go2oK0prNEtBVCQqK8gudbbKm_PUd7iIbDLouQ96k0wElY08jWkF-SZVct7q1bv3-UYOxzNnvEiDwUjPeAT-5Zc="
    
    # Try with DJANGO_SECRET_KEY from environment
    secret_key = os.getenv('DJANGO_SECRET_KEY')
    if secret_key:
        print(f"Found DJANGO_SECRET_KEY in environment (length: {len(secret_key)})")
        test_decrypt(encrypted_password, secret_key)
    else:
        print("❌ DJANGO_SECRET_KEY not found in environment variables")
    
    # Let the user try with a different key
    print("\nIf decryption failed, you can try with a different key:")
    custom_key = input("Enter a custom key (or press Enter to skip): ")
    if custom_key:
        test_decrypt(encrypted_password, custom_key)
