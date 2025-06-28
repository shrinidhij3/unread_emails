import os
import base64
from cryptography.fernet import Fernet

def get_fernet():
    # Get encryption key from environment variable
    encryption_key = os.getenv('ENCRYPTION_KEY')
    if not encryption_key:
        raise ValueError("ENCRYPTION_KEY environment variable not set")
    
    # Ensure the key is 32 URL-safe base64-encoded bytes
    key = base64.urlsafe_b64encode(encryption_key.encode().ljust(32)[:32])
    return Fernet(key)

def test_decrypt(encrypted_password):
    try:
        fernet = get_fernet()
        print(f"Encrypted password: {encrypted_password}")
        
        # Handle both 'ENC:' prefixed and raw encrypted passwords
        if encrypted_password.startswith('ENC:'):
            encrypted_password = encrypted_password[4:]
            
        # Decode the base64 string and then decrypt
        decrypted = fernet.decrypt(encrypted_password.encode()).decode()
        print(f"Successfully decrypted password")
        print(f"Decrypted length: {len(decrypted)}")
        print(f"First 2 chars: {decrypted[:2]}...")
        return True
    except Exception as e:
        print(f"Decryption failed: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_decrypt.py <encrypted_password>")
        print("Make sure to set ENCRYPTION_KEY environment variable")
        sys.exit(1)
        
    encrypted_password = sys.argv[1]
    test_decrypt(encrypted_password)
