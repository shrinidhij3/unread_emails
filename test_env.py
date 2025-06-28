from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Print all environment variables for debugging
print("All environment variables:")
for key, value in os.environ.items():
    if 'KEY' in key or 'PASS' in key or 'SECRET' in key:
        print(f"{key} = {'*' * 8} (hidden for security)")
    else:
        print(f"{key} = {value}")

# Check if DJANGO_SECRET_KEY is set
django_key = os.getenv('DJANGO_SECRET_KEY')
print(f"\nDJANGO_SECRET_KEY is {'set' if django_key else 'not set'}")
if django_key:
    print(f"DJANGO_SECRET_KEY length: {len(django_key)}")
    print(f"Starts with: {django_key[:4]}..." if len(django_key) > 4 else "Key is too short")
