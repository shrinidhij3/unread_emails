"""
This is a compatibility module that allows the application to be run with 'email_api:app'.
It simply imports and re-exports the FastAPI app from email_sender.
"""

# Import the FastAPI app from email_sender
from email_sender import app

# This makes the app importable as email_api:app
# No additional code needed as we're just re-exporting the app
