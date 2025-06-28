"""
Package initialization file for the email sender service.

This makes the directory a Python package and exposes the FastAPI app.
"""

# Import the FastAPI app
from email_sender import app

# This makes the app importable as a module
__all__ = ['app']
