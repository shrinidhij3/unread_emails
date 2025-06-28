"""
Main application entry point.
This file exists to support platforms that look for app.py by default.
It's recommended to use wsgi.py instead.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

# Import the FastAPI app from email_sender
from email_sender import app

# This makes the app importable as a module
application = app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("email_sender:app", host="0.0.0.0", port=port, reload=True)
