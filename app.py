"""
Main application entry point.
This file exists to support platforms that look for app.py by default.
"""

import os
from email_sender import app

# This makes the app importable as a module
application = app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("email_sender:app", host="0.0.0.0", port=port, reload=True)
