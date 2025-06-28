"""
WSGI config for the email sender service.

It exposes the WSGI callable as a module-level variable named ``application``.
"""

import os
from email_sender import app

# This allows gunicorn to find the app
application = app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("email_sender:app", host="0.0.0.0", port=port, reload=True)
