import multiprocessing
import os

# Server socket
bind = "0.0.0.0:" + os.environ.get("PORT", "10000")
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'uvicorn.workers.UvicornWorker'

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = os.environ.get("LOG_LEVEL", "info")

# Timeout
# Set timeout to 120 seconds (2 minutes) to handle long-running requests
timeout = 120

# Keep-alive
keepalive = 5

# Worker processes
max_requests = 1000
max_requests_jitter = 50

# Security
# Prevents the server from being attacked via an unsafe header
forwarded_allow_ips = "*"

# Debugging
reload = os.environ.get("ENVIRONMENT", "development") == "development"
