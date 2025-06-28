import multiprocessing
import os

# Server socket
bind = "0.0.0.0:" + os.environ.get("PORT", "10000")
workers = 2  # Reduced for Render's free tier
worker_class = 'uvicorn.workers.UvicornWorker'
worker_connections = 1000

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = os.environ.get("LOG_LEVEL", "info")

# Timeout
timeout = 120  # 2 minutes
keepalive = 5

# Worker processes
max_requests = 1000
max_requests_jitter = 50

# Security
forwarded_allow_ips = "*"

# Debugging
reload = os.environ.get("ENVIRONMENT", "production").lower() == "development"

# Error handling
preload_app = True  # Load application before forking workers

# Worker class specific settings
worker_tmp_dir = "/dev/shm"  # Use shared memory for worker tmp files
