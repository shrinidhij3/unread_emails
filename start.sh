#!/bin/bash
set -e

# Wait for database to be available (if needed)
# while ! nc -z $DB_HOST $DB_PORT; do
#   echo "Waiting for PostgreSQL..."
#   sleep 1
# done

echo "✅ Starting application..."
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "Files in directory: $(ls -la)"

exec gunicorn --worker-class=uvicorn.workers.UvicornWorker \
    --workers=2 \
    --bind=0.0.0.0:${PORT:-10000} \
    --timeout=120 \
    --access-logfile - \
    --error-logfile - \
    --log-level=debug \
    email_sender:app

# Run database migrations (if any)
# python manage.py migrate

# Start the application
exec "$@"
