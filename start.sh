#!/bin/bash
set -e

# Wait for database to be available (if needed)
# while ! nc -z $DB_HOST $DB_PORT; do
#   echo "Waiting for PostgreSQL..."
#   sleep 1
# done

echo "✅ PostgreSQL is ready"

# Run database migrations (if any)
# python manage.py migrate

# Start the application
exec "$@"
