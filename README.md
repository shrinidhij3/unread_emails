# Email Sync Service

A robust and flexible email synchronization service that supports multiple email providers including Gmail, Office 365, cPanel, Zoho, and custom IMAP/SMTP servers.

## Key Features

- 🚀 FastAPI-based email sending service
- 🔒 Secure SMTP authentication with encrypted credentials
- 📧 Support for multiple email providers (Gmail, Office 365, etc.)
- 🔄 Background email processing
- 🏥 Health check endpoint at `/health`
- 📊 Built-in logging and monitoring
- 🐳 Docker and Render.com ready

- 🔒 Secure password storage with encryption
- 🔄 Automatic provider detection
- 📧 Support for multiple email providers (Gmail, Office 365, cPanel, Zoho, custom IMAP/SMTP)
- 🔄 Background email synchronization
- 🚀 RESTful API for account management and synchronization
- 📊 Monitoring and error tracking

## Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Redis (optional, for background tasks)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd email-sync-service
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
   # For development with hot-reload
   pip install uvicorn
   ```

4. Set up environment variables:
   Create a `.env` file in the project root with the following variables:
   ```
   # Database (preferred method - single URL)
   DATABASE_URL=postgresql://username:password@localhost:5432/email_service
   
   # OR use individual variables (for backward compatibility)
   # DB_HOST=localhost
   # DB_PORT=5432
   # DB_NAME=email_service
   # DB_USER=postgres
   # DB_PASSWORD=your_secure_password
   
   # Encryption
   ENCRYPTION_KEY=your_secure_encryption_key
   
   # Logging
   LOG_LEVEL=INFO
   LOG_FILE=email_service.log
   ```

5. Initialize the database:
   ```bash
   # Create database tables
   # Initialize the database (if applicable)
   # python -c "from email_service.db import init_db; import asyncio; asyncio.run(init_db())"
   ```

## Running the Service

1. Start the API server:
   ```bash
   uvicorn email_sender:app --reload --host 0.0.0.0 --port 8000
   ```

2. The API documentation will be available at:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## Deployment

### Deploying to Render.com

1. **Create a new Web Service**
   - Go to your Render dashboard and click "New +" then select "Web Service"
   - Connect your GitHub/GitLab repository or deploy manually

2. **Configure Build & Deploy**
   - **Name**: Choose a name for your service (e.g., `email-sync-service`)
   - **Region**: Select the region closest to your users
   - **Branch**: Select your main branch
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn email_sender:app --host 0.0.0.0 --port $PORT`

3. **Environment Variables**
   Add the following environment variables in the Render dashboard:
   ```
   # Required
   PORT=10000
   DATABASE_URL=postgresql://user:password@host:5432/dbname
   ENCRYPTION_KEY=your_secure_encryption_key
   
   # SMTP Settings (example for Gmail)
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   
   # IMAP Settings (example for Gmail)
   IMAP_SERVER=imap.gmail.com
   IMAP_PORT=993
   
   # Optional
   LOG_LEVEL=INFO
   EMAIL_RETENTION_DAYS=30
   PROCESSING_BATCH_SIZE=50
   
   # For Gmail, you might need an App Password
   # See: https://support.google.com/accounts/answer/185833
   ```

4. **Database**
   - Create a new PostgreSQL database on Render
   - Copy the external database URL and set it as `DATABASE_URL`

5. **Deploy**
   - Click "Create Web Service" to deploy your application
   - Monitor the logs for any deployment issues

6. **Verify Deployment**
   - Once deployed, visit `https://your-service-name.onrender.com/health`
   - You should see a `{"status":"healthy"}` response

## API Endpoints

### Email Accounts

- `POST /api/accounts` - Create a new email account
- `GET /api/accounts` - List all email accounts
- `GET /api/accounts/{account_id}` - Get account details
- `DELETE /api/accounts/{account_id}` - Delete an account
- `POST /api/accounts/{account_id}/sync` - Start sync for an account
- `POST /api/sync/all` - Sync all active accounts

### Providers

- `GET /api/providers` - List all available email providers

### Health

- `GET /health` - Health check endpoint

## Usage Examples

### Create a new email account

```bash
curl -X 'POST' \
  'http://localhost:8000/api/accounts' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "email": "user@example.com",
    "password": "your_password",
    "provider_type": "auto"
  }'
```

### Start sync for an account

```bash
curl -X 'POST' \
  'http://localhost:8000/api/accounts/1/sync' \
  -H 'accept: application/json' \
  -d ''
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_HOST` | Database host | `localhost` |
| `DB_PORT` | Database port | `5432` |
| `DB_NAME` | Database name | `email_service` |
| `DB_USER` | Database user | `postgres` |
| `DB_PASSWORD` | Database password | - |
| `ENCRYPTION_KEY` | Encryption key for passwords | - |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_FILE` | Log file path | `email_service.log` |

### Provider Configuration

The service includes pre-configured settings for popular email providers. You can add or modify providers in the `provider_registry` in `email_service.py`.

## Security

- Passwords are encrypted at rest using Fernet symmetric encryption
- All database connections use SSL/TLS
- Input validation is performed on all API endpoints
- Rate limiting can be implemented at the API gateway level

## Monitoring

- Logs are written to both console and file
- Error tracking and monitoring can be integrated using services like Sentry
- Database connection pooling is used for better performance

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
