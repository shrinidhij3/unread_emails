-- Create credentials_email table (main accounts table)
CREATE TABLE IF NOT EXISTS credentials_email (
    account_id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_encrypted BYTEA NOT NULL,
    password_salt BYTEA NOT NULL,
    provider_type VARCHAR(50) NOT NULL,
    imap_host VARCHAR(255) NOT NULL,
    imap_port INTEGER NOT NULL,
    imap_use_ssl BOOLEAN DEFAULT TRUE,
    smtp_host VARCHAR(255),
    smtp_port INTEGER,
    smtp_use_ssl BOOLEAN DEFAULT TRUE,
    smtp_use_tls BOOLEAN DEFAULT TRUE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_synced_at TIMESTAMP WITH TIME ZONE,
    sync_status VARCHAR(50) DEFAULT 'idle',
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    custom_data JSONB
);

-- Create processed_messages table (matches the name used in the code)
CREATE TABLE IF NOT EXISTS processed_messages (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR(255) NOT NULL,
    email_address VARCHAR(255) NOT NULL,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    folder VARCHAR(100),
    status VARCHAR(50) DEFAULT 'processed',
    error_message TEXT,
    UNIQUE(message_id, email_address)
);

-- Create error_logs table
CREATE TABLE IF NOT EXISTS error_logs (
    id SERIAL PRIMARY KEY,
    email_address VARCHAR(255),
    error_type VARCHAR(100) NOT NULL,
    error_message TEXT NOT NULL,
    message_id VARCHAR(255),
    folder VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create metrics table
CREATE TABLE IF NOT EXISTS metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    email_address VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_processed_messages_email ON processed_messages(email_address);
CREATE INDEX IF NOT EXISTS idx_processed_messages_processed_at ON processed_messages(processed_at);
CREATE INDEX IF NOT EXISTS idx_error_logs_email ON error_logs(email_address);
CREATE INDEX IF NOT EXISTS idx_error_logs_created_at ON error_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_metrics_metric_name ON metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_created_at ON metrics(created_at);
