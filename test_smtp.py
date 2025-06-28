import smtplib
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_smtp_login(email, password, smtp_host, smtp_port):
    """Test SMTP login with the given credentials"""
    try:
        logger.info(f"Connecting to {smtp_host}:{smtp_port}...")
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            logger.info("Connected to SMTP server")
            
            # Enable debug output
            server.set_debuglevel(1)
            
            # Start TLS if available
            server.ehlo()
            if server.has_extn('STARTTLS'):
                logger.info("Starting TLS...")
                server.starttls()
                server.ehlo()
            
            logger.info(f"Attempting to login as {email}")
            logger.info(f"Password length: {len(password)}")
            
            # Try to login
            server.login(email, password)
            logger.info("Successfully logged in!")
            return True
            
    except Exception as e:
        logger.error(f"SMTP Error: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    import getpass
    
    print("SMTP Login Tester")
    print("=================")
    
    email = input("Email: ")
    password = getpass.getpass("Password: ")
    smtp_host = input("SMTP Host [smtp.gmail.com]: ") or "smtp.gmail.com"
    smtp_port = input("SMTP Port [587]: ") or "587"
    
    test_smtp_login(email, password, smtp_host, int(smtp_port))
