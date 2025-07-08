import os
import sys
import asyncio
import logging
from email_sender import send_email, EmailRequest, app
from smtp_sender import DEFAULT_MAILING_ADDRESS

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

async def test_send_email():
    # Set your test email details here
    test_data = EmailRequest(
        sender=os.getenv('TEST_SENDER_EMAIL', 'your-email@gmail.com'),
        password=os.getenv('TEST_ENCRYPTED_PASSWORD', 'YOUR_ENCRYPTED_PASSWORD'),
        recipient=os.getenv('TEST_RECIPIENT_EMAIL', 'recipient-email@example.com'),
        subject="Test Email from Local Test",
        message="This is a test email from the local test script.",
        html="<h1>Test Email</h1><p>This is a <strong>test email</strong> from the local test script.</p>",
        smtp_host=os.getenv('TEST_SMTP_HOST', 'smtp.gmail.com'),
        smtp_port=int(os.getenv('TEST_SMTP_PORT', '587')),
        mailing_address=DEFAULT_MAILING_ADDRESS
    )
    
    logger.info(f"Testing with sender: {test_data.sender}")
    logger.info(f"SMTP Server: {test_data.smtp_host}:{test_data.smtp_port}")
    logger.debug(f"Encrypted password: {test_data.password}")
    
    try:
        # Set the ENCRYPTION_KEY environment variable if not already set
        if 'ENCRYPTION_KEY' not in os.environ:
            encryption_key = os.getenv('TEST_ENCRYPTION_KEY', 'your-encryption-key-here')
            if encryption_key == 'your-encryption-key-here':
                logger.error("Please set the TEST_ENCRYPTION_KEY environment variable")
                return False
            os.environ['ENCRYPTION_KEY'] = encryption_key
            
        logger.info("Attempting to send test email...")
        
        # Create a test client and call the endpoint directly
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Convert the EmailRequest to a dictionary
        email_data = test_data.dict()
        
        # Call the API endpoint
        response = client.post("/send-email", json=email_data)
        
        if response.status_code == 200:
            logger.info("Email sent successfully via API!")
            logger.debug(f"Response: {response.json()}")
            return True
        else:
            logger.error(f"Failed to send email. Status: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    asyncio.run(test_send_email())
