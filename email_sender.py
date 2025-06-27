from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional
import smtplib
import email.utils
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models
class MailingAddress(BaseModel):
    name: str = "Thorsignia"
    address_line1: str = "123 Business Street"
    address_line2: Optional[str] = None
    city: str = "Bangalore"
    state: str = "Karnataka"
    postal_code: str = "560001"
    country: str = "India"

class EmailRequest(BaseModel):
    sender: EmailStr
    password: str
    recipient: EmailStr
    subject: str
    message: str = ""
    html: Optional[str] = None
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    reply_to: Optional[EmailStr] = None
    mailing_address: Optional[MailingAddress] = None

def format_mailing_address(address: Optional[MailingAddress] = None) -> str:
    """Format a mailing address from a MailingAddress model."""
    if not address:
        address = MailingAddress()
    
    lines = [address.name]
    if address.address_line1:
        lines.append(address.address_line1)
    if address.address_line2:
        lines.append(address.address_line2)
    
    city_line = []
    if address.city:
        city_line.append(address.city)
    if address.state:
        city_line.append(address.state)
    if address.postal_code:
        city_line.append(address.postal_code)
    
    lines.append(", ".join(city_line))
    if address.country:
        lines.append(address.country)
    
    return "\n".join(lines)

@app.post("/send-email")
async def send_email(email_data: EmailRequest):
    """
    Send an email using SMTP
    
    Required fields:
    - sender: Sender's email address
    - password: SMTP password/app password
    - recipient: Recipient's email address
    - subject: Email subject
    - message: Plain text message (or html)
    """
    try:
        # Use reply_to if provided, otherwise use sender
        reply_to = email_data.reply_to or email_data.sender
        
        # Create email message
        msg = MIMEMultipart('alternative')
        
        # Set basic headers
        msg['Subject'] = email_data.subject
        msg['From'] = f'Thorsignia Support <{email_data.sender}>'
        msg['To'] = email_data.recipient
        msg['Reply-To'] = reply_to
        msg['Date'] = email.utils.formatdate(localtime=True)
        msg['Message-ID'] = email.utils.make_msgid(domain=email_data.sender.split('@')[-1])
        
        # Add unsubscribe header
        msg['List-Unsubscribe'] = f'<mailto:{reply_to}?subject=Unsubscribe>'
        
        # Handle plain text content
        message_body = email_data.message
        if not message_body and email_data.html:
            # Create plain text from HTML if no plain text provided
            message_body = re.sub(r'<[^>]+>', ' ', email_data.html)
            message_body = re.sub(r'\s+', ' ', message_body).strip()
        
        if not message_body:
            message_body = "Please enable HTML to view this email."
        
        # Add plain text part
        text_part = MIMEText(message_body, 'plain', 'utf-8')
        msg.attach(text_part)
        
        # Add HTML part if provided
        if email_data.html:
            formatted_address = format_mailing_address(email_data.mailing_address)
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
                <title>{email_data.subject}</title>
            </head>
            <body style="margin: 0; padding: 20px; font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto;">
                    {email_data.html}
                    <p style="margin-top: 30px; font-size: 12px; color: #777; border-top: 1px solid #eee; padding-top: 10px;">
                        <a href="mailto:{reply_to}?subject=Unsubscribe" style="color: #0066cc; text-decoration: none;">Unsubscribe</a>
                        <span style="color: #ddd; margin: 0 10px;">|</span>
                        {formatted_address.replace('\n', '<br>')}
                    </p>
                </div>
            </body>
            </html>
            """
            
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
        
        # Set priority headers
        msg['X-Priority'] = '1'  # High priority
        msg['Importance'] = 'high'
        
        # Clean email addresses
        def clean_email(email_str):
            return ''.join(c for c in email_str if ord(c) < 128).strip()
            
        clean_recipient = clean_email(email_data.recipient)
        clean_sender = clean_email(email_data.sender)
        
        # Send email using SMTP
        with smtplib.SMTP(email_data.smtp_host, email_data.smtp_port, timeout=30) as server:
            server.ehlo()
            if server.has_extn('STARTTLS'):
                server.starttls()
                server.ehlo()
            server.login(clean_sender, email_data.password)
            
            # Update headers with cleaned emails
            msg.replace_header('From', clean_sender)
            msg.replace_header('To', clean_recipient)
            
            # Send email
            raw_message = msg.as_string()
            server.sendmail(clean_sender, [clean_recipient], raw_message)
            
            logger.info(f"Email sent successfully to {clean_recipient}")
            
            return {
                "status": "success",
                "message": "Email sent successfully",
                "to": clean_recipient,
                "subject": email_data.subject
            }
            
    except Exception as e:
        import traceback
        error_type = type(e).__name__
        error_msg = f"Error Type: {error_type}"
        error_msg += f"\nError Message: {str(e)}"
        logger.error(f"Email sending failed: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "error": f"Failed to send email: {str(e)}",
                "type": error_type
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
