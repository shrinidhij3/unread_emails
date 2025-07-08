from flask import Flask, request, jsonify
import smtplib
import email.utils
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email import policy
import logging
import uuid
from datetime import datetime

app = Flask(__name__)

# Email Configuration
DEFAULT_MAILING_ADDRESS = {
    'name': 'Thorsignia',
    'address_line1': '123 Business Street',
    'city': 'Bangalore',
    'state': 'Karnataka',
    'postal_code': '560001',
    'country': 'India'
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_mailing_address(address_data=None):
    """Format a mailing address from a dictionary."""
    if not address_data:
        address_data = DEFAULT_MAILING_ADDRESS
    
    # Handle both string (legacy) and dict formats
    if isinstance(address_data, str):
        return address_data
        
    # Build address lines
    lines = [address_data.get('name', '')]
    if 'address_line1' in address_data:
        lines.append(address_data['address_line1'])
    if 'address_line2' in address_data and address_data['address_line2']:
        lines.append(address_data['address_line2'])
    
    # Build city/state/zip line
    city_line = []
    if 'city' in address_data and address_data['city']:
        city_line.append(address_data['city'])
    if 'state' in address_data and address_data['state']:
        city_line.append(address_data['state'])
    if 'postal_code' in address_data and address_data['postal_code']:
        city_line.append(address_data['postal_code'])
    
    if city_line:
        lines.append(', '.join(city_line))
    
    if 'country' in address_data and address_data['country']:
        lines.append(address_data['country'])
    
    return '\\n'.join(lines)

@app.route("/send-email", methods=["POST"])
def send_email():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        # Required fields
        sender_email = data.get("from")
        sender_password = data.get("password")
        recipient_email = data.get("to")
        subject = data.get("subject")
        
        # Get mailing address from request or use default
        mailing_address = data.get('mailing_address', DEFAULT_MAILING_ADDRESS)
        
        # Optional fields with defaults
        message_body = data.get("message", "")
        html_content = data.get("html")
        smtp_host = data.get("smtp_host", "smtp.gmail.com")
        smtp_port = int(data.get("smtp_port", 587))
        reply_to = data.get("reply_to", sender_email)

        # Validate required fields
        missing_fields = []
        if not sender_email: missing_fields.append("from")
        if not sender_password: missing_fields.append("password")
        if not recipient_email: missing_fields.append("to")
        if not subject: missing_fields.append("subject")
        
        # Either message or html content is required
        if not message_body and not html_content:
            missing_fields.append("message or html (at least one is required)")
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}",
                "status": "error"
            }), 400

        # Create a simple email message
        msg = MIMEMultipart('alternative')
        
        # Basic headers only
        msg['Subject'] = subject
        msg['From'] = f'<{sender_email}>'
        msg['To'] = recipient_email
        msg['Reply-To'] = reply_to
        msg['Date'] = email.utils.formatdate(localtime=True)
        msg['Message-ID'] = email.utils.make_msgid(domain=sender_email.split('@')[-1])
        
        # Simple unsubscribe header
        msg['List-Unsubscribe'] = f'<mailto:{reply_to}?subject=Unsubscribe>'

        # Ensure we have plain text content
        if not message_body and html_content:
            # Create a simple plain text version from HTML
            import re
            message_body = re.sub(r'<[^>]+>', ' ', html_content)  # Remove HTML tags
            message_body = re.sub(r'\s+', ' ', message_body).strip()  # Normalize whitespace
        
        # Fallback if no content
        if not message_body:
            message_body = "Please enable HTML to view this email."
        
        # Add plain text part (always include this first)
        text_part = MIMEText(message_body, 'plain', 'utf-8')
        msg.attach(text_part)
        
        # Simple HTML version (only if HTML content was provided)
        if html_content:
            # Get the formatted address
            formatted_address = format_mailing_address(mailing_address)
            
            # Basic HTML structure
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
                <title>{subject}</title>
            </head>
            <body style="margin: 0; padding: 20px; font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto;">
                    {html_content}
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

        # Set additional headers for priority
        msg['X-Priority'] = '1'  # High priority
        msg['Importance'] = 'high'

        if not message_body and not html_content:
            return jsonify({"status": "error", "error": "No message content provided"}), 400

        # Clean and validate email addresses
        def clean_email(email_str):
            # Remove any hidden unicode characters
            return ''.join(c for c in email_str if ord(c) < 128).strip()
            
        clean_recipient = clean_email(recipient_email)
        clean_sender = clean_email(sender_email)
        
        # Send the email with proper EHLO/HELO handshake
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.ehlo()
            if server.has_extn('STARTTLS'):
                server.starttls()
                server.ehlo()
            server.login(clean_sender, sender_password)
            
            # Update the From and To headers with cleaned emails
            msg.replace_header('From', clean_sender)
            msg.replace_header('To', clean_recipient)
            
            # Use sendmail with raw message string
            raw_message = msg.as_string()
            server.sendmail(clean_sender, [clean_recipient], raw_message)
            logger.info(f"Email sent successfully to {clean_recipient}")

        return jsonify({
            "status": "success",
            "message": "Email sent successfully",
            "to": recipient_email,
            "subject": subject
        }), 200

    except Exception as e:
        import traceback
        error_type = type(e).__name__
        error_msg = f"Error Type: {error_type}"
        error_msg += f"\nError Message: {str(e)}"
        logger.error(f"Detailed error: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error(f"Error details: {error_msg}", exc_info=True)
        return jsonify({"status": "error", "error": error_msg}), 500

@app.route("/", methods=["GET"])
def home():
    return """
    <h1>SMTP Email Sender</h1>
    <p>Send POST requests to /send-email with JSON data:</p>
    <pre>
    {
        "from": "your-email@example.com",
        "password": "your-password",
        "to": "recipient@example.com",
        "subject": "Test Email",
        "message": "Hello from SMTP Sender!",
        "html": "&lt;h1&gt;Hello!&lt;/h1&gt;&lt;p&gt;This is HTML content.&lt;/p&gt;"
    }
    </pre>
    """

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
