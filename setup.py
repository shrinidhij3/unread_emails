from setuptools import setup, find_packages

setup(
    name="email_sender",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'fastapi>=0.68.0',
        'uvicorn>=0.15.0',
        'python-dotenv>=0.19.0',
        'cryptography>=3.4.7',
        'pydantic>=1.8.0',
        'python-multipart>=0.0.5',
        'gunicorn>=20.1.0',
        'uvloop>=0.16.0',
    ],
    python_requires='>=3.7',
)
