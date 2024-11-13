from app import app  # Import the Flask app from app.py
from flask import Flask

# Entry point to start the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
