"""
@author: { MUHAMMAD HASSAN KIYANI, SOFTWARE DEVELOPER, FALCONRY SOLUTIONS }
@description: A list of global variables related to the configuration of the application
"""

import os
from dotenv import load_dotenv

load_dotenv()

APP_TITLE = "FALCONRY GEN/AGENTIC AI"
APP_VERSION = "1.0"
API_KEY_HEADER_NAME = "X-API-Key"
API_KEY = os.getenv("FALCONRY_API_KEY")