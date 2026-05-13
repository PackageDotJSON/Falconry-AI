"""
@author: { MUHAMMAD HASSAN KIYANI, SOFTWARE DEVELOPER, FALCONRY SOLUTIONS }
@description: A list of enum classes that contain the application settings that are immutable
"""

from enum import Enum

class ServerSettings(Enum):
    SERVER_APP = "app:app"
    HOST = "127.0.0.1"
    PORT = 8000
    SERVER_RELOAD = False
    WORKERS = 2

class GenAIUrls(Enum):
    ROUTE_PREFIX = "/gen-ai"
    TAGS = ["Document"]
    CONVERT_DOCUMENT = "/document"