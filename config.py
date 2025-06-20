import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# AI Model Configuration
NUM_CLASSES = int(os.getenv("NUM_CLASSES"))
IMG_WIDTH = int(os.getenv("IMG_WIDTH"))
IMG_HEIGHT = int(os.getenv("IMG_HEIGHT"))