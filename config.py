import os
from dotenv import load_dotenv

load_dotenv()

NUM_CLASSES = int(os.getenv("NUM_CLASSES"))
IMG_WIDTH = int(os.getenv("IMG_WIDTH"))
IMG_HEIGHT = int(os.getenv("IMG_HEIGHT"))