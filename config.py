import os
from dotenv import load_dotenv

load_dotenv()

# DeepSeek API Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Database Configuration
DATABASE_PATH = "data/nyc311.db"
CSV_PATH = "311_Service_Requests_from_2010_to_Present.csv"

# LangGraph Configuration
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

# Visualization Configuration
DEFAULT_CHART_WIDTH = 800
DEFAULT_CHART_HEIGHT = 400
