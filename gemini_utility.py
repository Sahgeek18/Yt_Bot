from dotenv import load_dotenv
import os
import google.generativeai as genai  # Importing the Google Generative AI library

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("API_KEY") # to get the API key from environment variables
if not api_key:
    raise ValueError("API_KEY is not set in the environment variables.")


# Configuring the gemini model here
genai.configure(api_key=api_key)  # Configure the API key for Google Generative AI

# Defining the model
def load_genai_model():
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    return model  