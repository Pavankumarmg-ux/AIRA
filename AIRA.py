import bcrypt
import json
import os
import pyttsx3
from cryptography.fernet import Fernet
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

# Initialize pyttsx3 engine for speech
engine = pyttsx3.init()

# Load Sentence-BERT model and HuggingFace classifier
MODEL_NAME = 'all-MiniLM-L6-v2'
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
model = SentenceTransformer(MODEL_NAME)
classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)

# Text-to-speech function
def speak(text):
    """Convert text to speech."""
    print(f"Aira: {text}")
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say(text)
    engine.runAndWait()

# Encryption and key handling
def load_key():
    """Load or generate an encryption key."""
    if not os.path.exists("secret.key"):
        key = Fernet.generate_key()
        with open("secret.key", "wb") as key_file:
            key_file.write(key)
    with open("secret.key", "rb") as key_file:
        return key_file.read()

fernet = Fernet(load_key())

def encrypt_data(data):
    return fernet.encrypt(data.encode())

def decrypt_data(data):
    return fernet.decrypt(data).decode()

# User data handling
def save_user_data(name, hashed_password):
    user_data = {"name": name, "password": hashed_password}
    with open("user_data.enc", "wb") as file:
        file.write(encrypt_data(json.dumps(user_data)))

def load_user_data():
    if os.path.exists("user_data.enc"):
        with open("user_data.enc", "rb") as file:
            return json.loads(decrypt_data(file.read()))
    return None

# User authentication
def authenticate_user():
    user_data = load_user_data()
    if not user_data:
        name = input("Enter your name: ").strip()
        password = input("Create a password: ").strip()
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        save_user_data(name, hashed_password.decode())
        speak(f"Welcome, {name}! Your account has been created.")
        return name

    speak("Please authenticate yourself.")
    for _ in range(3):
        name = input("Enter your name: ").strip()
        password = input("Enter your password: ").strip()
        if name == user_data["name"] and bcrypt.checkpw(password.encode(), user_data["password"].encode()):
            speak(f"Welcome back, {name}!")
            return name
        speak("Incorrect name or password. Try again.")
    speak("Too many failed attempts. Access denied.")
    exit()

# Intent detection
def load_database():
    try:
        with open("database.json", "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        speak("Error loading intent database.")
        return {}

database = load_database()

def get_intent(user_input):
    intents = list(database.keys())
    if not intents:
        return "unknown"

    # Zero-shot classification
    result = classifier(user_input, intents)
    if result["scores"][0] > 0.5:
        return result["labels"][0]

    # Sentence-BERT fallback
    user_embedding = model.encode([user_input])
    highest_similarity, best_match = 0.0, "unknown"

    for intent, phrases in database.items():
        phrase_embeddings = model.encode(phrases)
        similarity = np.max(cosine_similarity(user_embedding, phrase_embeddings))
        if similarity > highest_similarity:
            highest_similarity, best_match = similarity, intent

    return best_match if highest_similarity > 0.5 else "unknown"

# Main interaction loop
def main():
    user_name = authenticate_user()
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            speak("Goodbye! See you later!")
            break
        intent = get_intent(user_input)
        if intent == "unknown":
            speak("I'm not sure I understand. Can you rephrase?")
        else:
            speak(f"Detected intent: {intent}")

if __name__ == "__main__":
    main()
