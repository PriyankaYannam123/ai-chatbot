import nltk
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and data structures
lemmatizer = WordNetLemmatizer()

# Define intents (questions and responses)
intents = [
    {"intent": "greeting", "patterns": ["Hi", "Hello", "Good morning", "How are you?"], "responses": ["Hello!", "Good morning!", "How can I assist you today?"]},
    {"intent": "goodbye", "patterns": ["Bye", "Goodbye", "See you later"], "responses": ["Goodbye!", "See you later!"]},
    {"intent": "age", "patterns": ["How old are you?", "What is your age?"], "responses": ["I am a virtual assistant, so I don't age!", "I am just a program, I don't have an age."]},
    {"intent": "thanks", "patterns": ["Thank you", "Thanks", "Appreciate it"], "responses": ["You're welcome!", "Happy to help!"]}
]

# Prepare the data for training
training_sentences = []
training_labels = []
responses = []

for intent in intents:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['intent'])
    responses.append(intent['responses'])

# Lemmatize and tokenize the training data
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

training_sentences = [clean_up_sentence(sentence) for sentence in training_sentences]

# Convert words to numerical data using a bag of words model
words = []
for sentence in training_sentences:
    for word in sentence:
        if word not in words:
            words.append(word)

# Create the bag of words
training_data = []
for sentence in training_sentences:
    bag = []
    for word in words:
        bag.append(1 if word in sentence else 0)
    training_data.append(bag)

# Encode the labels
label_encoder = LabelEncoder()
training_labels = label_encoder.fit_transform(training_labels)

# Build the neural network model
model = Sequential([
    Dense(128, input_shape=(len(words),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(len(responses), activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(np.array(training_data), np.array(training_labels), epochs=200, batch_size=5, verbose=1)

# Function to get a response based on the user's input
def chatbot_response(text):
    sentence_words = clean_up_sentence(text)
    bag = [1 if word in sentence_words else 0 for word in words]
    prediction = model.predict(np.array([bag]))[0]
    predicted_class = np.argmax(prediction)
    response = responses[predicted_class]
    return np.random.choice(response)

# Test the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    print("Bot:", chatbot_response(user_input))
