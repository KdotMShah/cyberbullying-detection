import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, GlobalMaxPooling1D, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Example dataset (replace with your actual dataset)
texts = ['This is a safe comment', 'This is bullying', 'Leave me alone', 'You are awesome']  # Sample texts
labels = [0, 1, 0, 0]  # 0 = Not Cyberbullying, 1 = Cyberbullying

# Tokenizer and padding
tokenizer = Tokenizer(num_words=5000, lower=True, split=" ")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Ensure that the padded sequences are in the correct shape (num_samples, maxlen)
maxlen = 100  # Length of the input sequences
X = pad_sequences(sequences, maxlen=maxlen)

# Convert labels to numpy array for compatibility
labels = np.array(labels)

# Define the model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=maxlen),
    Conv1D(128, 5, activation='relu'),
    MaxPooling1D(pool_size=2),
    GlobalMaxPooling1D(),
    Dense(10, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Check shape of input data
print(f"Shape of X: {X.shape}")  # Ensure it's of shape (num_samples, maxlen)

# Train the model
model.fit(X, labels, epochs=5, batch_size=32)

# Save the model
model.save('cyberbullying_cnn_model.h5')  # Save the model

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Model and tokenizer saved successfully.")
