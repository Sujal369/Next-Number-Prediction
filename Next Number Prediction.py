import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define the input sequence
a = float(input())
b = float(input())
c = float(input())
d = float(input())
e = float(input())
f = float(input())
g = float(input())
h = float(input())
i = float(input())
j = float(input())
k = float(input())
sequence = [a, b, c, d, e, f, g, h, i, j, k]

# Prepare the data for training
X, y = [], []
sequence_length = 3  # You can adjust this to use a different length of the input sequence

for i in range(len(sequence) - sequence_length):
    X.append(sequence[i:i+sequence_length])  # Use a sequence of specified length as input
    y.append(sequence[i+sequence_length])    # Predict the next number

X = np.array(X)
y = np.array(y)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=100, verbose=0)

# Predict the next number after 8
next_sequence = np.array([5, 6, 3]).reshape(1, sequence_length, 1)
next_number = model.predict(next_sequence)[0][0]

print("Predicted Next Number:", round(next_number))
