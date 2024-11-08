from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import numpy as np

# Define label map and sequences/labels
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

# Load data from npy files into sequences
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            file_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
            try:
                res = np.load(file_path)
                window.append(res)
            except FileNotFoundError:
                print(f"File not found: {file_path}, skipping this sequence...")
                continue
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])

# Convert sequences to numpy arrays
X = np.array(sequences)  # Shape should be (num_sequences, sequence_length, num_features)
y = to_categorical(labels).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# TensorBoard callback
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Model definition
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))  # Use len(actions) instead of actions.shape[0]

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# Print model summary
model.summary()

# Save model architecture as JSON and weights as H5
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save('model.h5')
