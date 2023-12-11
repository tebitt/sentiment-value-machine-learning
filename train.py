import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, pad_sequences
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
twitter_df = pd.read_csv('twitter_dataset.csv', header=0)

# Use 'Comment_Content' as X and 'Sentiment_Value' as y for training
X = twitter_df['Comment_Content']
y = twitter_df['Sentiment_Value']
#print the unique values of y and their amount
print(y.value_counts())

# Encoding the target variable
label_encoder = LabelEncoder()
encoded_sentiment = label_encoder.fit_transform(y)
y_cat = to_categorical(encoded_sentiment, num_classes=len(label_encoder.classes_))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.05, random_state=np.random.randint(0, 99))

# Define maximum number of words and maximum sequence length
MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 100


# Tokenize the text for the neural network
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad the sequences to have uniform length
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH)

print("X_train_pad shape:", X_train_pad.shape)
print("y_train shape:", y_train.shape)
print("X_test_pad shape:", X_test_pad.shape)
print("y_test shape:", y_test.shape)
# Define the CNN-LSTM model
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, 128, input_length=MAX_SEQUENCE_LENGTH))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(Conv1D(32, 3, activation='relu'))  # Additional Conv1D layer
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(64, dropout=0.6, recurrent_dropout=0.6))
model.add(Dense(len(label_encoder.classes_), activation='softmax', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0008), metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

model.fit(X_train_pad, y_train, epochs=60, batch_size=128, validation_split=0.05, callbacks=[early_stopping, model_checkpoint])

# Save the LSTM model
model.load_weights('best_model.h5')

# Evaluate the CNN-LSTM model
loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=1)
print("CNN-LSTM Model Evaluation:")
print(f"Accuracy: {accuracy}")

# Generate predictions
y_pred = model.predict(X_test_pad)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Generate a confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Generate a classification report
target_names = label_encoder.classes_
print("Classification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=target_names, zero_division=1))