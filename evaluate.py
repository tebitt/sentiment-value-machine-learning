from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, pad_sequences
from keras.models import load_model
import numpy as np
import pandas as pd


twitter_df = pd.read_csv('twitter_dataset.csv', header=0)

# Use 'Comment_Content' as X and 'Sentiment_Value' as y for training
X = twitter_df['Comment_Content']
y = twitter_df['Sentiment_Value']

# Encoding the target variable
label_encoder = LabelEncoder()
encoded_sentiment = label_encoder.fit_transform(y)
y_cat = to_categorical(encoded_sentiment, num_classes=len(label_encoder.classes_))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=np.random.randint(0, 99))

# Define maximum number of words and maximum sequence length
MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 100

model = load_model('best_model.h5')
# Tokenize the text for the neural network
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(X_train)  # Assuming X_train was used to fit the tokenizer during training
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH)

loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=1)
print("CNN-LSTM Model Evaluation:")
print(f"Accuracy: {accuracy}")
# Predict classes
y_pred = model.predict(X_test_pad)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

target_names = label_encoder.classes_
print("Classification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=target_names, zero_division=1))