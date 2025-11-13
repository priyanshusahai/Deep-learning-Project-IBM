Sentiment Analysis on IMDB Movie Reviews using LSTM

📘 Project Overview
This project performs Sentiment Analysis on the IMDB dataset of 50,000 movie reviews using a Long Short-Term Memory (LSTM) neural network. The goal is to classify each review as positive or negative based on the text content.

The model is trained using TensorFlow and Keras, with text preprocessing, tokenization, and sequence padding applied to handle natural language data efficiently.

🧠 Key Steps in the Project

1. Dataset Acquisition
The IMDB dataset is downloaded directly from Kaggle using the Kaggle API.
The dataset (IMDB Dataset.csv) contains 50,000 movie reviews with corresponding sentiment labels (positive or negative).

2. Data Preparation
The dataset is split into training (80%) and testing (20%) sets.
A Tokenizer from Keras is used to convert the text data into numerical sequences.
The sequences are padded to a fixed length of 200 to ensure uniform input dimensions for the neural network.

3. Model Architecture
A Sequential Neural Network is built using Keras with the following layers:
Embedding Layer: Converts words into dense vector representations (word embeddings).
LSTM Layer: Captures long-term dependencies in text data.
Dense Layer (Sigmoid): Outputs a single probability value indicating sentiment (positive or negative).

⚙️ Model Compilation and Training
Loss Function: Binary Crossentropy (for binary classification)
Optimizer: Adam (efficient gradient-based optimization)
Metrics: Accuracy
The model is trained for 5 epochs with a batch size of 64 and an 80–20 validation split.

📊 Model Evaluation

After training, the model is evaluated on the test dataset to measure its performance:

loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

This provides an estimate of how well the model generalizes to unseen data.

💬 Sentiment Prediction
A helper function predict_sentiment(review) is defined to predict the sentiment of any new review:

new_review = "This movie was fantastic. I loved it."
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")


✅ Example outputs:

"This movie was fantastic. I loved it." → Positive
"This movie was not that good." → Negative
"This movie was ok but not that good." → Negative

📁 Technologies Used

Python
TensorFlow / Keras
Pandas
Scikit-learn
Kaggle API

🚀 Conclusion

This project demonstrates how deep learning models, specifically LSTMs, can effectively analyze the sentiment of movie reviews. By preprocessing text data and using sequential models, we can capture contextual information and classify text with high accuracy.
