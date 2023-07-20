# Amazon-Product-Review--Sentiment-Analysis
Project Title: Sentiment Analysis of Product Reviews using Natural Language Processing
Project Description: In this project, the intern will be building a sentiment analysis model using
Natural Language Processing (NLP) to classify product reviews into positive, negative, or
neutral sentiment. The dataset used for this project is the Amazon Product Reviews dataset that
contains reviews of various products like books, electronics, and home appliances. The goal is
to achieve high accuracy in classifying the reviews into their respective sentiments.
Tasks:
1. Data Preprocessing: Load and preprocess the Amazon Product Reviews dataset. This
includes cleaning the text data, removing stop words, tokenizing the text, and converting
the text to numerical data that can be fed into a machine learning model.
2. Model Building: Build an NLP model using TensorFlow or Keras with the following layers:
● Embedding layer with a pre-trained word embedding like GloVe or Word2Vec
● LSTM layer with 128 units
● Dense layer with 3 neurons and softmax activation function
3. Model Training: Train the model using the preprocessed data. Use the Adam optimizer
with a learning rate of 0.001 and categorical cross-entropy loss function. Experiment with
different hyperparameters like learning rate, batch size, and number of epochs to
achieve high accuracy in classifying the reviews into their respective sentiments.
4. Model Evaluation: Evaluate the model using the test set and report the accuracy score.
Also, plot the training and validation loss and accuracy curves.
5. Model Deployment: Deploy the model on a web application using Flask or any other web
framework.
