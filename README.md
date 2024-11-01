# CSE151A-Project
### Project Overview:
This project will explore the potential for predicting a user’s rating of a recipe based on textual analysis of their comment as well as their potential history of comments and ratings on other recipes. We chose the problem of predicting how a given user will rate a recipe as the recommender system we are building would give insight into the qualities that are both desired or unwanted in specific recipes. Furthermore, a successful model would allow for improved recipe recommendations based off of a user’s past reviews. We will be using methods such as logistic regression to perform sentiment analysis on text to get the general sentiment of the text, as well as utilizing other features of the data such as the most important words or length of the text, we will fit these features on a regression model to determine if these features are an accurate portrayal of a user’s rating on a recipe. 
### Data Preprocessing:
- Missing Values
  - Drop the two missing text entries from the dataset to maintain data consistency.
- Text Tokenization and Cleaning
  - Remove special characters, numbers, and punctuation from the text attribute.
  - Convert all text to lowercase, tokenize the text into individual words, and stored the tokenized text into a list for futher analysis.
- Encoding
  - Encode the categorical attributes, recipe_name and user_name, using one hot encoding for the use in the model. 
- Scaling
  - Scale the numerical attributes, reply_count, thumbs up/down, best_score, and user_reputation, using normalization to ensure range.
- Splitting Data
  - Split the dataset into training and testing sets using an 80:20 ratio to evaluate model performance and avoid overfitting.
### Jupyter Notebook
https://colab.research.google.com/drive/1B-XxFEforvRwHhIIadXGSZB1HqIw5l3e?usp=sharing
