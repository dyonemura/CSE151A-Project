# CSE151A-Project
---
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

---
# Methods

## Model One

### - Analysis

For our first model we utilized a SGDClassifier, which is a linear classifier that utilizes stochastic gradient descent. This model takes in the td-idf features of the processed text data and outputs a prediction of the stars (rating) category for the given review. In order to deal with the skew in our dataset towards five-star reviews, we utilized SMOTE to oversample our data and give a more uniform sample sapce to operate on. Our first model yielded a 84% accuracy on the training set and a 67% accuracy on the testing set. The combination of a high accuracy on the training set and a significant difference (17%) between the training and testing accuracy indicates that this model is most likely suffering from overfitting. As there are five different outcomes for this classifier (1-5 stars), having a testing accuracy of 67% is well above the random threshold of 20%. 

### - Fitting Graph Location

Based off of our current error rates, we see a low error on the training set and a comparatively higher error on the test set. Based off of the fitting graph, this would indicate we are beyond the ideal range for model complexity and must seek to reduce the complexity of our model to help lessen the gap between the testing and training erorr. Our model is relatively high in complexity as a result of the sheer number of features which are generated by the if-idf values. In the future, we will seek to reduce this aspect of complexity by continuing to increase the granularity of the cleaning of text feature. Through the reduction of false duplicates ('hamburger' vs 'hamberger') and error text ('freerange' vs 'free range') we will reduce the number of tf-idf features and thus create a more generalizable model via the reduction of model complexity. Furthermore, we believe that including some other features such as overall sentiment or review length could also positively impact the accuracy of our model without significantly increasing the overall model complexity.  

### - Future Models
In the future, we are considering looking at potentially utilizing a Naive Bayes model such as Sci-Kit Learn's MultinomalNB model. The Naive Bayes model seems ideal as it is able to efficiently handle the volume of data at hand and will effectively utilize the sparse nature of the TF-IDF feature vectors more effectively than the linear regression classifier. Furthermore, we are also interested in looking into utilizing a Random Forest model to see if the leveraging of decision trees would help to aid in more effective classification in the context of having an extreme number of features.   

## Model Two

### - Analysis

For our second model, we utilized a Multinomial Naive Bayes classifier, a probabilistic model that is commonly used for text classification tasks. This model was chosen for its suitability in handling sparse data, such as the TF-IDF feature vectors derived from the processed text. To address the issue of class imbalance, we trained the MNB model using the same oversampled dataset employed for the first model. However, we adjusted the parameters of our TF-IDF generation process by limiting the number of words to be considered. This culled out a significant number of words that saw use in <0.1% of the recipe reviews. By new reducing the complexity of our input data, we thus directly reduce the complexity of our model. Initially, the model was trained with the alpha parameter set to 0.01, yielding a training accuracy of 88% and a testing accuracy of 63%. To optimize the model, we conducted a grid search over the hyperparameters `alpha` and `fit_prior`. Retraining the model with the parameters found by grid search—alpha=0.0001 and fit_prior=False—did not affect the training accuracy, but it did bump the testing accuracy up to 64%. Additionally, we performed manual tuning over a larger range of alpha values and did not find a better configuration. By graphing these results, we were able to conclude that 64% was our threshold in regards to hyperparameter tuning.  These results mark a slight decrease in testing accuracy compared to the first model, and the similarly large gap between training and testing performance indicated that overfitting was still an issue.


### - Fitting Graph Location

By reducing model complexity, we aimed to make the model more generalizable; however, the testing error is still high compared to the trainining error, indicating that our second model is also in the overfitting region despite these efforts.


### - Future Models 

In addition to Random Forest, which was mentioned previously, we may want to consider an XGBoost model. XGBoost is well-suited for handling data with high dimensionality, such as our TF-IDF feature vectors, and is designed to efficiently manage sparse data. It also has the ability to address class imbalance through built-in regularizaion techniques, making it a promising candidate for improving classification performance while potentially mitigating overfitting.

---
# Conclusion
## Model One
Overall, our first model served as a good launch point for our investigation of this dataset. The SGDClassifier yielded a relatively low training error and a *comparatively* higher testing error! This indicates that there is significant room to grow in terms of changing our features and underlying model in order to venture closer to the ideal level of model complexity which would yield a more generalizable model. Based off of these error rates, we can conclude that utilization of the TF-IDF of words yields a better prediction than randomly selecting a rating; however, there is still significant room to improve and increase the model's overall efficacy. More specifically, we believe that the model could be improved by cleaning the text data even more thoroughly to help increase the accuracy of the TF-IDF values. This would help to reduce the complexity of our model by decreasing the number of features we are inputting. Furthermore, we are seeking to explore the inclusion of other attributes of the text as features in our model. Whether this surrounds something simple, such as review length, or something more complicated, such as sentiment, we hope that pulling more information from the review text column will help to improve our model!   

## Model Two
Our second model, the Multinomial Naive Bayes classifier, demonstrated a slightly decreased performance compared to the SGDClassifier, with a testing accuracy of 64% vs. 67%. Additionally, the large gap between training and testing accuracies (24%) persists, indicating that overfitting remains a key challenge despite taking steps to reduce feature dimensionality. The hyperparameter tuning we performed confirmed that we has found the configuration that yielded the best performance, suggesting that further improvements will not come from additional tuning. Instead, some options to improve performance include employing more robust cross validation techniques (eg. k-fold) to help prevent overfitting to the training data, exploring other types of regularization techniques, and possibly exploring other methods for addressing class imbalance.
