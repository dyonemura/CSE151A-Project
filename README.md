# CSE151A-Project
---
## Introduction
Through the internet, people from all across the world are able to connect with each other in an instant. This has allowed people from all walks of life to share and receive content, and this explosion of user-generated content has given rise to the importance of an individual's opinion. This can most notably be seen through online shopping. It is extremely rare that someone buys a product without first checking the reviews left by those who have purchased the product. This is because reviews are the lifeblood of a product. Products plagued with bad reviews are an excellent deterrent for others to purchasing those specific products, while products praised with good reviews will most likely bring in more people to buy it since it has been proven to be a good product. This is the foundation for a recommender system or a system that uses an algorithm to recommend users products that they are likely to purchase, and this type of system is important since it brings benefits to both the user and company. For the user, a well built recommender system allows for a user to receive products relevant to them without the effort of having to look through specific products to find the one they need, nor spend too much time sifting through poorly rated products. For the company, a good recommender system can massively improve revenue and keep users on their platform. With this in mind, our group wanted to attempt to make a recommender system, not based on products, but on food recipes.

This project will explore the potential for predicting a user’s rating of a recipe based on textual analysis of comments created by users when reviewing a recipe. We will utilize techniques such as Term Frequency-Inverse Document Frequency, Stochastic Gradient Descent, and Naive Bayes to try to get the most important features of a review’s text and try to produce a model that can transform these features into a prediction. The review data we will be using has been retrieved from the UC Irvine Machine Learning Repository under the title “Recipe Reviews and User Feedback” and contains the necessary features to undergo this task.  

We chose the problem of predicting how a given user will rate a recipe, given their comment because we felt that building such a system for recipes would give insight into the qualities that are both desired or unwanted in specific recipes.

---
## Methods

### Data Exploration:
We explored the data of a single user to see whether any useful information could be extracted from this alone. In other words, we wanted to see if it was possible to predict the rating left by a particular user based on their history of reviews and comment stats. For this selected user, we generated a correlation matrix and a pairplot with the following features: number of replies, number of thumbs up/down, best comment score, and number of stars given.

Next, we explored the possibility of extracting information from a single recipe and the users who reviewed it. We chose one recipe and looked at the distribution of ratings as well as review text length, and explored whether there was a relationship between review length and rating given for that recipe.

  
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
# Results
### Data Exploration  
From exploring a single user:
  - Correlation matrix showing relationship between user-level features and number of stars:
    
    ![correlation map](./figures/correlation_map.png)
    - None of these features showed strong correlations to number of stars. The highest were number of thumbs down and best score, both with correlations of 0.3.
      
  - Pairplot of the same features:
    
    ![pairplot](./figures/pairplot.png)
    - Our data for this particular user is too sparse to glean any meaningful information from the pairplot.
   
From exploring a single recipe:
  - Scatterplot showing relationship between review text length and rating for one recipr:
 
    ![stars_textlen](./figures/stars_textlen.png)
    
    - From the plot, we found that there exists some correlation between the length of a review's text, and the star rating given, particularly shown between a longer review and a five star rating.
   


---
# Discussion
---
# Conclusion
## Model One
Overall, our first model served as a good launch point for our investigation of this dataset. The SGDClassifier yielded a relatively low training error and a *comparatively* higher testing error! This indicates that there is significant room to grow in terms of changing our features and underlying model in order to venture closer to the ideal level of model complexity which would yield a more generalizable model. Based off of these error rates, we can conclude that utilization of the TF-IDF of words yields a better prediction than randomly selecting a rating; however, there is still significant room to improve and increase the model's overall efficacy. More specifically, we believe that the model could be improved by cleaning the text data even more thoroughly to help increase the accuracy of the TF-IDF values. This would help to reduce the complexity of our model by decreasing the number of features we are inputting. Furthermore, we are seeking to explore the inclusion of other attributes of the text as features in our model. Whether this surrounds something simple, such as review length, or something more complicated, such as sentiment, we hope that pulling more information from the review text column will help to improve our model!   

## Model Two   
Our second model, the Multinomial Naive Bayes classifier, demonstrated a slightly decreased performance compared to the SGDClassifier, with a testing accuracy of 64% vs. 67%. Additionally, the large gap between training and testing accuracies (24%) persists, indicating that overfitting remains a key challenge despite taking steps to reduce feature dimensionality. The hyperparameter tuning we performed confirmed that we has found the configuration that yielded the best performance, suggesting that further improvements will not come from additional tuning. Instead, some options to improve performance include employing more robust cross validation techniques (eg. k-fold) to help prevent overfitting to the training data, exploring other types of regularization techniques, and possibly exploring other methods for addressing class imbalance.  

---
# Statement of Collaboration
David Yonemura: Coder: Coded the exploratory data analysis, only on the entire dataset and not on the specific users and recipes. Also coded the text clean up, TF-IDF, oversampling, and SGDClassifier code.
