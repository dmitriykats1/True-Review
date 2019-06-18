# True Review - A personalized Yelp Restaurant Recommender

![title](https://user-images.githubusercontent.com/47621473/59706049-7aafe780-91b4-11e9-8f7e-06acd0a64eaa.png)

Yelp has been around since 2004, helping people find great local businesses by presenting an easy to read star rating along with more detailed users’ reviews. By the beginning of 2019, the number of users and reviews has exploded to over 100M and 184M, respectively. We are going to look at a small subset of the data in order to draw insights in review trends and restaurant trends and use these insights to guide users to a better experience by predicting their personal rating of a specific restaurant. 

### Prerequisites

Library requirements are located in the requirements.txt file

### Data

Yelp dataset can be obtained from:
https://www.yelp.com/dataset/challenge

Dataset Documentation:
https://www.yelp.com/dataset/documentation/main

## Jupyter Notebooks

There are five notebooks in the "Notebook" folder which contain all code used to generate the report and predictive models. Notebooks are labeled per the documentation in the README file, and can be viewed in the ascending order.

## Cleaning and combining data for analysis

 - Queried the tables only with restaurants that are not fast-food
 - Joined the restaurant data with review data
 - Removed any users with no friends and less than 200 reviews
 - Limited the analysis to one city

## Cleaning, combining, and feature engineering data for modeling

 - Extracted category data and added features to the Scottsdale review dataset
 - Using LDA Topic modeling added features to the Scottsdale review dataset

## Modeling

In order to predict a given user's restaurant rating based on their past experiences and reviews, we'll use multiple Machine Learning Algorithms to build a more accurate predictive model. 
With extremely highly dimensional datasets, such as a user / item matrix, it can be easy to overfit traditional models to specific features rather than the underlying themes of the data. Using singular features to extract useful information will result in poor performance. Instead, we can use a technique called Singular Value Decomposition (SVD) to reduce a large set of features into a smaller set of themes.
Expressing each user’s taste/style in a single vector of k-taste/style values and each restaurant’s description in same k-taste/style values, then we can compute a dot product to predict how much a user will like a given restaurant.
We'll be using Surprise's SVD algorithm to make initial predictions. Documentation can be found here:
https://surprise.readthedocs.io/en/stable/index.html

### Boosting SVD predictions

In order to improve SVD predictions we'll use Random Forest Classifier and Regressor algorithms to predict whether a user will like a given restaurant and what rating are they likely to give, respectively. We'll utilize each user's unique review style which was extracted from the LDA Topic modeling to help the algorithm learn users' preferences. 
Once the training is complete we'll fit the entire dataset and combine SVD, RF Classifier, and RF Regressor into a single predicted star value for the given user/restaurant combination.

## Results

Due to the nature of the dataset, many features used for our modeling were readily available from Yelp. However, given the strong correlation between a users’ review text and their rating, we needed to extract as much information from the reviews as possible to improve our modeling techniques. Using LDA Topic modeling we were able to generalize 25,000 reviews into just 50 topics that are easy to understand. Using these additional “features” we predicted a given user’s rating and whether they would like a specific restaurant.
Numerous models were used on the dataset to make predictions. RF Classification and Regression, and Surprise SVD were used in combination to come up with final user rating for a given restaurant. Alone, these models performed relatively good, but combining them resulted in a much better prediction as measure by total RMSE of 0.74.

 - RMSE - SVD + RF-Classifier = 0.91

 - RMSE - RF  = 0.94 - INITIAL
 - RMSE - RF  = 0.89 - TUNED

 - RMSE - SVD + RF-Classifier + RF-Regressor = 0.74



Although the final goal of combining collaborative and content-based models has been achieved, there are many ways to improve the performance of this particular application.
- LDA Topic model can include reviews from entire dataset to provide a wider range of language dynamic and regional dialects
- Use of other regression and classification algorithms can be deployed to improve performance further. (XGBoost, LightGBM, AdaBoost, etc.)
- Combine models through a grid search to optimize the weight of each model on the final results.
- Extract latent features from the SVD model to better understand feature creation and improve
overall results
- Create user profile vectors based on feature preference and use them to help predict similarities
or likes of restaurants with those features

## Authors

* **Dmitriy Kats** - *Initial work* - 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

