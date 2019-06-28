# True Review - A Personalized Yelp Restaurant Recommender

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

In order to improve SVD predictions we'll use a stacked ensemble model with multiple regression algorithms to predict what rating a user is likely to give. We'll utilize each user's unique review style which was extracted from the LDA Topic modeling to help the algorithm learn users' preferences. 
We'll also deploy a content-based filtering system by generating user and restaurant profiles and then compute the similarities for each user/restaurant combination. Combining this method with a RF Classifier, and stacked ensemble algorithm, we'll produce a final predictions for each user.

Stacked Ensemble Methodology

![stacked](https://user-images.githubusercontent.com/47621473/59885167-39b00280-936f-11e9-87cb-ffb8307a95be.png)

Content-Based Filtering Methodoly

 - Create restaurant attribute matrix

![user_att](https://user-images.githubusercontent.com/47621473/59885149-28ff8c80-936f-11e9-8d82-1ea674feb0a6.png)

 - Create user profiles

![user_profile](https://user-images.githubusercontent.com/47621473/59885157-2ef56d80-936f-11e9-9371-a63d464996d2.png)

 - Create prediction matrix

![final_predict](https://user-images.githubusercontent.com/47621473/59885164-36b51200-936f-11e9-843e-7f0390c7a695.png)

## Results

Due to the nature of the dataset, many features used for our modeling were readily available from Yelp. However, given the strong correlation between a users’ review text and their rating, we needed to extract as much information from the reviews as possible to improve our modeling techniques. Using LDA Topic modeling we were able to generalize 25,000 reviews into just 50 topics that are easy to understand. Using these additional “features” we predicted a given user’s rating and whether they would like a specific restaurant.

Numerous models were used on the dataset to make predictions.

### Classifier Models - Precision / Recall

| Model    | Precision / Recall | Tuned       |
|----------|:-------------:|-------------|
| RF       |  0.84 / 0.67  | 0.80 / 0.82 |
| RF-Multi |  0.40 / 0.39  | 0.41 / 0.44 |
| XGB-Multi|  0.41 / 0.43  | 0.44 / 0.46 |

 - RF: Random Forest Classifier
 - RF-Multi: Random Forest Multi-class Classifier
 - XGB-Multi: XGBoost Multi-class Classifier


### Stacked Ensemble - RMSE

| Model    |     RMSE      |  Tuned |
|----------|:-------------:|--------|
| XGBoost  |  0.89         | 0.87   |
| LightGBM |  0.94         | 0.94   |
| AdaBoost |  0.96         | 0.95   |
| Linear Regression - Stacked Ensemble |  | 0.88|


### Weighted Average Ensemble - RMSE

| Model    |     RMSE      | 
|----------|:-------------:|
| Linear Regression - Stacked Ensemble |  0.88         | 
| SVD|  0.88        |
| Weighted Ensemble |  0.84       |


### Content Filtering Model - Precision / Recall

| Model    | Precision / Recall |
|----------|:-------------:|
| CF       |  0.75 / 0.87  | 

 - CF: Content Filtering Algorithm


### Weighted Average Ensemble - Content and Collaborative - RMSE

| Model    |     RMSE      | 
|----------|:-------------:|
| Weighted Average |  0.84 | 



Although the final goal of combining collaborative and content-based models has been achieved, there are many ways to improve the performance of this particular application.
- LDA Topic model can include reviews from entire dataset to provide a wider range of language dynamic and regional dialects
- Combine models through a grid search to optimize the weight of each model on the final results.
- Extract latent features from the SVD model to better understand feature creation and improve
overall results


## Authors

* **Dmitriy Kats** - *Initial work* - 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

