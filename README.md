
![Banner](ProjectBanner.jpg)

# Predicting AirBnB Review Scores

Names:
- Artur Rodrigues, arodrigues (at) ucsd (dot) edu 
- Doanh Nguyen, don012 (at) ucsd (dot) edu 
- Ryan Batubara, rbatubara (at) ucsd (dot) edu

**NOTE:** This README is also available as a website [here](https://doanhandonly.github.io/RAD-CSE151A-Project/)!

## Table of Contents
- [Predicting AirBnB Review Scores](#predicting-airbnb-review-scores)
  - [Table of Contents](#table-of-contents)
  - [Abstract](#abstract)
  - [Dataset](#dataset)
  - [Data Preprocessing](#data-preprocessing)
    - [Dropping Unecessary Columns](#dropping-unecessary-columns)
    - [Dropping 0 Review Listings](#dropping-0-review-listings)
    - [Fixing Datatypes](#fixing-datatypes)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Choosing a First Model](#choosing-a-first-model)
  - [Model Evaluation](#model-evaluation)
  - [Model on Fitting Graph](#model-on-fitting-graph)
  - [Model Improvements](#model-improvements)
  - [Conclusion and Next Model](#conclusion-and-next-model)


## Abstract

With the end of the Covid 19 pandemic, there has been a huge boom in travel and entertainment industries worldwide. With the pandemic still in the minds of many people, travelers may tend to seek more private and personal accommodations such as those on AirBnB. Here, we predict the review score (on a scale of 0 to 100) of an AirBnB posting based on various features, like amenities available, number of reviews, price, and others. The intention is that this may provide AirBnB hosts better insight into what makes a highly rated AirBnB experience.

## Dataset

This project will be based on data gathered by [Inside AirBnb](https://insideairbnb.com/get-the-data/) May to June 2024. To keep our analysis more focused, we will only be analyzing AirBnB listings from the United States. Since Inside AirBnB only offers datasets per city, we have downloaded all US cities with AirBnB listings and combined them into one csv file. Due to the size of this file, [Inside AirBnB reposting policies](https://insideairbnb.com/data-policies/), and [Github Data storage policies](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-storage-and-bandwidth-usage), we will not be uploading this combined file to the repository. That said, the combined dataset is available [here](https://drive.google.com/file/d/1DwNaHBBgTesytLoGn23QZMURfK41Du2K/view?usp=sharing), but requires a UCSD account.

A data dictionary for the data can be found at [Inside AirBnB's data dictionary](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit?gid=1322284596#gid=1322284596).

## Data Preprocessing

Our data preprocessing can be split into three steps:

### Dropping Unecessary Columns

Some columns in the original data are unecessary for our purposes. For a detailed description of each column, see the [Inside AirBnB Data Dictionary](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit?gid=1322284596#gid=1322284596).

We list reasons for dropping these columns:
- `All URL`: Unique elements for each listing. Does not contribute anything when predicting the review score.

- `All ID`: Unique elements for each listing. Does not contribute anything when predicting the review score.

- `host_name`: Indiviudally unique elements for each listing. Does not contribute anything when predicting the review score.

- `license`: Unique elements for each listing. Does not contribute anything when predicting the review score.

- `source`: Holds whether or not the listing was found via searching by city or if the listing was seen in a previous scrape. There is no logical
connection between this and the target variable, which is review score.

- `host_location`: Private information.

- `host_total_listings_count`: There exists another feature called `host_listings_count`, this is a duplicate feature.

- `calendar_last_scarped`: Holds the date of the last time the data was scrapped, no logical connection between this and predicting `review_score_rating`.

- `first & last review`: provides temporal data for the first & last review date. Last review date can be misleading as an unpopular listing may have no reviews for an extended amount of time, and suddenly get a review.

- `minimum_minimum_nights, maximum_minimum_nights, minimum_maximum_nights, maximum_maximum_nights`: The all time minimum and maximum of a listing's minimum and maximum nights requirement for booking. This has no correlation to review score because you cannot write a review if you have not stayed at the listing. A person who wants to book a listing for 10 days is not going to book a listing that has a maximum night stay of 9 days.

### Dropping 0 Review Listings

Since we are trying to predict AirBnB review scores (for the purpose of finding out what makes a good review score), we will be dropping all listings that have 0 reviews. This is because, from the perspective of our model, these rows do not provide any meaningful information into what makes a highly rated listing (since their review scores are `NaN`, and are missing by design).

We remark that this still leaves us with almost 200 thousand rows, so the data remains large enough for a model.

### Fixing Datatypes

Some columns have incorrect datatypes, such as:
- `host_response_rate` and `host_acceptance_rate` should be change from a `str` percent to a `float`.
- `host_is_superhost`, `host_has_profile_pic`, `host_identity_verified`, `has_availability`, `instant_bookable` should be booleans.
- `last_scraped` and `host_since` should be Pandas timestamps.
- `price` should be a float in dollars.
- `amenities` should be a list.

## Exploratory Data Analysis

This exploratory data analysis will be split into three parts:

- [General Data EDA](#general-data-eda), where we visualize general information about the dataset.
- [Numerical Data EDA](#numerical-data-eda), where we see how numerical features relate to predicting review scores.
- [Text and Categorical Data EDA](#text-and-categorical-eda), where we see how textual and categorical data may help our predictions.

You can see our EDA in the Jupyter notebook called `eda.ipynb` in the `eda` folder [here](./eda/eda.ipynb).

![Model 1 Banner](Model1Banner.png)

## Choosing a First Model

At this point, we have two groups of features:
- Categorical Columns, encoded into some sort of numerical feature as described in the previous section.
- Numerical Columns, which is essentially every other feature in our dataset.

We plan to put all of these features into a LinearRegression model from sklearn. There are three main reasons a LinearRegression model is ideal for our base model:
1. It is relatively simple, and features almost entirely dictate how well the model performs. This allows us to focus on enginnering good, relevant features for our second, more complicated model.
2. LinearRegression allows us to check the coefficient of every variable, and when standardized or normalized, a sense of which features play the largest role in determining the review score.
3. It is extremely fast and easy to implement LinearRegression, allowing us to test many different features and encodings quickly.

We summarize our preprocessing and features below:

```python
preproc = make_column_transformer(
    (SubstringsTransformer(names_meaningful), 'name'),
    (SubstringsTransformer(desc_meaningful), 'description'),
    (WeekTransformer(), 'host_since'),
    (LengthTransformer(), 'host_verifications'),
    (OneHotEncoder(handle_unknown='ignore'), ['property_type']),
    (OneHotEncoder(handle_unknown='ignore'), ['room_type']),
    (LengthTransformer(), 'amenities'),
    (StandardScaler(), numeric_features)
)
```

The above preprocessor simply applies the given transformer to the columns on the second item in the tuple. As such, all that is left to do is to put this preprocessor into a pipeline alongside our model. Note however, that sklearn cannot support missing values, and so we will conduct mean imputation for these missing values. This is because LinearRegression (based on minimizing mean_square_error) will not get any better nor any worse if a data point with all mean values is added or removed from the dataset. We also provide an illustration of our model below:

![First Model Pipeline](first_model.png)

## Model Evaluation

We now do 10-fold cross validaton and report the metrics for each fold.

|              |   mean_squared_error |   mean_absolute_error |   r2_score |
|:-------------|---------------------:|----------------------:|-----------:|
| (0, 'train') |             0.12613  |              0.198191 |  0.0879473 |
| (0, 'test')  |             0.130564 |              0.20048  |  0.08552   |
| (1, 'train') |             0.127414 |              0.199016 |  0.0872061 |
| (1, 'test')  |             0.118979 |              0.195752 |  0.0924823 |
| (2, 'train') |             0.126681 |              0.198671 |  0.0875633 |
| (2, 'test')  |             0.125549 |              0.19663  |  0.0893076 |
| (3, 'train') |             0.127029 |              0.198596 |  0.0875516 |
| (3, 'test')  |             0.123754 |              0.197797 |  0.0795305 |
| (4, 'train') |             0.126899 |              0.198766 |  0.0880649 |
| (4, 'test')  |             0.123835 |              0.196832 |  0.0828053 |
| (5, 'train') |             0.12554  |              0.197758 |  0.0881043 |
| (5, 'test')  |             0.135944 |              0.203179 |  0.0836539 |
| (6, 'train') |             0.126947 |              0.198569 |  0.0878757 |
| (6, 'test')  |             0.123132 |              0.198147 |  0.0866351 |
| (7, 'train') |             0.126227 |              0.198311 |  0.0874304 |
| (7, 'test')  |             0.129676 |              0.198832 |  0.0900592 |
| (8, 'train') |             0.126344 |              0.198277 |  0.0879102 |
| (8, 'test')  |             0.128622 |              0.199914 |  0.0859053 |
| (9, 'train') |             0.126359 |              0.198572 |  0.0884995 |
| (9, 'test')  |             0.128394 |              0.198563 |  0.0812679 |

We summarize it by taking the averages below:

|       |   mean_squared_error |   mean_absolute_error |   r2_score |
|:------|---------------------:|----------------------:|-----------:|
| test  |             0.126845 |              0.198613 |  0.0857167 |
| train |             0.126557 |              0.198473 |  0.0878153 |

We will now interpret the above metrics in more detail. We start with the `mean_absolute_error`, which tells us that on average our predicted rating is 0.19 off for both the test and train cases. The similarity between these values tell us that our model has not overfitted, but the large values (considering review ratings go from 0 to 5 only) show us that the model is not very good.

Same can be said for our `mean_squared_error`. We note that its value is smaller than the `mean_absolute_error`, which makes sense as it is roughly the square root of the `mean_absolute_error`.

The `r2_score` however tells a very interesting story. The `r2_score` is a correlation metric that goes from 0 to 1, where 0 implies no correlation and 1 is identical correlation. Our low value of 0.08 tells us that despite the large number of features, we still have not effectively represented the data. This makes sense - it is very hard to even describe in words what would make one AirBnb listing more highly rated than the other. However, this also gives us plenty of room to improve in our second model.

## Model on Fitting Graph

Based on our relatively high MSE, and generally almost identical test and train MSE, MAE, and r2, it is safe to say that our model is not that far along the fitting graph. In other words, our model has not overfitted the data since we see very similar values between the test and train metrics. This is very good for us, as there is much room to add more model complexity - such as by adding more features, or changing to a more complex model - and improve the metrics in our second, better model.

## Model Improvements

Our results are not ideal, we have a relatively high MSE and MAE, and our r^2 score isn't where we would like it to be. This is unfortunate as we were hoping for better results, however, the good part of this is that firstly, we can be pretty confident that we are not overfitting the data and secondly, we have a lot of room for improvement both for our models, and for our machine learning skills as we this will allow us to further develop/apply the things that we are learning in this class to improve our model.

Here are some ways in which we may be able to improve our model:
- Increasing the number of features
- Better feature engineering
- Trying different complexity models such as polynomial regression or neural networks
- We could try using coalesced features.

## Conclusion and Next Model

Since our mean squared error of our first model being linear regression is not exactly ideal, we decided that the next potential model we can train and test would be a **neural network**. Since the fit of the linear regression line is producing a unoptimal mean squared error, we believe that a simple linear best fitting line is simply not good enough to fit our supposedly complex data. By using a neural network and messing around with the amount of neurons per hidden layer, the amount of hidden layers, and the many activation functions, we add more depth and complexity in hopes of finding the best fit that captures the relationship between our features while in turn producing a fairly accurate prediction.   