---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Customer Segmentation and Acquisition using Machine Learning"
subtitle: "Udacity Machine Learning Nanodegree Capstone Project"
summary: "Using Unsupervised Learning to Cluster Customers into segments based on Demographic data and Supervised Learning to predict potential customers"
authors: [Pranay Modukuru]
tags: [Machine Learning, Data Science]
categories: []
date: 2020-04-16T16:45:57+02:00
lastmod: 2020-04-16T16:45:57+02:00
featured: true
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "Photo by [Stefano Alemani](https://unsplash.com/@stefano75?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com/?utm_source=medium&utm_medium=referral)"
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: ["customer-segmentation-acquisition"]
---

## Introduction

This blog post is about the final project that I did in Udacity’s Machine Learning Engineer Nanodegree program. This project is based on real-world data provided by Arvato Financial Solutions. The task is to understand the customer segments of a mail-order company which sells organic products and compare these segments with the general population data to predict probable future customers.

This project is as close as it can get to real-world data science project. It was challenging and fun to do, and I learnt a lot by working on this. I decided to write this post about my learnings.

Structure of this article:
* What is Customer Segmentation
* Project Introduction
* Data Description and Analysis
* Customer Segmentation using Unsupervised Learning
* Predicting future customers using Supervised Learning

## Customer Segmentation

The process of grouping customers into sections of individuals who share common characteristics is called Customer Segmentation. This segmentation enables marketers to create targeted marketing messages for a specific group of customers which increases the chances of the person buying a product. It allows them to create and use specific communication channels to communicate with different segments to attract them. A simple example would be that the companies try to attract the younger generation through social media posts and older generation with maybe radio advertising. This helps the companies in establishing better customer relationships and their overall performance as an organisation.

### Three most common types of Customer Segmentation
Although there are more than three types of customer segmentation, we are going to look at the three most common strategies to do customer segmentation.

#### 1. Demographic Segmentation
The parameters such as age, gender, education, income, financial status, etc. come under the demographics of a person. This kind of segmentation is the most common approach to segment customers since this data is easy to obtain and analyse. Also, the demographics correspond to the most important characteristics of a person which will help the marketers in making informed decisions. For example, an airlines company can send emails about offers on economy class tickets to people coming under a low-income group, and about first-class tickets to high-income groups.

#### 2. Geographic Segmentation

As the name suggests, this kind of customer segmentation is done based on the physical location of a person. An example, in this case, would be a company which manufactures air conditioning systems. It cannot offer the same products to people in India and Iceland.

#### 3. Behavioral Segmentation

This kind of customer segmentation is based on the behavioural data of the customers. The grouping is done based on the purchasing habits, spending habits, brand interactions, browsing history or any other data which corresponds to behaviour or a person. All the targeted ads we see online today use some kind of behavioural segmentation to decide which ad to target to which customer.

## Project Introduction

The data which Arvato has provided in this project is the demographic data of their customers and the demographic data of the general population in Germany. So the task is to do customer segmentation based on the demographic data. The data includes 366 features corresponding to each person, which indicate age, gender, life stage, financial status, family status, family situation, house location, neighbourhood information. These features are only a few of 366 features which are there in the data.

#### Problem Statement
The problem statement is formulated as “Given the demographic data of a person, how can a mail-order company efficiently acquire new customers”.

Given this statement, we can conclude that we have to compare the existing customer data and the general population data in someway to deduce a relationship between them. A manual way of doing this is to compare the statistics between the customers and the general population. For example, the mean and standard deviation of age can be compared to determine which age group is more likely to be a customer or the salaries can be compared to see what group of people fall into customers, etc.

But this analysis would give out many results which again have to be analysed to come up with a final strategy. This process will require a lot of time, and by the time this analysis completes, the competitor in the market will capture most of the population, and the company will be out of business. Today with the advent of Machine Learning (ML) techniques used in every domain, this problem can also be addressed with the help of ML algorithms.

## Data Description and Analysis

As explained earlier, the data that Arvato provided contains demographics of existing customers and general population data. Additionally, two extra files have been provided for supervised learning section, one for training and one for testing. In the end, the predictions on the test set were to be submitted to Kaggle competition. Also, two additional files were provided, which contain information about feature values and their description. These two files are beneficial as all the feature names were in German and in short forms. Let us look at the information regarding the dataset.

* General population — consists of demographic data for the general population in Germany corresponding to 891,211 people with each person having 366 features. (891211x366)
* Customers Data — consists of demographic data for existing customers for the mail-order company corresponding to 191,652 people each with 369 features. The three extra features were company-specific regarding how did the place the order and how much quantity the order was. (191652x366)
* Training data — consists of demographic data of 42,982 people with an additional column other than 366, indicating whether a person is a customer, to be used to train supervised learning models.
* Test data — consists of demographic data of 42,833 people with the same 366 features but no targets.
* Two additional files containing information about features

#### Data Cleaning
The data analysis started with replacing the misrecorded values to NaN values. These misrecorded values were determined using the information files given. For example, as per the description given in the Attribute information file the column ‘LP_STATS_FEIN’ needs to contain only values from ‘1–5’, but the data which is given contains ‘0’. This means that these values were recorded with errors and these values have to be treated as missing values.The attribute information file also contains information about what value corresponds to unknown values in some columns. This information was helpful in a way that all the misrepresented information can be converted to missing values.

{{<figure library="1" src="customer-segmentation-acquisition/miss_colwise.png" title="Columns with more than 30% missing values" numbered="true">}}


After cleaning the misrecorded values, the next step is to deal with the missing values itself. An analysis of percentage missing values per column is performed to determine how many columns have missing values and if they did contain how much percentage. Figure 1 shows the columns which have more than 30% missing values. After this analysis, a threshold of 30 was selected to drop the columns. Later an analysis on missing values row-wise had to be performed in order to remove observations that have missing features. Here a threshold of 50 missing features per observation was selected to drop rows. After this analysis, the resulting shapes were:
* General population - (737288x356)
* Customers data - (13426x356) (neglecting the additional customer specific features)

#### Feature Engineering
There were certain categorical features which were encoded with numeric values (in fact many, but only a few were addressed for simplicity). These features were coded with binary encoding. Also, some features were containing too much information, for example, information about financial status and age in one single column. Such kind of features was identified and was either recoded to contain broader information or divided into two columns to contain both features separately. Any feature that contained more than 20 categories was either dropped or reconstructed into something useful with the help of the Attribute information file.

Since many of the features contained categorical values dumped into a single column, this step has helped to simplify the data for the later steps. This step resulted in having 353 features.

#### Imputing Missing Values

Even after dropping columns and rows based on a certain threshold, we are left with data which still has missing values. This problem is addressed with the help of simple imputer, which fills in the missing data with some values which we can control. A general approach for numerical features would be to impute the missing values with the median or mean. But a more common approach for categorical features is to impute with most common values. Since data corresponds to population, so it is more sensitive to impute the missing values with the most common values.

Now, the data is clean and ready for modelling, one final step is to scale the data, i.e. to bring all the features to the same range. This is done using a standard scaler.


## Customer Segmentation using Unsupervised Learning

For cluster segmentation there are two steps to be performed.
* Dimensionality Reduction
* Clustering

#### Dimensionality Reduction

Although we have 353 features, not all of them will have variation, i.e. some features might be the same for all the people. We can go through all the features here to see how many unique values are there per feature to select the features which have the required variation. But a more systematic approach would be to perform a certain analysis before dropping any column. Hence, Principal Component Analysis (PCA) has been performed to analyse the explained variance of the PCA components. PCA applies a linear transformation on the data to form a new coordinate system such that the components in the new coordinate system represent the variation in the data.

{{< figure library="1" src="customer-segmentation-acquisition/pca.png" title="PCA Explained Variance plot" numbered="true">}}

This analysis will help us determine how many features have enough variance to explain the variation in the data. An explained variance plot is used to select the number of components, i.e. the number of dimensions in the reduced coordinate space. As seen from the above plot, almost 90% of the variance can be explained with the help of approximately 150 components. Now, after the PCA transformation, we are left with 150 PCA components, each made up of a linear combination between the main features.

#### Clustering

After the dimensionality reduction, the next step is to divide the general population and customer population into different segments. K-Means clustering algorithm has been chosen for this task. Since it is simple and is apt for this task since it measures the distance between two observations to assign a cluster. This algorithm will help us in separating the general population with the help of the reduced features into a specified number of clusters and use this cluster information to understand the similarities in the general population and customer data. The number of clusters is selected to be ‘8’ with the help of an elbow plot.

#### Cluster Analysis

 {{< figure library="1" src="customer-segmentation-acquisition/cluster_percentages.png" title="Cluster proportions" numbered="true">}}

The general population and the customer population have been clustered into segments. Figure 3 represents the proportions of the population coming into each cluster. The cluster distribution of the general population is uniform, meaning that the general population has been uniformly clustered into 8 segments. But the customer population seems to be coming from the clusters ‘0’, ‘3’, ‘4’ and ‘7’. We can further confirm this by taking the ratio of proportions of customers segments and general population segments, as shown in Figure 4.

 {{< figure library="1" src="customer-segmentation-acquisition/cluster_prop_ratio.png" title="Cluster proportions" numbered="true">}}

As seen in Figure 4, if the ratio of proportions is greater than one, that means this cluster has a greater number of customers in the existing population and has the potential to have more future customers. If the ratio is less than one, that means these clusters have the least possibility to have future customers.


A more detailed cluster analysis, which explains each cluster and corresponding components is also performed. It is documented in the [jupyter notebook](https://github.com/pranaymodukuru/Bertelsmann-Arvato-customer-segmentation/blob/master/Arvato%20Project%20Workbook.ipynb) used for this project. I am not explaining it here as this blog post is already too long.


## Customer Acquisition using Supervised Learning

After analysing the general population and customers data understanding which segments to concentrate on. We can further extend this analysis to make use of ML algorithms to take this decision too. Since we already have the customers data and general population data, we can combine them to form training data and train the ML models to make predictions about whether to approach a customer or not.

In this case, supervised learning is done with the given train and test data. AUROC score has been selected as the evaluation metric since the problem is a highly imbalanced classification. The baseline performance was set with a Logistic Regression model, which was further improved with the help of Tree-based ensemble models. The AdaboostClassifier and XGBoostClassifier were the final selected models, whose predictions were submitted to Kaggle to attain a position in top 30 percentile (on the date of submission) with only two submissions.

 {{< figure library="1" src="customer-segmentation-acquisition/kaggle_position.png" title="Kaggle Leaderboard" numbered="true">}}

## Conclusion

The general population and customer population have been compared and segmented using an Unsupervised learning algorithm. We were able to determine which clusters have more customers and which are potential clusters to have probable customers. We have also used supervised learning algorithms to predict a possible future customer based on demographic data.

The resulting analysis has produced good results to put me in the top 30 percentile in the competition Leader board. The top score in the Leaderboard is 0.81063, which is not far away from the score that I achieved (0.80027). There is scope for improvement in the data preparation steps.

A more comprehensive explanation of each step and the reasons behind choices of algorithms and metrics has been given in the [project report](https://github.com/pranaymodukuru/Bertelsmann-Arvato-customer-segmentation/blob/master/Report.pdf) and all the steps are documented in this [notebook](https://github.com/pranaymodukuru/Bertelsmann-Arvato-customer-segmentation/blob/master/Arvato%20Project%20Workbook.ipynb).

Finally, I would like to thank Arvato Financial Solutions and Udacity for providing this wonderful opportunity to work with real-world data. This helped me gain valuable experience and helped me use and improve my skills.


## References
1. https://www.business2community.com/customer-experience/4-types-of-customer-segmentation-all-marketers-should-know-02120397
2. https://blog.alexa.com/types-of-market-segmentation/
3. https://clevertap.com/blog/customer-segmentation-examples-for-better-mobile-marketing/
4. https://www.datanovia.com/en/lessons/determining-the-optimal-number-of-clusters-3-must-know-methods/
5. https://towardsdatascience.com/what-metrics-should-we-use-on-imbalanced-data-set-precision-recall-roc-e2e79252aeba
