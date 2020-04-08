---
title: Concrete Compressive Strength Prediction
summary: Predicting compressive strength of Concrete using Machine Learning.
tags:
- Machine Learning
- Industry 4.0
- Civil Engineering
date: "2020-03-09"

# Optional external URL for project (replaces project detail page).
external_link: ""

image:
  caption:
  # focal_point: Smart

links:
- icon: twitter
  icon_pack: fab
  name: Follow
  url: https://twitter.com/pranaymns
- icon: linkedin
  icon_pack: fab
  name: Connect
  url: https://www.linkedin.com/in/pranaymodukuru/

url_code: "https://github.com/pranaymodukuru/Concrete-compressive-strength/blob/master/ConcreteCompressiveStrengthPrediction.ipynb"
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
# slides: example
---
Photo by Ricardo [Gomez Angel](https://unsplash.com/@ripato) on [Unsplash](https://unsplash.com/)

## Concrete Compressive Strength Prediction using Machine Learning

Concrete is one of the most important materials in Civil Engineering. Knowing the compressive strength of concrete is very important when constructing a building or a bridge. The Compressive Strength of Concrete is a highly nonlinear function of ingredients used in making it and their characteristics. Thus, using Machine Learning to predict the Strength could be useful in generating a combination of ingredients which result in high Strength.

Please read this [blog](https://towardsdatascience.com/concrete-compressive-strength-prediction-using-machine-learning-4a531b3c43f3) for more explanation. Also leave some claps to show appreciation!

Please refer [*ConcreteCompressiveStrengthPrediction.ipynb*](https://github.com/pranaymodukuru/Concrete-compressive-strength/blob/master/ConcreteCompressiveStrengthPrediction.ipynb) for code.

## 1. Problem Statement
Predicting Compressive Strength of Concrete given its age and quantitative measurements of ingredients.

## 2. Data Description

Data is obtained from UCI Machine Learning Repository.
https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength

* Number of instances - 1030
* Number of Attributes - 9
  * Attribute breakdown - 8 quantitative inputs, 1 quantitative output

#### Attribute information
##### Inputs
* Cement
* Blast Furnace Slag
* Fly Ash
* Water
* Superplasticizer
* Coarse Aggregate
* Fine Aggregate

All above features measured in kg/$m^3$

* Age (in days)

##### Output
* Concrete Compressive Strength (Mpa)

## 3. Modelling and Evaluation

* Algorithms used
  * Linear regression
  * Lasso regression
  * Ridge regression
  * Decision Trees
  * Random Forests

* Metric - Since the target variable is a continuous variable, regression evaluation metric RMSE (Root Mean Squared Error) and R2 Score (Coefficient of Determination) have been used.

## 4. Results

#### Feature correlation
{{< figure library="1" src="concrete-compressive-strength-imgs/pearson_coeff.png" >}}

#### Feature importance
{{< figure library="1" src="concrete-compressive-strength-imgs/tree_feat_imps.png" >}}


#### Final Comparison
{{< figure library="1" src="concrete-compressive-strength-imgs/final.png" >}}


## 5. References
1. https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
