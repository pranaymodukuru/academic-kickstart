---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Concrete Compressive Strength Prediction using Machine Learning"
subtitle: ""
summary: "The purpose of this post is to demonstrate the use of Machine learning as a tool for Civil Engineering"
authors: [Pranay Modukuru]
tags: [Machine Learning, Data Analysis, Data Visualization, Industry 4.0]
categories: []
date: 2020-03-05T01:53:52+01:00
lastmod: 2020-03-05T01:53:52+01:00
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "Photo by Ricardo [Gomez Angel](https://unsplash.com/@ripato) on [Unsplash](https://unsplash.com/)"
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: ["concrete-compressive-strength"]
---

## Concrete Compressive Strength

The Compressive Strength of Concrete determines the quality of Concrete. This is generally determined by a standard crushing test on a concrete cylinder. This requires engineers to build small concrete cylinders with different combinations of raw materials and test these cylinders for strength variations with a change in each raw material. The recommended wait time for testing the cylinder is 28 days to ensure correct results. This consumes a lot of time and requires lot of labour to prepare different prototypes and test them. Also, this method is prone to human error and one small mistake can cause the wait time to drastically increase.

One way of reducing the wait time and reducing the amount of combinations to try is to make use of digital simulations, where we can provide information to the computer about what we know and the computer tries different combinations to predict the compressive strength. This way we can reduce the amount of combinations we can try physically and reduce the amount of time for experimentation. But, to design such software we have to know the relations between all the raw materials and how one material affects the strength. It is possible to derive mathematical equations and run simulations based on these equations, but we cannot expect the relations to be same in real-world. Also, these tests have been performed for many number of time now and we have enough real-world data that can be used for predictive modelling.

In this article, we are going to analyse [Concrete Compressive Strength](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength) dataset and build Machine Learning models to predict the compressive strength. This [notebook](https://github.com/pranaymodukuru/Concrete-compressive-strength/blob/master/ConcreteCompressiveStrengthPrediction.ipynb) containing all the code can be used in parallel.

### Dataset Description
The dataset consists of 1030 instances with 9 attributes and has no missing values. There are 8 input variables and 1 output variable. Seven input variables represents the amount of a raw material (measured in $kg/m^3$) and one represents Age (in Days). The target variable is Concrete Compressive Strength measured in ($MPa$ - Mega Pascal). We shall explore the data to see how input features are affecting compressive strength.

### Exploratory Data Analysis
The first step in a Data Science project is to understand the data and gain insights from the data before doing any modelling. This includes checking for any missing values, plotting the features with respect to the target variable, observing the distributions of all the features and so on. Lets import the data and
start analysing.

Lets check the correlations between the input features, this will give an idea about how each variable is affecting all other variables. This can be done by calculating Pearson correlations between the features as shown in the code below.

<!-- **Note** - Complete code used for generating plots (titles, axes labels, etc.) is not shown here for simplicity. The complete code can be viewed [here](https://github.com/pranaymodukuru/Concrete-compressive-strength/blob/master/ConcreteCompressiveStrengthPrediction.ipynb). -->

```python
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='Blues')
```

{{< figure library="1" src="concrete-compressive-strength-imgs/pearson_coeff.png" >}}

We can observe a high positive correlation between **compressive Strength** (CC_Strength) and **Cement**. this is true because strength concrete indeed increases with an increase in amount of cement used in preparing it. Also, **Age** and **Super Plasticizer** are other two factors influencing Compressive strength.

There are other strong correlations between the fetures,
* A strong negative correlation between **Super Plasticizer** and **Water**.
* positive correlations between **Super Plasticizer** and **Fly Ash**, **Fine Aggregate**.

These correlations are useful to understand the data in detail, as they give an idea about how a variable is affecting the other. We can further use a **pairplot** in seaborn to plot pair wise relations between all the features and distributions of features along the diagonal.

```python
sns.pairplot(data)
```

{{< figure library="1" src="concrete-compressive-strength-imgs/pairplot.png" >}}

The pair plot gives a visual representation of correlations between all the features.

We can plot scatter plots between **CC_Strength** and other features to see more complex relations.

##### CC_Strength vs (Cement, Age, Water)

```python
sns.scatterplot(y="CC_Strength", x="Cement", hue="Water",
                  size="Age", data=data, ax=ax, sizes=(50, 300))
```

{{< figure library="1" src="concrete-compressive-strength-imgs/scatter_1.png" >}}

The observations we can make from this plot,
* **Compressive strength increases as amount of cement increases**, as the dots move up when we move towards right on the x-axis.
* **Compressive strength increases with age** (as the size of dots represents the age), this not the case always but can be up to an extent.  
* **Cement with less age requires more cement for higher strength**, as the smaller dots are moving up when we move towards right on x-axis.
* **The older the cement is the more water it requires**, can be confirmed by observing the colour of the dots. Larger dots with dark colour indicate high age and more water.  
* **Concrete strength increases when less water is used** in preparing it, since the dots on the lower side (y-axis) are darker and the dots on higher end (y-axis) are brighter.

##### CC Strength vs (Fine aggregate, Super Plasticizer, Fly Ash)

```python
sns.scatterplot(y="CC_Strength", x="FineAggregate", hue="FlyAsh", size="Superplasticizer",
                data=data, ax=ax, sizes=(50, 300))
```

{{< figure library="1" src="concrete-compressive-strength-imgs/scatter_2.png" >}}

Observations,
* **Compressive strength decreases Fly ash increases**, as more darker dots are concentrated in the region representing low compressive strength.
* **Compressive strength increases with Super plasticizer**, since larger the dot the higher they are in the plot.

 We can visually understand 2D, 3D and max up to 4D plots (features represented by colour and size) as shown above, we can further use row wise and column wise plotting features by seaborn to do further analysis, but still we lack the ability to track all these correlations by ourselves. For this reason, we can turn to Machine Learning to capture these relations and give better insights into the problem.

### Data preprocessing

Before we fit machine learning models on the data, we need to split the data into train, test splits. The features can be rescaled to have a mean of zero and a standard deviation of 1 i.e. all the features fall into the same range.

```python
X = data.iloc[:,:-1]         # Features
y = data.iloc[:,-1]          # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### Model Building
After preparing the data, we can fit different models on the training data and compare their performance to choose the algorithm with good performance. As this is a regression problem, we can use RMSE (Root Mean Square Error) and $R^2$ score as evaluation metrics.

#### 1. Linear Regression
We will start with Linear Regression, since this is the go-to algorithm for any regression problem. The algorithm tries to form a linear relationship between the input features and the target variable i.e. it fits a straight line given by, $$y = W*X + b = \sum_{i=1}^{n} w_i * x_i + b$$ Where $w_i$ corresponds to the coefficient of feature $x_i$.

The magnitude of these coefficients can be further controlled by using regularization terms to the cost functions. Adding the sum of the magnitudes of the coefficients will result in the coefficients being close to zero, this variation of linear regression is called **Lasso** Regression. Adding the sum of squares of the coefficients to the cost function will make the coefficients be in the same range and this variation is called **Ridge** Regression. Both these variations help in reducing the model complexity and therefore reducing the chances of overfitting on the data.

```python
# Importing models
from sklearn.linear_model import LinearRegression, Lasso, Ridge

# Linear Regression
lr = LinearRegression()
# Lasso Regression
lasso = Lasso()
# Ridge Regression
ridge = Ridge()

# Fitting models on Training data
lr.fit(X_train, y_train)
lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)

# Making predictions on Test data
y_pred_lr = lr.predict(X_test)
y_pred_lasso = lasso.predict(X_test)
y_pred_ridge = ridge.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Model\t\t\t RMSE \t\t R2")
print("""LinearRegression \t {:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_lr)), r2_score(y_test, y_pred_lr)))
print("""LassoRegression \t {:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_lasso)), r2_score(y_test, y_pred_lasso)))
print("""RidgeRegression \t {:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_ridge)), r2_score(y_test, y_pred_ridge)))
```

###### Output

| Model			       |  RMSE  |	  R2   |
| ---------------- | ------ | ------ |
| LinearRegression |	10.29 |		0.57 |
| LassoRegression  |	10.68 |		0.54 |
| RidgeRegression  | 	10.29 |	 	0.57 |

There is not much difference between the performance with these three algorithms, we can plot the coefficients assigned by the three algorithms for the features with the following code.

```python
coeff_lr = lr.coef_
coeff_lasso = lasso.coef_
coeff_ridge = ridge.coef_

labels = req_col_names[:-1]

x = np.arange(len(labels))
width = 0.3

fig, ax = plt.subplots(figsize=(10,6))
rects1 = ax.bar(x - 2*(width/2), coeff_lr, width, label='LR')
rects2 = ax.bar(x, coeff_lasso, width, label='Lasso')
rects3 = ax.bar(x + 2*(width/2), coeff_ridge, width, label='Ridge')

ax.set_ylabel('Coefficient')
ax.set_xlabel('Features')
ax.set_title('Feature Coefficients')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.show()
```

{{< figure library="1" src="concrete-compressive-strength-imgs/lr_coeffs.png" >}}

As seen in the figure, Lasso regression pushes the coefficients towards zero and the coefficients with the normal Linear Regression and Ridge Regression are almost the same.

We can further see how the predictions are by plotting the true values and predicted values,

```python
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,4))

ax1.scatter(y_pred_lr, y_test, s=20)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax1.set_ylabel("True")
ax1.set_xlabel("Predicted")
ax1.set_title("Linear Regression")

ax2.scatter(y_pred_lasso, y_test, s=20)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax2.set_ylabel("True")
ax2.set_xlabel("Predicted")
ax2.set_title("Lasso Regression")

ax3.scatter(y_pred_ridge, y_test, s=20)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax3.set_ylabel("True")
ax3.set_xlabel("Predicted")
ax3.set_title("Ridge Regression")

fig.suptitle("True vs Predicted")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
```

{{< figure library="1" src="concrete-compressive-strength-imgs/lr_true_pred.png" >}}

If the predicted values and the target values are equal, then the points on the scatter plot will lie on the straight line. As we can see here, non of the model predicts the Compressive Strength correctly.

#### 2. Decision Trees

A Decision Tree Algorithm represents the data with a tree like structure, where each node represents a decision taken on a feature. This algorithm would give better performance in this case, since we have a lot of zeros in some of the input features as seen from their distributions in the pair plot above. This would help the decision trees build trees based on some conditions on features which can further improve performance.

```python
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()

dtr.fit(X_train, y_train)

y_pred_dtr = dtr.predict(X_test)

print("Model\t\t\t\t RMSE  \t\t R2")
print("""Decision Tree Regressor \t {:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_dtr)), r2_score(y_test, y_pred_dtr)))

plt.scatter(y_test, y_pred_dtr)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Decision Tree Regressor")
plt.show()
```

| Model			              |  RMSE  |  R2   |
| ----------------------- | ------ | ----- |
| Decision Tree Regressor |	 7.31  |	0.78 |

{{< figure library="1" src="concrete-compressive-strength-imgs/dtr_true_pred.png" >}}

The Root Mean Squared Error (RMSE) has come down from 10.29 to 7.31, so the Decision Tree Regressor has improved the performance by a significant amount. This can be observed in the plot as well as more points are closer to the line.

#### 3. Random Forests
Since Using a Decision Tree Regressor has improved our performance, we can further improve the performance by ensembling more trees. Random Forest Regressor trains randomly initialized trees with random subsets of data sampled from the training data, this will make our model more robust.

```python
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=100)

rfr.fit(X_train, y_train)

y_pred_rfr = rfr.predict(X_test)

print("Model\t\t\t\t RMSE  \t\t R2")
print("""Random Forest Regressor \t {:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_rfr)), r2_score(y_test, y_pred_rfr)))

plt.scatter(y_test, y_pred_rfr)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Random Forest Regressor")
plt.show()
```

| Model			              |  RMSE  |  R2   |
| ----------------------- | ------ | ----- |
| Random Forest Regressor |	 5.08  |	0.89 |

{{< figure library="1" src="concrete-compressive-strength-imgs/rfr_true_pred.png" >}}

The RMSE has further reduced by ensembling multiple trees. We can plot the feature importance's for tree based models. The feature importance's show how important a feature is for a model when making a prediction.

```python

feature_dtr = dtr.feature_importances_
feature_rfr = rfr.feature_importances_

labels = req_col_names[:-1]

x = np.arange(len(labels))
width = 0.3

fig, ax = plt.subplots(figsize=(10,6))
rects1 = ax.bar(x-(width/2), feature_dtr, width, label='Decision Tree')
rects2 = ax.bar(x+(width/2), feature_rfr, width, label='Random Forest')

ax.set_ylabel('Importance')
ax.set_xlabel('Features')
ax.set_title('Feature Importance')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend(loc="upper left", bbox_to_anchor=(1,1))

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()
```
{{< figure library="1" src="concrete-compressive-strength-imgs/tree_feat_imps.png" >}}

Cement and Age are treated as the most important features by tree based models. Fly ash, Coarse and Fine aggregates are least important factors when predicting the strength of Concrete.

#### Comparison

Finally, lets compare the results of all the algorithms.

```python

models = [lr, lasso, ridge, dtr, rfr]
names = ["Linear Regression", "Lasso Regression", "Ridge Regression",
         "Decision Tree Regressor", "Random Forest Regressor"]
rmses = []

for model in models:
    rmses.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

x = np.arange(len(names))
width = 0.3

fig, ax = plt.subplots(figsize=(10,7))
rects = ax.bar(x, rmses, width)
ax.set_ylabel('RMSE')
ax.set_xlabel('Models')
ax.set_title('RMSE with Different Algorithms')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45)
autolabel(rects)
fig.tight_layout()
plt.show()
```
{{< figure library="1" src="concrete-compressive-strength-imgs/final.png" >}}

### Conclusion
We have analysed the Compressive Strength Data and used Machine Learning to predict the Compressive Strength of Concrete. We have used Linear Regression and its variations, Decision Trees and Random Forests to make predictions and compared their performance. Random Forest Regressor has the lowest RMSE and is a good choice for this problem. Also, we can further improve the performance of the algorithm by tuning the hyperparameters by performing a grid search or random search.  

### References

1. I-Cheng Yeh, "[Modeling of strength of high performance concrete using artificial neural networks](https://www.sciencedirect.com/science/article/abs/pii/S0008884698001653)," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998).
2. Ahsanul Kabir, Md Monjurul Hasan, Khasro Miah, "[Strength Prediction Model for Concrete](https://www.researchgate.net/publication/258255660_Strength_Prediction_Model_for_Concrete)", ACEE Int. J. on Civil and Environmental Engineering, Vol. 2, No. 1, Aug 2013.
3. https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
