# ML4MSD - Homework 2
*Created By*: Prof. Peter Schindler<br>
*Due*: Monday, 10/6/2025, 11:59 pm

## Overview

In the second homework, we will focus on working with data utilizing Pandas and cover some ML basics similar to what we covered during the lectures.

## Generative AI Usage and Setting up and Submitting Your Homework Submission

Same as homework 1. We will be working with the same data we used during the lectures: `Week3/band_gap_data.json`.

## Assignments

### Assignment 1

For the band gap data (`Week3/band_gap_data.json`), please filter for materials where `temperature_K>50`, `crystallinity=="Polycrystalline"`, and `exp_method=="Reflection"` (all three constraints need to be applied simultaneously). Do this three times: Once with a boolean mask, once with a `for`-loop while utilizing `iat` to access the information for each row, and once with `iloc`. Time all three methods and report the durations.

### Assignment 2

Let us do ridge regression on the same dataset (assuming data was loaded into variable `bgdata`). **Don't use Scikit-Learn** for any of the following steps.

1. Remove missing data for the relevant columns: `bgdata = bgdata.dropna(subset=['crystallinity', 'band_gap_type', 'exp_method', 'temperature_K', 'band_gap'])`
2. Remove all other columns except the ones specified in step 1.
3. Create one-hot features (that's a way of representing categorical variables as numerical vectors) for columns `'crystallinity'`, `'band_gap_type'`, and `'exp_method'`. For example, for the crystallinity column, this can be done with `crys_dummies = pd.get_dummies(bgdata['crystallinity'], prefix='crys')`. 
4. Plot the Pearson correlation matrix for all features (i.e., the ones created in step 3, plus `temperature_K`) and the target label `band_gap`. (this can easily be done with Pandas `bgdata.corr(method='pearson')` functionality). Based on this matrix, what can you say about the features' correlations?
5. Now split the dataset into train and test sets (e.g., 80/20).
6. Loop through different regularization strengths alpha and train ridge regression on the training set. Log both the training set and test set mean squared errors (MSEs) in a list. You can reuse the functions I used during the lecture demonstrations.
7. Finally, create a parity plot for the optimal alpha (true vs. predicted `band_gap`). Also create a MSE vs. alpha plot for the training and test set errors. Please, submit the final Figures in your GitHub submission as png or pdf files. 
8. Based on these plots and metrics, how well does ridge regression perform and what may be a bottleneck here?

### Assignment 3

Go through the following chapters of *"Machine Learning Refined"* ([free link](https://github.com/neonwatty/machine-learning-refined)): 5.6, 6.5-6.10, and 7.3-7.4. Also, please go through one or two of chapters 9, 10, 11, or 14 (based on your interest). Briefly summarize what you learned in each of these chapters and highlight what you found useful that is in the book but was not covered during the main lecture.

## Evaluation

I will grade your homework submission based on how I perceive your effort:
- 0%: Very little or no effort
- 50%: Basic/average effort (tried to make an effort)
- 100%: Effort is evident
