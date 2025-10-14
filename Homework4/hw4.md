# ML4MSD - Homework 3
*Created By*: Prof. Peter Schindler<br>
*Due*: Monday, 10/20/2025, 11:59 pm

## Overview

In the fourth homework, we will put the full ML pipeline into practice on real materials science datasets.

## Generative AI Usage and Setting up and Submitting Your Homework Submission

Same as homework 1.

## Assignments

### Assignment 1

Finalize Exercise 11.1 in `Week6/11_Full_MatSci-ML_Pipeline.ipynb`. Please, fill in the details and results in this [Google Sheet](https://docs.google.com/spreadsheets/d/1xbT4lRMYQGrhBQGFczFD5wscrriwWQGf82Gv9AnDkbA/).

### Assignment 2

Go through an entire ML pipeline again, but this time use the Materials Project API to query a dataset that contains around 1-3k data points. Make the query more and more selective until only a few thousand entries are returned (e.g., by limiting the number of unique elements in the unit cell, limiting the band gap range, limiting the energy above hull value, etc.). Select one of the many possible target properties that the Materials Project has to offer (see homework 3, assignment 2, on how to get the information of all possible fields that can be requested and/or queried). 

Featurize the dataset you queried with a structural featurizer from `DScribe` and go through the remaining ML pipeline (similar to assignment 1). Summarize the performance metrics on the test set and upload the parity plots as a PNG files to the repository. 

### Assignment 3

Please, go through the first 6 pages of this paper: `Resources/Papers/Musil_Chemical_Review_2021_Review-of-Descriptors.pdf`. We discussed these aspects during the lectures, but please summarize any key insights that you may have missed during the lecture but understand better after reading the paper.

## Evaluation

I will grade your homework submission based on how I perceive your effort:
- 0%: Very little or no effort
- 50%: Basic/average effort (tried to make an effort)
- 100%: Effort is evident