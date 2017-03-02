# Yelp Dataset Challenge

This project was done as a part of the coursework for Information Retrieval (Search). The task has two components,

## 1. Restraunt classification using Reviews
The task puts a little twist in the normal ways reviews are written. If you were a say AirBnB or Housing.com, there are predefined listings to hotels. However is there a possible way to generate additional categories for the listings through their reviews?
Python was used to built a parallel data pipeline, uses NLTK for stemming, stopword removal and further feature generation.
Then a Naive Bayes is used for the #### MultiLabel Classification

A detailed analysis of multiple stemmers, feature generature generation mechanisms(Unigram, Bigram, Trigram) etc and the trade-off between precision and recall for the threshold.

## 2. Restraunt Genre Recommendation

Identify the pattern of visits to the restraunts through user reviews and recommend new restraunt to the users.
Colloborative filtering was used to model the data and recommend new restraunts to the user.
