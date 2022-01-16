# Charity classifier

## What is this work about?
When registering with the Commission, charities must declare their sector of activity. This is a tick box with 16 options, due to the nature of the sector, most organsiation tick at least 2, and some as many as all 16 boxes. 

This notebook is about trying to classify organisations based on their charitable object: given the wording of their object, it returns the most likely sector of activity.

## Why does it matter?
The simple question of _How much charitable expenditure goes towards Education and Training vs. Health?_ is hard to answer because a lot of charities fall into both categories. This tool helps defining the most likely main charitable activity to avoid double counting.

## Some elements of the method used
This notebook uses `tidymodels` to train 4 machine learning models and create a voting classifier.

I used the charities that had only one category as the training and test set and validated the model on charities that have ticked 2 boxes. I considered that the model was successful if it picked up one of the 2 boxes.

## What data was used
The data used is the data from the charity [commission register](https://register-of-charities.charitycommission.gov.uk/charity-search)

I'm restricting the analysis only to service delivery charities and only the 'what' of their activity classification.

## Conclusions
The model performs quite well with over 80% accuracy. Some categories are very easy to pick up, other are almost never accurately predicted though.
