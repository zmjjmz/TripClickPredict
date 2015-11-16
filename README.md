# TripClickPredict
Repository for the TripAdvisor interview assignment

# Setup

- Clone this repository, enter the directory
- Run `# pip install -r requirements.txt'`
- Ensure that `activity.txt`, `hotels.txt`, and `users.txt` are in the directory.

# Code

There's two major pieces of code

1. An IPython notebook (`Exploratory.ipynb`) which shows the (very rough) prototyping process I went through and some of the considerations that went into deciding on the final method.
2. `make_predictions.py` which assumes the presence and names of the given dataset to be the original names, and will take as argument a file to write its (tab-separated with a header) predictions of the next click for each user. It also takes an optional verbosity argument and a switch on whether or not to run the evaluation detailed below, as well as a value _k_ which controls how many similar users are considered.

It is worth noting that getting the predictions the full evaluation will take about 15 minutes (~0.2 seconds per prediction), and if you run it with the evaluate switch it will have to rerun the predictions (as the similarities will change) taking another 15 minutes.

## Data

The data given consists of a set of users, hotels, and a list of 'clicks' describing a user clicking on a hotel. These clicks are not necessarily unique.

### Users

Each user (~4500 of them) has a gender and continent that describes them. On average, there's each user makes about 4 clicks but the mode is very strongly 2 clicks, which is unsurprising.

### Hotels

Each hotel (there are 66 hotels) has a single rating, with the minimal rating being 2 (although theoretically on a 1-5 scale) and the average being around 3.5 stars but the mode being 4 stars. Interestingly there's not a very strong correlation between star rating and number of clicks received.

### Activity

The activity is just a list of users and the corresponding hotel they clicked on. The order of these clicks is not given, which limits the possible solutions that can be explored. Additionally, it is not known what hotels the user saw prior to clicking on a hotel, which also limits the possible solutions.

## Model

Due to a variety of considerations outlined in the IPython Notebook, I decided to go with a raw collaborative filtering approach, ignoring the user and hotel attributes. While my gut feeling is that the user attributes aren't predictive of which hotel a user will click on, the hotel's attributes are and if I were to continue working on this I would try to find a way to incorporate it.

### Collaborative Filtering

The basic idea behind collaborative filtering in this scenario is to make a matrix _M_ such that for a user _i_ and a hotel _j_, _M_(_i_,_j_) = 1 if _i_ clicked on _j_ and 0 otherwise. I ignore the multiple clicks aspect, as only one user in this dataset made a second click on a hotel. 

In order to predict what hotel a user will click on next, I then simply find its closest users (i.e. by the correlation between their clicks) and sum up their clicks weighted by the correlation coefficient. I then use the maximum value of this weighted sum for my hotel prediction (of course ignoring those hotels that the user already clicked on).

This method is slow, but could be reasonably scalable with appropriate lower-bounding techniques. It does however ignore potentially useful attributes of the hotels and users.

## Results

Evaluating this approach is not straightforward without knowing the order of the clicks given. I decided to randomly hold out a hotel from each user on the assumption that the other clicks could predict it, and found that the results were pretty awful, though significantly better than random -- getting around 12% accuracy.

## Conclusion and Future Work

There are a lot of ways that this could be improved. One consideration is that collaborative filtering usually requires some more normalization to work, but I decided against normalizing hotel click frequency because it didn't seem like the correct intuition -- i.e. it's not unreasonable to predict that a user would click on a popular hotel. This did however lead to the collaborative filter predicting popular hotels very frequently (see notebook).

The other thing that would make sense to take into account is of course the hotel ratings -- even if another similar user clicks on a lower rated hotel, it is reasonable to assume that we should discount the weight of this hotel in predicting the target user's next click.

The implementation of collaborative filtering that I've made is incredibly slow, and of course could be made faster.
Due to timing I never really evaluated how cutting off similar users after some _k_ affects the score (i.e. I never optimized that hyperparameter), which would be doable if this was implemented more efficiently.

