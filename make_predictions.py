from __future__ import division, print_function
import csv
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser
import random

def load_tsv(fn):
    #result = defaultdict(lambda : [])
    with open(fn, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        result = {field:[] for field in reader.fieldnames}
        for row in reader:
            for field in row:
                result[field].append(row[field])
    return result

def write_tsv(fn, list_of_dict):
    with open(fn, 'w') as f:
        fieldnames = list_of_dict[0].keys()
        writer = csv.DictWriter(f, fieldnames, delimiter='\t')
        writer.writeheader()
        for dict_ in list_of_dict:
            writer.writerow(dict_)


class CollaborativeFilter:
    def __init__(self, users, hotels, topk=None, verbosity=1):
        self.users_list = users
        self.hotels_list = hotels
        self.users_rind = {usr:ind for ind, usr in enumerate(users)}
        self.hotels_rind = {hot:ind for ind, hot in enumerate(hotels)}
        self.verbosity = verbosity
        self.topk = topk

    def build_user_hotel_mat(self, activity):
        # build user_hotel_mat
        # build user_hotel_map of each user to the hotels they clicked on
        if self.verbosity > 0:
            print("Building matrix for filtering")
        self.user_hotel_mat = np.zeros((len(self.users_rind),
                                       len(self.hotels_rind)))
        self.user_hotel_map = defaultdict(lambda : [])
        # assume activity is a dict structured exactly wrong
        for user, hotel in zip(activity['user'], activity['hotel']):
            self.user_hotel_map[user].append(hotel)
            # ignore multiple clicks
            self.user_hotel_mat[self.users_rind[user], self.hotels_rind[hotel]] = 1

    def build_left_out_map(self):
        self.left_out_map = {}
        for user in self.users_list:
            random.shuffle(self.user_hotel_map[user])
            self.left_out_map[user] = self.user_hotel_map[user][-1]

    def find_similar_users(self, user, user_hotel_mat, leave_out=None):
        # figure out for each user how well their clicks correlate with this user's clicks
        user_similarities = {}
        if leave_out is not None: # for evaluation purposes
            self.user_hotel_mat[self.users_rind[user],
                                self.hotels_rind[leave_out]] = 0
        for other_user in self.users_rind:
            if other_user == user:
                continue
            user_similarities[other_user] = np.corrcoef(self.user_hotel_mat[self.users_rind[user],:],
                                                        self.user_hotel_mat[self.users_rind[other_user],:])[0,1]
        if leave_out is not None: # I'm not proud of this, but copying the matrix just to make one change is slow
            self.user_hotel_mat[self.users_rind[user],
                                self.hotels_rind[leave_out]] = 1
        if self.topk is not None:
            user_similarities = {usr:user_similarities[usr] for usr in
                                 sorted(user_similarities.keys(), key=lambda x:
                                        -user_similarities[x])[:self.topk]}

        return user_similarities

    def predict_next(self, user, use_left_out):
        # first find a similar user
        if use_left_out:
            leave_out = self.left_out_map[user]
        else:
            leave_out = None
        similar_users = self.find_similar_users(user, self.user_hotel_mat, leave_out=leave_out)
        # so now we'll try to fill in the non-clicks by weighting each of the similar users by their similarity
        already_clicked_mask = [self.hotels_rind[hotel] for hotel in self.user_hotel_map[user] if (hotel != leave_out)]
        recommendations = np.sum([self.user_hotel_mat[self.users_rind[usr]]*similar_users[usr] for usr in similar_users], axis=0)
        recommendations[already_clicked_mask] = -np.inf
        return self.hotels_list[np.argmax(recommendations)]

    def predict(self, users=None, use_left_out=False):
        if users is None:
            users = self.users_rind.keys()
        predictions = {}
        for user in users:
            predictions[user] = self.predict_next(user, use_left_out)
            if self.verbosity > 1:
                print("User %s: prediction %s%s" % (user, predictions[user],
                                                    '' if not use_left_out else (", leave out: %s" % self.left_out_map[user])))
        return predictions

    def evaluate(self, predictions):
        correct = 0
        for user in predictions:
            if predictions[user] == self.left_out_map[user]:
                correct += (1 / len(predictions))
        return correct

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-v", "--verbosity", action='store', type=int,
                            default=1)
    arg_parser.add_argument("outfile", action='store', type=str)
    arg_parser.add_argument("-e", "--evaluate", action='store_true')
    arg_parser.add_argument("-k", "--topksim", action='store', type=int,
                            default=0)
    args = arg_parser.parse_args()

    # Assume that the users, activity, and hotels are in users.txt, hotels.txt,
    # and activity.txt respectively
    all_activity = load_tsv('activity.txt')
    all_hotels = load_tsv('hotels.txt')
    all_users = load_tsv('users.txt')

    if args.topksim <= 0:
        topk = None
    else:
        topk = args.topksim
    click_predictor = CollaborativeFilter(all_users['user'], all_hotels['hotel'], topk=topk, verbosity=args.verbosity)
    click_predictor.build_user_hotel_mat(all_activity)
    if args.verbosity > 0:
        print("Building predictions")
    predictions = click_predictor.predict()
    if args.evaluate:
        click_predictor.build_left_out_map()
        if args.verbosity > 0:
            print("Building predictions w/randomly held out clicks for evaluation")
        predictions_leftout = click_predictor.predict(use_left_out=True)
        correct = click_predictor.evaluate(predictions_leftout)
        print("Got %0.2f accuracy on predicting left out clicks" % correct)

    converted_list = []
    for user in predictions:
        converted_list.append({'user':user, 'hotel':predictions[user]})
    write_tsv(args.outfile, converted_list)
