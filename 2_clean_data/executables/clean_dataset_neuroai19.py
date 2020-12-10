import pandas as pd
import json
import os
import numpy as np

from cleaning_helper import clean_data


# def a func that accepts a directory containing the json files as an input
# and converts the same into a csv files for the papers and the reviews.
def dataset_prep(source_path, destination_path):
    papers = []
    reviews = []
    paper_id = 0
    no_of_reviews = 0
    # Reading the json as a dict
    for file_name in [file for file in os.listdir(source_path)
                      if file.endswith('.json')]:
        with open(source_path + file_name) as json_data:
            paper_id += 1
            data = json.load(json_data)
            papers.append(paper_id)
            papers.append(clean_data(data["submission_content"]["title"]))
            # keywords is a list & is therefore not cleaned
            papers.append(data["submission_content"]["keywords"])
            papers.append(clean_data(data["submission_content"]["abstract"]))
            papers.append(clean_data(data["meta_review_metareview"]))
            paper_decision = data["decision"]
            for x in range(len(data["review_replyto"])):
                reviews.append("neuroai19_" + str(paper_id) + "_" + str(x + 1))
                try:
                    reviews.append(clean_data(data["review_content"][x]["importance_comment"]) + " " + clean_data(data["review_content"][x]["rigor_comment"]) + " " + clean_data(data["review_content"][x]["clarity_comment"]) + " " + clean_data(data["review_content"][x]["intersection_comment"]) + " " + clean_data(data["review_content"][x]["comment"]))  # noqa
                except KeyError:
                    reviews.append(clean_data(data["review_content"][x]["importance_comment"]) + " " + clean_data(data["review_content"][x]["rigor_comment"]) + " " + clean_data(data["review_content"][x]["clarity_comment"]) + " " + clean_data(data["review_content"][x]["intersection_comment"]))  # noqa
                neuroai_rating = int((data["review_content"][x]["evaluation"]).split(":")[0])  # noqa
                # neuroai evaluations of 4 & 5 get mapped to a rating of 4
                if neuroai_rating == 5:
                    neuroai_rating = 4
                reviews.append(neuroai_rating)
                if paper_decision == "Reject":
                    reviews.append(0)
                elif paper_decision == "Accept (Poster)":
                    reviews.append(1)
                elif paper_decision == "Accept (Oral)":
                    reviews.append(1)
                else:
                    print("Unexpected case, decision for paper with ID: "
                          + str(paper_id) + " is not defined properly")
                    reviews.append(0)
            if len(data["review_content"]) != 0:
                no_of_reviews = no_of_reviews + x + 1

        # converting to pandas Dataframes for easier processing
        paper_col_names = ["paper_id", "title", "keywords", "abstract",
                           "meta_review"]
        papers_ = np.array(papers).reshape(paper_id, len(paper_col_names))
        papers_df = pd.DataFrame(papers_, columns=paper_col_names)

        reviews_col_names = ["review_id", "review", "rating", "decision"]
        reviews_ = np.array(reviews).reshape(no_of_reviews, len(reviews_col_names))  # noqa
        reviews_df = pd.DataFrame(reviews_, columns=reviews_col_names)

        # defining boundaries around the text categories
        papers_df['title'] = '\"' + papers_df['title'].astype(str) + '\"'
        papers_df['abstract'] = '\"' + papers_df['abstract'].astype(str) + '\"'
        papers_df['meta_review'] = '\"' + papers_df['meta_review'].astype(str) + '\"'  # noqa
        reviews_df['review'] = '\"' + reviews_df['review'].astype(str) + '\"'

    # renaming the filenames with the conference name
    papers_filename = os.path.join(os.path.dirname(__file__),
                                   destination_path + "neuroai19_papers.csv")
    reviews_filename = os.path.join(os.path.dirname(__file__),
                                    destination_path + "neuroai19_reviews.csv")
    papers_df.to_csv(papers_filename, sep=',', index=False)
    reviews_df.to_csv(reviews_filename, sep=',', index=False)


source_path = '../../1_scrape_data/data/neuroai19/'
destination_path = '../data/'
dataset_prep(source_path, destination_path)
