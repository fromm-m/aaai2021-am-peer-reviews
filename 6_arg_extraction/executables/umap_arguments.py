import numpy as np
import pandas as pd
import umap.plot

confidence_file = 'prediction/confidence.csv'
cls_token_file = 'prediction/representation.npy'
percentage = 0.95
#################################################################


def get_complete_id(df):
    conference = df['conference'].copy()
    paper_id = df['paper_id'].copy().apply(str)
    review_id = df['review_id'].copy().apply(str)
    sentence_id = df['sentence_id'].copy().apply(str)
    df['complete_id'] = conference.str.cat(
        [paper_id, review_id, sentence_id], sep = "_")
    return df


if __name__ == '__main__':
    df = pd.read_csv(confidence_file, sep = ',')
    print(df.head())
    df_complete_id = get_complete_id(df)
    print(df_complete_id)
    highest_confidence = df.loc[df['label'] >= percentage]
    print(highest_confidence)
    print(highest_confidence['complete_id'])
    x = np.load(cls_token_file, allow_pickle = True).item()

    representations = []
    for row in highest_confidence['complete_id']:
        cls_representation = x[row]
        representations.append(cls_representation)
    print(representations)

    embedding = umap.UMAP().fit(representations)
    plot = umap.plot.points(embedding, labels=highest_confidence['conference'])
    #umap.plot.show(plot)
    umap.plot.plt.savefig('umap_plot.svg', format='svg', dpi=2400)

