import openreview
import json
from collections import defaultdict


def download(client, invitation):
    """
    Main function for downloading conference submissions, reviews, and meta
    reviews, and organize them by forum ID
    A unique identifier for each paper; as in "discussion forum"
    :param invitation: invitation ID, provided by conference
    :return: conference metadata
    """

    # TODO: modify invitation for different conference (current version
    # works for ICLR/2019)

    # Getting Submissions
    submissions = openreview.tools.iterget_notes(
        client=client, invitation=invitation + '/-/Blind_Submission')
    submissions_by_forum = {n.forum: n for n in submissions}
    # print("submissions_by_forum")
    # print(submissions_by_forum[2])

    # Getting reviews
    reviews = openreview.tools.iterget_notes(
        client=client, invitation=invitation + '/Paper.*/-/Official_Review')
    reviews_by_forum = defaultdict(list)
    for review in reviews:
        reviews_by_forum[review.forum].append(review)

    # Getting decision
    all_decision_notes = openreview.tools.iterget_notes(
        client, invitation=invitation + '/Paper.*/-/Decision')
    decisions_by_forum = defaultdict(list)
    for decision in all_decision_notes:
        decisions_by_forum[decision.forum].append(decision)
    # print(decisions_by_forum)

    # Getting official comments
    commentso = openreview.tools.iterget_notes(
        client=client, invitation=invitation + '/-/Paper.*/Official_Comment')
    commentso_by_forum = defaultdict(list)
    for commento in commentso:
        commentso_by_forum[commento.forum].append(commento)

    # There are no "decision notes" for ICLR, instead, decisions are taken
    # directly form Meta Reviews
    meta_reviews = openreview.tools.iterget_notes(
        client=client, invitation=invitation + '/Paper.*/-/Meta_Review')
    meta_reviews_by_forum = {n.forum: n for n in meta_reviews}
    if client.get_notes(invitation=invitation + '/Paper.*/-/Meta_Review') == []:  # noqa
        MetaReviewFlag = False
    else:
        MetaReviewFlag = True

    # For every paper (forum), get the review ratings, the decisions, and the
    # paper's content
    data = []
    # print("total length: " + str(len(submissions_by_forum)))
    index = 0
    for forum in submissions_by_forum:
        # print("processing ... iteration" + str(index))
        index += 1
        # precessing submission info
        submission_cdate = submissions_by_forum[forum].cdate
        submission_tcdate = submissions_by_forum[forum].tcdate
        submission_tmdate = submissions_by_forum[forum].tmdate
        submission_ddate = submissions_by_forum[forum].ddate
        submission_content = submissions_by_forum[forum].content
        try:
            str_to_process = submissions_by_forum[forum].content['_bibtex']
            strlist = str_to_process.split('url={')
            submission_url = strlist[1].split('}')[0]
        except Exception:
            submission_url = 'https://openreview.net/forum?id=' + \
                submissions_by_forum[forum].id
            # print("submission_url type 2 = " + str(submission_url))

        # processing reviews info
        forum_reviews = reviews_by_forum[forum]
        review_cdate = [n.cdate for n in forum_reviews]
        review_tcdate = [n.tcdate for n in forum_reviews]
        review_tmdate = [n.tmdate for n in forum_reviews]
        review_readers = [n.readers for n in forum_reviews]
        review_writers = [n.writers for n in forum_reviews]
        review_reply_count = [n.details for n in forum_reviews]
        review_content = [n.content for n in forum_reviews]
        review_id = [n.id for n in forum_reviews]
        review_replyto = [n.replyto for n in forum_reviews]
        review_url = []
        try:
            for id in review_id:
                review_url.append(submission_url + '&noteId=' + id)
        except Exception:
            # print("NO submission_url in review")
            for id in review_id:
                review_url.append('&noteId=' + id)

        # processing official comment info
        forum_commentso = commentso_by_forum[forum]
        comment_cdate = [n.cdate for n in forum_commentso]
        comment_tcdate = [n.tcdate for n in forum_commentso]
        comment_tmdate = [n.tmdate for n in forum_commentso]
        comment_readers = [n.readers for n in forum_commentso]
        comment_writers = [n.writers for n in forum_commentso]
        comment_reply_content = [n.details for n in forum_commentso]
        comment_content = [n.content for n in forum_commentso]
        comment_id = [n.id for n in forum_commentso]
        comment_replyto = [n.replyto for n in forum_commentso]
        comment_url = []
        try:
            for id in comment_id:
                comment_url.append(submission_url + '&noteId=' + id)
        except Exception:
            print("NO submission_url in commento")

        # processing meta_review info
        if MetaReviewFlag:
            forum_meta_review = meta_reviews_by_forum[forum]
            print(forum_meta_review)
            meta_review_cdate = forum_meta_review.cdate
            meta_review_tcdate = forum_meta_review.tcdate
            meta_review_tmdate = forum_meta_review.tmdate
            meta_review_ddate = forum_meta_review.ddate
            meta_review_title = forum_meta_review.content['title']
            meta_review_metareview = forum_meta_review.content['metareview']
            meta_review_confidence = forum_meta_review.content['confidence']
            meta_review_readers = forum_meta_review.readers
            meta_review_writers = forum_meta_review.writers
            meta_review_reply_count = forum_meta_review.details
            decision = decisions_by_forum[forum][0].to_json()[
                'content']['decision']
            meta_review_url = [submission_url +
                               '&noteId=' + forum_meta_review.id]
        else:
            meta_review_cdate = None
            meta_review_tcdate = None
            meta_review_tmdate = None
            meta_review_ddate = None
            meta_review_title = None
            meta_review_metareview = None
            meta_review_confidence = None
            meta_review_readers = None
            meta_review_writers = None
            meta_review_reply_count = None
            decision = decisions_by_forum[forum][0].to_json()[
                'content']['decision']
            meta_review_url = None

        forum_metadata = {
            'forum': forum,
            'submission_url': submission_url,
            'submission_content': submission_content,
            'submission_cdate': submission_cdate,
            'submission_tcdate': submission_tcdate,
            'submission_tmdate': submission_tmdate,
            'submission_ddate': submission_ddate,

            'review_id': review_id,
            'review_url': review_url,
            'review_cdate': review_cdate,
            'review_tcdate': review_tcdate,

            'review_tmdate': review_tmdate,
            'review_readers': review_readers,
            'review_writers': review_writers,
            'review_reply_count': review_reply_count,
            'review_replyto': review_replyto,
            'review_content': review_content,

            'comment_id': comment_id,
            'comment_cdate': comment_cdate,
            'comment_tcdate': comment_tcdate,
            'comment_tmdate': comment_tmdate,
            'comment_readers': comment_readers,
            'comment_writers': comment_writers,
            'comment_reply_content': comment_reply_content,
            'comment_content': comment_content,
            'comment_replyto': comment_replyto,
            'comment_url': comment_url,

            'meta_review_cdate': meta_review_cdate,
            'meta_review_tcdate': meta_review_tcdate,
            'meta_review_tmdate': meta_review_tmdate,
            'meta_review_ddate ': meta_review_ddate,
            'meta_review_title': meta_review_title,
            'meta_review_metareview': meta_review_metareview,
            'meta_review_confidence': meta_review_confidence,
            'meta_review_readers': meta_review_readers,
            'meta_review_writers': meta_review_writers,
            'meta_review_reply_count': meta_review_reply_count,
            'meta_review_url': meta_review_url,
            'decision': decision,
        }
        data.append(forum_metadata)

        print("writing metadata to file...")

        # file_name = '_'.join(invitation.split('/'))
        file_name = "neuroai19"
        with open("1_scrape_data/data/neuroai19/" + file_name + '_' + str(forum) + '.json', 'w') as file_handle:  # noqa
            file_handle.write(json.dumps(forum_metadata))
        # local test
        # break


if __name__ == '__main__':
    client = openreview.Client(baseurl='https://openreview.net')
    download(client=client, invitation='NeurIPS.cc/2019/Workshop/Neuro_AI')
