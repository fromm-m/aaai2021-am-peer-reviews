import argparse
import openreview
import json
import os
from collections import defaultdict


def download(arg):
    '''
    Main function for downloading conference submissions, reviews, and meta
    reviews, and organize them by forum ID
    A unique identifier for each paper; as in "discussion forum"
    :return: conference metadata
    '''

    # Getting Submissions
    submissions = openreview.tools.iterget_notes(
        client=client, invitation=arg.submission_invitation)
    submissions_by_forum = {n.forum: n for n in submissions}

    # Getting reviews
    reviews = openreview.tools.iterget_notes(
        client=client, invitation=arg.review_invitation)
    reviews_by_forum = defaultdict(list)
    for review in reviews:
        reviews_by_forum[review.forum].append(review)

    # Getting official comments
    comments = openreview.tools.iterget_notes(
        client=client, invitation=arg.comment_invitation)
    comments_by_forum = defaultdict(list)
    for comment in comments:
        comments_by_forum[comment.forum].append(comment)

    # There are no "decision notes" for ICLR, instead, decisions are taken
    # directly form Meta Reviews
    meta_reviews = openreview.tools.iterget_notes(
        client=client, invitation=arg.metareview_invitation)
    meta_reviews_by_forum = {n.forum: n for n in meta_reviews}
    if client.get_notes(invitation=arg.metareview_invitation) == []:
        MetaReviewFlag = False
    else:
        MetaReviewFlag = True

    # Getting decision if conference is neuroai19
    if arg.invitation == "NeurIPS.cc/2019/Workshop/Neuro_AI" or arg.invitation == "graphicsinterface.org/Graphics_Interface/2020/Conference":  # noqa
        all_decision_notes = openreview.tools.iterget_notes(
            client, invitation=arg.invitation + '/Paper.*/-/Decision')
        decisions_by_forum = defaultdict(list)
        for decision in all_decision_notes:
            decisions_by_forum[decision.forum].append(decision)

    folder = os.path.exists(arg.outdir)
    if not folder:
        os.makedirs(arg.outdir)
    os.chdir(arg.outdir)

    # For every paper (forum), get the review ratings, the decisions, and
    # the paper's content
    data = []
    for forum in submissions_by_forum:
        print("processing")
        # precessing submission info
        submission_cdate = submissions_by_forum[forum].cdate
        submission_tcdate = submissions_by_forum[forum].tcdate
        submission_tmdate = submissions_by_forum[forum].tmdate
        submission_ddate = submissions_by_forum[forum].ddate
        submission_content = submissions_by_forum[forum].content
        submission_url = 'https://openreview.net/forum?id=' + \
            submissions_by_forum[forum].id
        # str_to_process = submissions_by_forum[forum].content['_bibtex']
        # strlist = str_to_process.split('url={')
        # submission_url = strlist[1].split('}')[0]

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
        for id in review_id:
            review_url.append(submission_url + '&noteId=' + id)

        # processing official comment info
        forum_comments = comments_by_forum[forum]
        comment_cdate = [n.cdate for n in forum_comments]
        comment_tcdate = [n.tcdate for n in forum_comments]
        comment_tmdate = [n.tmdate for n in forum_comments]
        comment_readers = [n.readers for n in forum_comments]
        comment_writers = [n.writers for n in forum_comments]
        comment_reply_content = [n.details for n in forum_comments]
        comment_content = [n.content for n in forum_comments]
        comment_id = [n.id for n in forum_comments]
        comment_replyto = [n.replyto for n in forum_comments]
        comment_url = []
        for id in comment_id:
            comment_url.append(submission_url + '&noteId=' + id)

        # processing meta_review info
        if MetaReviewFlag:
            if forum not in meta_reviews_by_forum:
                meta_review_cdate = None
                meta_review_tcdate = None
                meta_review_tmdate = None
                meta_review_ddate = None
                meta_review_title = None
                meta_review_metareview = None
                meta_review_readers = None
                meta_review_writers = None
                meta_review_reply_count = None
                decision = None
                meta_review_url = None
            else:
                forum_meta_review = meta_reviews_by_forum[forum]
                meta_review_cdate = forum_meta_review.cdate
                meta_review_tcdate = forum_meta_review.tcdate
                meta_review_tmdate = forum_meta_review.tmdate
                meta_review_ddate = forum_meta_review.ddate
                meta_review_title = forum_meta_review.content['title']
                if 'Decision' in arg.metareview_invitation:
                    meta_review_metareview = forum_meta_review.content['comment']  # noqa
                elif 'metareview' not in forum_meta_review.content:
                    meta_review_metareview = forum_meta_review.content['metaReview']  # noqa
                else:
                    meta_review_metareview = forum_meta_review.content['metareview']  # noqa

                meta_review_readers = forum_meta_review.readers
                meta_review_writers = forum_meta_review.writers
                meta_review_reply_count = forum_meta_review.details
                if 'midl20' in arg.conference_name:
                    if 'recommendation_for_accepted_papers' not in forum_meta_review.content:  # noqa
                        decision = 'reject'
                    else:
                        decision = 'accept'
                elif 'Decision' in arg.metareview_invitation:
                    decision = forum_meta_review.content['decision']
                else:
                    decision = forum_meta_review.content['recommendation']
                meta_review_url = [submission_url +
                                   '&noteId=' + forum_meta_review.id]
        else:
            meta_review_cdate = None
            meta_review_tcdate = None
            meta_review_tmdate = None
            meta_review_ddate = None
            meta_review_title = None
            meta_review_metareview = None
            meta_review_readers = None
            meta_review_writers = None
            meta_review_reply_count = None
            # decision = None
            meta_review_url = None

            if arg.invitation == "NeurIPS.cc/2019/Workshop/Neuro_AI":
                decision = decisions_by_forum[forum][0].to_json()['content']['decision']  # noqa
            elif arg.invitation == "graphicsinterface.org/Graphics_Interface/2020/Conference":  # noqa
                try:
                    decision = decisions_by_forum[forum][0].to_json()['content']['decision']  # noqa
                except Exception:
                    decision = "Reject"
            else:
                decision = None

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
            'meta_review_readers': meta_review_readers,
            'meta_review_writers': meta_review_writers,
            'meta_review_reply_count': meta_review_reply_count,
            'meta_review_url': meta_review_url,
            'decision': decision,
        }
        data.append(forum_metadata)

        print("writing metadata to file...")

        cwd = os.getcwd()
        # cwd = os.path.abspath(os.curdir)
        file_name = arg.conference_name
        if not os.path.exists(cwd + '/' + arg.conference_name):
            os.makedirs(cwd + '/' + arg.conference_name)
        with open(cwd + '/' + arg.conference_name + '/' + file_name + '_' + str(forum) + '.json', 'w') as file_handle:  # noqa
            file_handle.write(json.dumps(forum_metadata))


if __name__ == '__main__':
    client = openreview.Client(baseurl='https://openreview.net')

    parser = argparse.ArgumentParser()
    parser.add_argument('--submission_invitation')
    parser.add_argument('--review_invitation')
    parser.add_argument('--comment_invitation')
    parser.add_argument('--metareview_invitation')
    parser.add_argument('--invitation')
    parser.add_argument('--outdir')
    parser.add_argument('--conference_name')
    args = parser.parse_args()
    download(args)
