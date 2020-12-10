import pandas as pd
from src.alpha import Unit, Annotation
import copy


class Agreement:
    def __init__(self, lengths, annos):
        self.annotations = {}  # review_id: [Annotation, length]

        data = pd.read_csv(lengths, usecols=['review',
                                             'length']).values.tolist()

        for line in data:
            review, length = line[0], line[1]
            try:
                annotation = Annotation.read_from_file(annos + review +
                                                       '_anno.csv')
                self.annotations[review] = (annotation, length)
            except FileNotFoundError:
                print('file not found')
        for k, v in self.annotations.items():
            if len(v[0].tags) >= 3:
                print('rio original: ', k, *v[0].tags)

    def alpha_general(self, samples, output):
        """
        Computes alpha per review and an average over all reviews;
        Create a csv file containing alpha scores and review information.
        :param samples: File of review information.
        :param output: Path of output file.
        """

        scores = []
        count_cu = 0  # Count of reviews with valid cuAlpha
        sum1 = 0  # cuAlpha
        sum2 = 0  # uAlpha
        sum3 = 0  # cuAlphaNON
        sum4 = 0  # biAlpha

        for k, v in self.annotations.items():
            cu_alpha = v[0].alpha_score()
            u_alpha = v[0].alpha_score(True, v[1], 0)
            v[0].remove_gap(['NA'])
            cu_alpha2 = v[0].alpha_score(False, v[1], 3)
            anno2 = copy.deepcopy(v[0])
            anno2.merge_tag()
            bi_alpha = anno2.alpha_score(True, v[1], 0)
            if isinstance(cu_alpha, float):
                sum1 += cu_alpha
                count_cu += 1
            sum2 += u_alpha
            sum3 += cu_alpha2
            sum4 += bi_alpha
            scores.append(
                dict(review_id=k,
                     uAlpha=u_alpha,
                     biAlpha=bi_alpha,
                     cuAlphaNON=cu_alpha2,
                     cuAlpha=cu_alpha))
        # Alpha scores
        right = pd.DataFrame(
            scores, columns=['review_id', 'uAlpha', 'biAlpha',
                             'cuAlpha', 'cuAlphaNON'])
        # Sample information (review_id, rating, word_count)
        left = pd.read_excel(samples, usecols=[0, 2, 4])

        data = pd.merge(left, right, how='inner').round(3)
        data.sort_values(by="uAlpha", inplace=True)
        data = data.append(
            {
                'review_id': 'avg',
                'uAlpha': round(sum2 / len(self.annotations), 3),
                'biAlpha': round(sum4 / len(self.annotations), 3),
                'cuAlphaNON': round(sum3 / len(self.annotations), 3),
                'cuAlpha': round(sum1 / count_cu, 3)
            },
            ignore_index=True)
        data.to_csv(output)

    def alpha_leave_one(self, rater_dict, gap, minL, output):
        """
        Compute alpha scores after removing each rater per review.
        :param rater_dict: Dictionary mapping rater name to an anonymity.
        :param gap: True for uAlpha, false otherwise.
        :param minL: Minal length of gap.
        :param output: Path of output file
        """
        rater_dict = rater_dict
        raters = list(rater_dict.keys())
        scores = []
        sums = []
        col_names = []

        for i in range(len(raters)):
            rater = raters[i]
            sums.insert(i, 0)
            scores.insert(i, [])
            col_names.insert(i, '')
            for review, v in self.annotations.items():
                annotation, length = v[0], v[1]
                if len(annotation.tags) > 3:
                    print('original: ', review, rater, annotation.gap,
                          *annotation.tags)
                if rater in annotation.raters:
                    units = []
                    for r in annotation.raters:
                        if r != rater:
                            units.extend(annotation.units[r])
                    anno2 = Annotation(units)
                else:
                    anno2 = copy.deepcopy(annotation)
                if gap:
                    alpha = anno2.alpha_score(gap, length, minL)
                    if alpha > 1:
                        print(length)
                        print(anno2.tags, anno2.length, anno2.begin,
                              anno2.raters, anno2.gap)
                elif minL > 0:
                    alpha = anno2.alpha_score(False, length, minL)
                else:
                    alpha = anno2.alpha_score()
                sums[i] += alpha
                scores[i].append(alpha)
            avg = round(sums[i] / len(self.annotations), 3)
            scores[i].append(avg)
            col_names[i] = 'r\\' + rater_dict[rater]

        result = pd.DataFrame()
        reviews = list(self.annotations.keys())
        reviews.append('avg')
        result['review_id'] = reviews
        for j in range(len(raters)):
            result[col_names[j]] = scores[j]
        result.to_csv(output)

    def alpha_tag(self, tags, general, output):
        """
        Compute uAlpha per tag.
        Generate a csv containing score per tag and general Score.
        :param tags: List of tags.
        :param general: File containing general scores.
        :param output: Path of output file.
        """
        u_scores = []
        u_sums = []
        u_names = []
        avgs = []

        for i in range(len(tags)):
            u_scores.insert(i, [])
            u_sums.insert(i, 0)
            u_names.insert(i, '')
            avgs.insert(i, 0)
            for k, v in self.annotations.items():
                anno, length = v[0], v[1]
                units_new = []
                for units in anno.units.values():
                    units_new.extend([u for u in units if u.tag == tags[i]])
                if units_new:
                    anno2 = Annotation(units_new)
                    if len(anno2.raters) == 1:
                        print("only rater" + anno2.raters[0] +
                              " left, uAlpha = 1")
                        u_scores[i].append('single rater')
                    else:
                        u_alpha = anno2.alpha_score('NA', length)
                        if u_alpha < 0 or u_alpha >= 1:
                            print(u_alpha, length, *units_new)
                        u_sums[i] += u_alpha
                        u_scores[i].append(u_alpha)
                else:
                    print(tags[i] + " does not exist")
                    u_scores[i].append('no ' + tags[i])
            u_names[i] = tags[i] + '_uAlpha'
            score_new = [
                score for score in u_scores[i] if isinstance(score, float)
            ]
            avgs[i] = round(u_sums[i] / len(score_new), 3)

        # Alpha per tag
        left = pd.DataFrame()
        left['review_id'] = list(self.annotations.keys())
        for j in range(len(tags)):
            left[u_names[j]] = u_scores[j]
        # Genral alpha
        right = pd.read_csv(general, usecols=['review_id', 'uAlpha'])
        avg_general = right.values.tolist()[-1][1]
        right.drop([len(right) - 1], inplace=True)
        data = pd.merge(left, right, how='inner').round(3)
        data.sort_values(by="uAlpha", inplace=True)
        avg_row = ['avg']
        avg_row.extend(avgs)
        avg_row.append(avg_general)
        data.loc[len(data)] = avg_row

        data.to_csv(output)

    def alpha_merge(self, general, merge_file, output):
        """
        Merge all annotations together and compute an alpha score.
        Create a file containing merged score and general score;
        And a file containing merged annotation.
        :param general: File for general scores.
        :param merge_file: Path for merged annotation.
        :param output: Path for score file.
        :return: Number of total units, pos/neg tags and gaps,
        offset length of merged file.
        """

        units_new = []
        length_total = 0
        rows = []
        raters = ['r1', 'r2', 'r3']

        for k, v in self.annotations.items():
            print("Calculating " + k + "...")
            anno, length = v[0], v[1]
            i = 0
            for rater, units in anno.units.items():
                for unit in units:
                    rows.append(
                        dict(rater=raters[i],
                             tag=unit.tag,
                             begin=unit.begin + length_total,
                             end=unit.end + length_total))
                    unit_new = Unit(raters[i], unit.tag,
                                    unit.begin + length_total,
                                    unit.end + length_total)
                    units_new.append(unit_new)
                i += 1
            length_total += length
        print("merged annotation: ", length_total)

        length_name = 'length=' + str(length_total)
        merged = pd.DataFrame(
            {
                'rater': length_name,
                'tag': 0,
                'begin': 0,
                'end': 0
            }, index=[0])
        merged = merged.append(rows, ignore_index=True)
        merged.to_csv(merge_file, index=None, header=False)

        print(len(units_new))
        count = [0, 0, 0]  # [#POS, #NEG, #GAP]
        anno2 = Annotation(units_new)
        # print(anno2.tags, anno2.gap)
        anno2.fill_gap(length_total, 0)
        # print(anno2.tags)
        for rater, units in anno2.units.items():
            for unit in units:
                if unit.tag == 'POS':
                    count[0] += 1
                elif unit.tag == 'NEG':
                    count[1] += 1
                else:
                    count[2] += 1

        anno2.remove_gap(['NA'])
        # print(len(units_new), count, length_total)
        # print(anno2.tags)
        print("Calculating cu_alpha...")
        cu_alpha = anno2.alpha_score()
        print('cu_alpha =', cu_alpha)
        print("Calculating u_alpha...")
        u_alpha = anno2.alpha_score(True, length_total)
        print('u_alpha =', u_alpha)
        print('Calculating cu_alphaNON...')
        cu_alphaNON = anno2.alpha_score(False, length_total, 3)
        print('cu_alphaNON =', cu_alphaNON)
        print('Calculating bi_alpha...')
        anno2.merge_tag()
        print(anno2.tags)
        bi_alpha = anno2.alpha_score(True, length_total)
        print('bi_alpha =', bi_alpha)

        result = pd.DataFrame(dict(type='merge',
                                   uAlpha=u_alpha,
                                   biAlpha=bi_alpha,
                                   cuAlpha=cu_alpha,
                                   cuAlphaNON=cu_alphaNON
                                   ),
                              index=[0])
        avg = pd.read_csv(general,
                          usecols=['uAlpha', 'biAlpha', 'cuAlpha',
                                   'cuAlphaNON']).values.tolist()[-1]
        # print(avg)
        result = result.append(dict(type='general',
                                    uAlpha=avg[0],
                                    biAlpha=avg[1],
                                    cuAlpha=avg[2],
                                    cuAlphaNON=avg[3]
                                    ),
                               ignore_index=True)
        result.to_csv(output,
                      columns=['type', 'uAlpha', 'biAlpha',
                               'cuAlpha', 'cuAlphaNON'])

        return len(units_new), count, length_total


if __name__ == "__main__":

    agreement = Agreement(
        '../../Data Folder/annotated/score/review_length.csv',
        '../../Data Folder/annotated/annotation/')

    # Example Implementations
    """
    agreement.alpha_general(
        '../../Data Folder/sampling/samples.xlsx',
        '../../Data Folder/annotated/score/review_alpha_test.csv')
    rater = {'R1': 'r1', 'R2': 'r2',
              'R3': 'r3', 'R4': 'r4',
              'R5': 'r5', 'R6': 'r6', 'R7': 'r7'}
    agreement.alpha_leave_one(
        rater, True, 0,
        '../../Data Folder/annotated/score/alpha_leave_one_uAlpha.csv')
    agreement.alpha_tag(['POS', 'NEG'],
                        '../../Data Folder/annotated/score/review_alpha.csv',
                        '../../Data Folder/annotated/score/alpha_tag.csv')
    agreement.alpha_merge(
        '../../Data Folder/annotated/score/review_alpha.csv',
        '../../Data Folder/annotated/annotation/merged_review.csv',
        '../../Data Folder/annotated/score/alpha_merge.csv')
    """
