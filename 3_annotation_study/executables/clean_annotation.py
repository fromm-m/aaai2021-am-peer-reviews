import re
import pandas as pd
from collections import defaultdict


def get_assignment(file):
    """
    Map text to raters
    :param file: sample file
    :return: dictionary text -> raters
    """
    print('Loading sample information...')
    text_to_raters = defaultdict(list)
    data = pd.read_excel(file, usecols=[0, 5], names=None)
    assignment_list = data.values.tolist()
    for mapping in assignment_list:
        mapping[1] = [eval(name) for name in mapping[1].strip()[1:-1].split()]
        text_to_raters[mapping[0]] = mapping[1]
    return text_to_raters


def get_units(path, text, names):
    """
    Get all annotated units of a text
    :param path: Path of text
    :param text: Annotated text
    :param names: List of raters
    :return: Annotation units
    """
    unit_count = 1
    unit_rows = []
    text_end = 0

    for name in names:
        tags = {}
        try:
            with open(path + text + '.txt/' + name + '.tsv', 'r') as f:
                print('Loading ' + text + ' annotated by ' + name)
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 4:
                        token_num, offset, _, tag = parts[0], parts[1], \
                                                    parts[2], parts[3]
                        match_offset = re.match(r'([0-9]+)-([0-9]+)', offset)
                        match_tag = re.match(r'(POS|NEG)\[([0-9]+)]', tag)
                        begin_offset, end_offset = \
                            match_offset.group(1), match_offset.group(2)
                        if text_end < int(end_offset):
                            text_end = int(end_offset)
                        if match_tag:
                            if tag not in tags.keys():
                                tags[tag] = [token_num, token_num,
                                             begin_offset, end_offset]
                            else:
                                tags[tag][3] = end_offset
                                tags[tag][1] = token_num
        except FileNotFoundError:
            print(path + text + '.txt/' + name + '.tsv does not exist')

        for k, v in tags.items():
            unit_rows.append([str(unit_count), name,
                              k[:3], v[0], v[1], v[2], v[3]])
            unit_count += 1
    return unit_rows, text_end


def write_annotation(path, file, unit_list):
    """
    Write annotation units to a csv file
    :param path: Directory, where the file should be stored
    :param file: File name
    :param unit_list: List of units
    :return: CSV file
    """
    print('Writing annotation ' + file)
    unit_lines = []
    for unit in unit_list:
        unit_lines.append(
            dict(unit_id=unit[0], rater=unit[1], tag=unit[2],
                 token_start=unit[3], token_end=unit[4],
                 offset_start=unit[5], offset_end=unit[6]))
    df = pd.DataFrame(unit_lines,
                      columns=['unit_id', 'rater', 'tag',
                               'token_start', 'token_end',
                               'offset_start', 'offset_end'])
    df.to_csv(path + file + '_anno.csv')


def write_review_length(file, reviews, lengths):
    print("Writing review length...")
    length_lines = []
    for i in range(len(reviews)):
        length_lines.append(dict(review=reviews[i], length=lengths[i]))
    f = pd.DataFrame(length_lines, columns=['review', 'length'])
    f.to_csv(file + '.csv')


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="This script cleans exported annotations; " +
        "For each annotation it generates a csv file "
        "storing each unit per line;" + "Besides, it generates a csv file "
        "storing total offset length of each annotation")
    parser.add_argument("sample",
                        metavar="sample.xlsx",
                        help="excel storing the information of samples")
    parser.add_argument("path1",
                        metavar="path_annotation",
                        help="parent directory of exported annotation folder")
    parser.add_argument("path2",
                        metavar="path_units",
                        help="directory of annotated units")
    parser.add_argument("path3",
                        metavar="path_length",
                        help="directory/file name for annotation length")

    args = parser.parse_args()

    assignment = get_assignment(args.sample)
    length = []
    rows = []

    for review, raters in assignment.items():
        units, end = get_units(args.path1, review, raters)
        length.append(end)
        write_annotation(args.path2, review, units)
    write_review_length(args.path3, list(assignment.keys()), length)
