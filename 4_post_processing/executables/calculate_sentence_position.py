import numpy as np
import pandas as pd
import ast

old_path = "../data/text_sentences_majority.csv"
sentences_with_segment_position = pd.read_csv(
                old_path, sep='\t ',
                names=['sentence_id', 'text', 'positions']).to_numpy()


def check_length(positions):
    part1 = positions.split("', '")[0][2:-1]
    part2 = positions.split("', '")[1][:-3]
    all_positions = part2.split(";")
    positives = 0
    negatives = 0
    neutrals = 0
    pos_nr = 0
    for position in all_positions:
        position_length_tuple = ast.literal_eval(part1.split(";")[pos_nr])
        length = position_length_tuple[1]
        if position == 'POS':
            positives = positives + length
        elif position == 'NEG':
            negatives = negatives + length
        elif position == 'NA':
            neutrals = neutrals + length
        else:
            return('ERROR')
        pos_nr = pos_nr + 1
    if positives > negatives:
        return 'POS'
    if negatives > positives:
        return 'NEG'
    if positives == negatives:
        return 'NA'
    return('ERROR')


def get_sentence_position(positions):
    positives = positions.count('POS')
    negatives = positions.count('NEG')
    neutrals = positions.count('NA')
    if positives > 0 and negatives == 0:
        return 'POS'
    if positives == 0 and negatives > 0:
        return 'NEG'
    if positives == 0 and negatives == 0 and neutrals > 0:
        return 'NA'
    if positives == 0 and negatives == 0 and neutrals == 0:
        return 'ERROR IN DATA'
    if positives > 0 and negatives > 0:
        if positives > negatives:
            return 'POS'
        if negatives > positives:
            return 'NEG'
        if positives == negatives:
            return(check_length(positions))
    return 'FORGOTTEN'


if __name__ == "__main__":
    sentence_positions = np.empty((0, 3), dtype=object)
    for sentence in sentences_with_segment_position:
        position = get_sentence_position(sentence[2])
        sentence_positions = np.append(
            sentence_positions,
            [[sentence[0], sentence[1], position]],
            axis=0)

    output_file = "../data/sentences_just_one_position.csv"
    np.savetxt(output_file, sentence_positions,
               fmt="%1s", delimiter='\t ', newline="\n")
