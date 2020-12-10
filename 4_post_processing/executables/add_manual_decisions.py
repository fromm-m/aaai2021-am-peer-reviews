path = "../data/text_sentences_majority.csv"
with open(path) as f:
    data = f.readlines()
    new_data = ""
    count_replaced = 0
    for line in data:
        if 'None' in line:
            text_1 = 'It seems that this sort of design issues can be'
            if 'graph20_29_3' in line and text_1 in line:
                line = line.replace('None', 'NEG')
                count_replaced = count_replaced + 1
            if 'graph20_39_3' in line:
                line = line.replace('None', 'POS')
                count_replaced = count_replaced + 1
            if 'graph20_43_1' in line:
                line = line.replace('None', 'NA')
                count_replaced = count_replaced + 1
            if 'graph20_53_2' in line:
                line = line.replace('None', 'NEG')
                count_replaced = count_replaced + 1
            text2 = 'The presented results indicate that SRL is useful '
            text3 = 'appendix includes some tests in this direction'
            text4 = 'The'
            if 'iclr19_1091_1' in line and text2 in line:
                line = line.replace('None', 'POS')
                count_replaced = count_replaced + 1
            if 'iclr19_1091_1' in line and text3 in line:
                line = line.replace('None', 'NA')
                count_replaced = count_replaced + 1
            if 'iclr19_1091_1' in line and text4 in line:
                line = line.replace('None', 'NA')
                count_replaced = count_replaced + 1
            text_5 = 'if you had a stronger baseline e.g.'
            text_6 = 'a bag-of-words Drawer model which works off of the'
            if 'iclr19_261_3' in line and text_5 in line:
                line = line.replace('None', 'NEG')
                count_replaced = count_replaced + 1
            if 'iclr19_261_3' in line and text_6 in line:
                line = line.replace('None', 'NA')
                count_replaced = count_replaced + 1
            text_7 = 'the concept of normalizing flow is simple, and it has '
            text_8 = 'there seems no work on applying it for policy'
            if 'iclr19_495_1' in line and text_7 in line:
                line = line.replace('None', 'NA', 1)
                count_replaced = count_replaced + 1
            if 'iclr19_495_1' in line and text_8 in line:
                line = line.replace('None', 'POS', 1)
                count_replaced = count_replaced + 1
            if 'iclr19_659_2' in line:
                line = line.replace('None', 'NA')
                count_replaced = count_replaced + 1
            if 'iclr19_997_3' in line:
                line = line.replace('None', 'NEG')
                count_replaced = count_replaced + 1
            text_9 = 'steer NNs towards the correct decision'
            text_10 = 'completely alleviating this concern may once again be'
            text_11 = 'it could be significantly alleviated by generating '
            if 'iclr20_1493_2' in line and text_9 in line:
                line = line.replace('None', 'NA', 1)
                count_replaced = count_replaced + 1
            if 'iclr20_1493_2' in line and text_10 in line:
                line = line.replace('None', 'NA', 1)
                count_replaced = count_replaced + 1
            if 'iclr20_1493_2' in line and text_11 in line:
                line = line.replace('None', 'NEG', 1)
                count_replaced = count_replaced + 1
            if 'iclr20_526_3' in line:
                line = line.replace('None', 'POS')
                count_replaced = count_replaced + 1
            if 'midl19_13_2' in line:
                line = line.replace('None', 'NEG')
                count_replaced = count_replaced + 1
            if 'midl19_51_1' in line:
                line = line.replace('None', 'NA')
                count_replaced = count_replaced + 1
            if 'midl19_52_2' in line:
                line = line.replace('None', 'POS')
                count_replaced = count_replaced + 1
            if 'midl20_127_4' in line:
                line = line.replace('None', 'NA')
                count_replaced = count_replaced + 1
            if 'midl20_96_3' in line:
                line = line.replace('None', 'POS')
                count_replaced = count_replaced + 1
            if 'neuroai19_2_2' in line:
                line = line.replace('None', 'NA')
                count_replaced = count_replaced + 1
            if 'neuroai19_32_1' in line:
                line = line.replace('None', 'NA')
                count_replaced = count_replaced + 1
            text_12 = 'The present paper makes the important case that random'
            text_13 = 'Very well written'
            if 'neuroai19_34_2' in line and text_12 in line:
                line = line.replace('None', 'NA')
                count_replaced = count_replaced + 1
            if 'neuroai19_34_2' in line and text_13 in line:
                line = line.replace('None', 'POS')
                count_replaced = count_replaced + 1
            if 'neuroai19_37_3' in line:
                line = line.replace('None', 'NA')
                count_replaced = count_replaced + 1

        new_data = new_data + line
print(count_replaced)
with open(path, 'w') as f:
    f.writelines(new_data)
    # print(new_data)
