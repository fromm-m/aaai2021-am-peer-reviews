import re


def clean_data(string):
    if string is None:
        return string
    else:
        return re.sub(' +', ' ', remove_markdown(encode_decode(remove_dollar(
            remove_escape_sequences(remove_url(string))))))
        # string = remove_url(string)
        # string = remove_escape_sequences(string)
        # string = remove_dollar(string)
        # string = encode_decode(string)
        # string = remove_markdown(string)
        # string = re.sub(' +', ' ', string)
        # return string


def clean_data_iclr(string):
    if string is None:
        return string
    else:
        return re.sub(' +', ' ', encode_decode(remove_dollar(
            remove_escape_sequences(remove_url(string)))))


def remove_markdown(string):
    try:
        string.index('\\')
    except ValueError:
        # no markdown in the string, so it gets returned as it is
        return string
    # space at the end of the string is important for the if-elif-else-case
    string = string + ' '
    while 1:
        try:
            markdown_index = string.index('\\')
            markdown_index2 = string.find('{', markdown_index)
            markdown_index4 = string.find(' ', markdown_index)
            if string.find('{', markdown_index) != -1\
                    and markdown_index2 < markdown_index4:
                markdown_index3 = string.find('}', markdown_index2)
                string = string[:markdown_index] + ' '\
                    + string[markdown_index2 + 1:markdown_index3]\
                    + ' ' + string[markdown_index3 + 1:]
            elif markdown_index4 != -1:
                # removing everything between a backslash and the next space
                # works, because all escape sequences should be removed by
                # the encode_decode()-function
                string = string[:markdown_index] + ' '\
                    + string[markdown_index4 + 1:]
            else:
                break
        except ValueError:
            break
    return string[:-1]


def remove_dollar(string):
    string = string + ' '
    while 1:
        try:
            dollar_index = string.index('$')
            if dollar_index == -1:
                break
            elif string.find('$', dollar_index+1)\
                    < string.find(' ', dollar_index+1)\
                    and string.find('$', dollar_index+1) != -1:
                dollar_index2 = string.find('$', dollar_index+1)
                string = string[:dollar_index] + " pseudo-formula "\
                    + string[dollar_index2 + 1:]
            elif string.find('$', dollar_index+1)\
                    > string.find(' ', dollar_index+1):
                dollar_index2 = string.find(' ', dollar_index)
                string = string[:dollar_index] + ' ' + string[dollar_index2:]
            else:
                break
        except ValueError:
            break
    return string[:-1]


def remove_url(string):
    urls = find_urls(string)
    for url in urls:
        string = string.replace(url, 'pseudo-url')
    return string


def remove_escape_sequences(string):
    return string.replace('\n', ' ').replace('\t', ' ')\
        .replace('\o', '').replace('\b', ' ') # noqa


def encode_decode(string):
    # ascii encode and decode removes the \u followed by some unicode symbols,
    # which is really hard to remove with a str.replace()
    return string.encode('ascii', 'ignore').decode()


def find_urls(string):
    try:
        # regex for urls in string
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))" # noqa
        url = re.findall(regex, string)
        return [x[0] for x in url]
    except IndexError:
        return []
