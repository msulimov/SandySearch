import re
from bs4 import BeautifulSoup
from nltk.stem import porter
from krovetzstemmer import Stemmer

token_split_pattern = re.compile(r"[ .,!#\-]")  # split into tokens on dashes, commas, and punctuation
token_filter_pattern = re.compile(r"[^a-zA-Z0-9]")  # filter out non-alphanumeric chars
# "<.*?>| "

stemmer = Stemmer()


def tokenize_html(html_content: str, encoding: str):
    """
    Returns a dict containing stemmed token as key with list of positions
    TODO return the extent list for title, headers, bold text, italic text
    """
    soup = BeautifulSoup(html_content, features="lxml")
    token_dict = __tokenize(soup)
    return token_dict


def __tokenize(soup) -> dict:

    # TODO build extent list too
    # TODO index bi-grams and tri-grams

    term_pos_dict = {}  # store str: list of positions

    pos = 1
    for line in soup.get_text().split('\n'):  # split retrieved page text by line
        for token in re.split(token_split_pattern, line):  # split line into tokens
            token = re.sub(token_filter_pattern, '', token).lower()  # filter invalid chars from token
            term = stemmer.stem(token)  # stem the token into a term

            if len(term) > 0:  # store term if not empty
                term_pos_dict.setdefault(term, [])
                term_pos_dict[term].append(pos)
            pos += 1

    return term_pos_dict


def tokenize_query(query: str) -> [str]:

    term_set = set()

    for token in re.split(token_split_pattern, query):  # split line into tokens
        token = re.sub(token_filter_pattern, '', token).lower()  # filter invalid chars from token
        term = stemmer.stem(token)  # stem the token into a term

        if len(term) > 0:  # store term if not empty
            term_set.add(term)

    return list(term_set)
