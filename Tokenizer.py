import re

import bs4
from bs4 import BeautifulSoup

from krovetzstemmer import Stemmer

token_split_pattern = re.compile(r"[ .,!#\-]")  # split into tokens on dashes, commas, and punctuation
token_filter_pattern = re.compile(r"[^a-zA-Z0-9]")  # filter out non-alphanumeric chars
opening_tag_filter_pattern = re.compile(r"<(?P<tag_name>[A-Za-z0-9]+).*>")
closing_tag_filter_pattern = re.compile(r"</(?P<tag_name>[A-Za-z0-9]+).*>")

stemmer = Stemmer()


def tokenize_html(html_content: str, encoding: str, max_n_gram_size) -> {str: {str: [int]}}:
    """
    Returns a dict containing stemmed token as key with list of positions
    """
    soup = BeautifulSoup(html_content, features="lxml")
    doc_term_dict = {
        "title": {},
        "header": {},
        "bold": {},
        "text": {},
    }

    header_tag_names = {"h1", "h2", "h3", "h4", "h5", "h6"}
    important_tag_names = {"b", "i", "em", "strong"}

    last_n_terms = []
    pos = 1

    def explore_r(parent_tag: bs4.element.Tag):
        nonlocal pos
        for child in parent_tag.children:
            if type(child) is bs4.element.Tag:
                explore_r(child)
            elif type(child) is bs4.element.NavigableString:
                last_n_terms.clear()
                for token in re.split(token_split_pattern, str(child)):
                    term = stemmer.stem(re.sub(token_filter_pattern, "", token).lower())

                    if len(term) > 0:

                        last_n_terms.insert(0, term)
                        if len(last_n_terms) > max_n_gram_size:
                            del last_n_terms[max_n_gram_size]
                        pos += 1

                        for i in range(1, len(last_n_terms) + 1):

                            term = " ".join(last_n_terms[:i])
                            term_pos = pos - i

                            doc_term_dict["text"].setdefault(term, [])
                            doc_term_dict["text"][term].append(term_pos)

                            if parent_tag.name == "title":
                                doc_term_dict["title"].setdefault(term, [])
                                doc_term_dict["title"][term].append(term_pos)
                                continue
                            if any(parent.name in header_tag_names for parent in parent_tag.parents):
                                doc_term_dict["header"].setdefault(term, [])
                                doc_term_dict["header"][term].append(term_pos)
                                continue
                            if any(parent.name in important_tag_names for parent in parent_tag.parents):
                                doc_term_dict["bold"].setdefault(term, [])
                                doc_term_dict["bold"][term].append(term_pos)
                                continue

    if len(soup.contents) > 0:
        explore_r(soup.html)

    return doc_term_dict


def get_doc_simhash(html_content: str):

    soup = BeautifulSoup(html_content, features="lxml")

    term_frequency_counts = {}

    for line in soup.get_text().split('\n'):
        for token in re.split(token_split_pattern, line):
            term = stemmer.stem(re.sub(token_filter_pattern, "", token).lower())
            if len(term) > 0:
                term_frequency_counts.setdefault(term, 0)
                term_frequency_counts[term] += 1

    term_hashes = {term: hash(term) for term in term_frequency_counts}

    v = [0] * 32

    for i in range(32):
        for term, frequency in term_frequency_counts.items():
            if term_hashes[term] & 1 == 1:
                v[i] += frequency
            else:
                v[i] -= frequency
            term_hashes[term] >>= 1

    sim_hash = 0
    for i in reversed(range(32)):
        if v[i] > 0:
            sim_hash += 1
        sim_hash <<= 1

    return sim_hash

#def get_doc_fingerprints(html_content: str) -> {int}:





    # n = 3
    # last_n_terms = []
    # all_fingerprints: {int} = set()
    #
    # max_fingerprints = 1000
    #
    # def explore_r(parent_tag: bs4.element.Tag):
    #     for child in parent_tag.children:
    #         if type(child) is bs4.element.Tag:
    #             explore_r(child)
    #         elif type(child) is bs4.element.NavigableString:
    #
    #             last_n_terms.clear()
    #             for token in re.split(token_split_pattern, str(child)):
    #                 term = stemmer.stem(re.sub(token_filter_pattern, "", token).lower())
    #
    #                 if len(term) > 0:
    #
    #                     last_n_terms.insert(0, term)
    #                     if len(last_n_terms) > n:
    #                         del last_n_terms[n]
    #
    #                     if len(last_n_terms) == n:
    #
    #                         all_fingerprints.add(hash(" ".join(last_n_terms)))
    #
    # if len(soup.contents) == 1:
    #     explore_r(soup.html)
    #
    # doc_finger_prints: {int} = set()
    #
    # for _ in range(min(max_fingerprints, len(all_fingerprints))):
    #     doc_finger_prints.add(all_fingerprints.pop())
    #
    # return doc_finger_prints


def get_page_links(html_content: str, max_n_gram_size):
    soup = BeautifulSoup(html_content, features="lxml")
    target_url_term_frequency_dict = {}

    last_n_terms = []

    def explore_r(parent_tag: bs4.element.Tag):
        for child in parent_tag.children:
            if type(child) is bs4.element.Tag:
                explore_r(child)
            elif type(child) is bs4.element.NavigableString:

                if not any(parent.name == "a" and "href" in parent.attrs for parent in child.parents):
                    continue
                target_url = ""
                for parent in child.parents:
                    if parent.name == "a" and "href" in parent.attrs:
                        target_url = parent["href"]

                last_n_terms.clear()
                for token in re.split(token_split_pattern, str(child)):
                    term = stemmer.stem(re.sub(token_filter_pattern, "", token).lower())

                    if len(term) > 0:

                        last_n_terms.insert(0, term)
                        if len(last_n_terms) > max_n_gram_size:
                            del last_n_terms[max_n_gram_size]

                        for i in range(1, len(last_n_terms) + 1):

                            term = " ".join(last_n_terms[:i])

                            target_url_term_frequency_dict.setdefault(target_url, {})
                            target_url_term_frequency_dict[target_url].setdefault(term, 0)
                            target_url_term_frequency_dict[target_url][term] += 1

    if len(soup.contents) > 0:
        explore_r(soup.html)

    return target_url_term_frequency_dict


def tokenize_query(query: str, max_n_gram_size: int) -> {str: int}:
    last_n_terms = []

    term_frequencies = {}

    for token in re.split(token_split_pattern, query):
        current_term = stemmer.stem(re.sub(token_filter_pattern, "", token).lower())

        if len(current_term) > 0:

            last_n_terms.insert(0, current_term)
            if len(last_n_terms) > max_n_gram_size:
                del last_n_terms[max_n_gram_size]

            for i in range(1, len(last_n_terms) + 1):

                term = " ".join(reversed(last_n_terms[:i]))

                term_frequencies.setdefault(term, 0)
                term_frequencies[term] += 1

    return term_frequencies
