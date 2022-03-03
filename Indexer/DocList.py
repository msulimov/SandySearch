# from sortedcontainers.sortedlist import SortedList
import math


class PostingsList:

    # posting_list
    # term_frequency

    # TODO method to sort postings by Page Rank/Hit Rank on request given dict of doc_id rankings

    delim = ','

    def __init__(self, dump_data: str = None, raw_posting_data_list: [str] = None):

        self.term_frequency = 0
        self.postings_list = None
        self.doc_ids = None
        self.postings_position = 0
        self.num_docs = 0

        if dump_data is not None:  # load directly from index file
            data = dump_data.split(PostingsList.delim)
            self.term_frequency = int(data[0])
            self.postings_list = [Posting(posting_data=posting_data) for posting_data in data[1:]]

        elif raw_posting_data_list is not None:  # loading in multiple partial indexes to merge them, can be slower
            self.postings_list = [Posting(posting_data=posting_data) for posting_list_data in raw_posting_data_list
                                  for posting_data in posting_list_data.split(PostingsList.delim)]
            self.term_frequency = sum(len(posting) for posting in self.postings_list)
            self.num_docs = len(self.postings_list)
        else:  # building a new PostingList as tokenize html
            self.postings_list = []

    def compute_stats(self, total_docs: int):
        """Computes tf_idf + other postings stats and stores them"""
        for posting in self.postings_list:
            posting.tf_idf_score = (1 + math.log10(posting.term_frequency))*math.log10(total_docs/self.num_docs)

    def create_posting(self, doc_id: int, pos_list: [int]):
        """
        Creates a posting for the doc_id and term positions_list and adds it to the end of the posting list
        """
        self.postings_list.append(Posting(doc_id=doc_id, pos_list=pos_list))

    def get_doc_ids(self) -> [int]:
        if self.doc_ids is None:
            self.doc_ids = set(posting.doc_id for posting in self.postings_list)
        return self.doc_ids

    def dump(self):
        """Dumps the whole data for this PostingsList to a string for loading directly from index later"""
        dumped_postings_data = PostingsList.delim.join(posting.dump() for posting in self.postings_list)
        return f"{self.term_frequency}{PostingsList.delim}{dumped_postings_data}"

    def dump_raw_postings(self):
        """Dumps only the raw postings to a string for storage in a partial index file, allowing later merging"""
        return PostingsList.delim.join(posting.dump() for posting in self.postings_list)

    def current_posting(self):
        return self.postings_list[self.postings_position]

    def next_posting(self):
        self.postings_position += 1
        if self.postings_position == len(self.postings_list):
            self.postings_position = 0
            return None
        return self.postings_list[self.postings_position]

    def reset_postings_position(self):
        self.postings_position = 0

    def skip_to_doc_id(self, doc_id: int) -> bool:

        while self.postings_list[self.postings_position] < doc_id:
            self.postings_position += 1
            if self.postings_position == len(self.postings_list):
                self.postings_position = 0
                return False

        return True

    def __len__(self):
        return self.num_docs


class Posting:

    delim = ':'

    def __init__(self, doc_id=None, pos_list: [int] = None, posting_data: str = None, ):

        self.doc_id = doc_id
        self.term_pos_list = pos_list
        self.tf_idf_score = -1

        if posting_data is not None:
            data = posting_data.split(Posting.delim)
            self.doc_id = int(data[0])
            self.tf_idf_score = float(data[1])
            self.term_pos_list = [int(pos) for pos in data[2:]]

    def __lt__(self, other):
        if type(other) is int:
            return self.doc_id < other
        if type(other) is Posting:
            return self.doc_id < other.doc_id
        raise ValueError(f"Object of different type than int or Posting compared to Posting")

    @property
    def term_frequency(self):
        return len(self.term_pos_list)

    def __len__(self):
        return len(self.term_pos_list)

    def dump(self) -> str:
        term_positions_data = self.delim.join(f"{pos}" for pos in self.term_pos_list)
        return f"{self.doc_id}{Posting.delim}{round(self.tf_idf_score, 2)}{Posting.delim}{term_positions_data}"


def intersection(postings_lists: [PostingsList], k=10) -> [int]:

    if len(postings_lists) == 0:
        return []
    if len(postings_lists) == 1:
        postings_list = postings_lists[0].postings_list
        output = []
        for i in range(min(len(postings_list), k)):
            output.append(postings_list[i].doc_id)
        return output
        # return [posting.doc_id for posting in postings_list[:min(len(postings_list), k)]]

    postings_lists.sort(key=lambda posting_list: posting_list.num_docs)

    ordered_doc_ids = [posting.doc_id for posting in postings_lists[0].postings_list]

    for i in range(1, len(postings_lists)):
        next_doc_ids = []
        for doc_id in ordered_doc_ids:
            if not postings_lists[i].skip_to_doc_id(doc_id):
                break
            if postings_lists[i].current_posting().doc_id == doc_id:
                next_doc_ids.append(doc_id)
        postings_lists[i].reset_postings_position()
        ordered_doc_ids = next_doc_ids

    return ordered_doc_ids[:min(len(ordered_doc_ids), k)]










