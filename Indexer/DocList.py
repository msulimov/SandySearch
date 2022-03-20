# from sortedcontainers.sortedlist import SortedList
import math


class PostingsList:
    # posting_list
    # term_frequency

    # TODO method to sort postings by Page Rank/Hit Rank on request given dict of doc_id rankings

    delim = ','

    def __init__(self, store_positions: bool, dump_data: str = None, raw_posting_data_list: [str] = None):

        self.store_positions: bool = store_positions
        self.term_frequency: int = 0
        self.postings_list: [Posting] = []
        self.postings_dict: {int: Posting} = {}

        if dump_data is not None:  # load directly from index file
            data = dump_data.split(PostingsList.delim)
            self.term_frequency = int(data[0])
            self.postings_list = [Posting(posting_data=posting_data)
                                  for posting_data in data[1:]]

        elif raw_posting_data_list is not None:  # loading in multiple partial indexes to merge them, can be slower
            self.postings_list = [Posting(posting_data=posting_data)
                                  for posting_list_data in raw_posting_data_list
                                  for posting_data in posting_list_data.split(PostingsList.delim)]
            self.term_frequency = sum(posting.doc_term_frequency for posting in self.postings_list)

        self.postings_dict = {posting.doc_id: posting for posting in self.postings_list}

    def compute_local_tf_idf(self, total_docs: int, copy_to_global: bool = False):
        """Computes tf_idf for postings in THIS tiered index (local)"""
        for posting in self.postings_list:
            posting.local_tf_idf_score = (1 + math.log10(posting.doc_term_frequency)) * math.log10(
                total_docs / len(self.postings_dict))
            if copy_to_global:
                posting.global_tf_idf_score = posting.local_tf_idf_score

    def add_global_tf_idf(self, global_postings_list: 'PostingsList'):

        for local_posting in self.postings_list:
            # if global_postings_list is None or local_posting.doc_id not in global_postings_list.postings_dict:
            #     local_posting.global_tf_idf_score = local_posting.local_tf_idf_score
            #     continue
            local_posting.global_tf_idf_score = \
                global_postings_list.postings_dict[local_posting.doc_id].global_tf_idf_score

    def create_posting(self, doc_id: int, pos_list: [int]):
        """
        Creates a posting for the doc_id and term positions_list and adds it to the end of the posting list
        """
        self.postings_list.append(Posting(doc_id=doc_id,
                                          term_frequency=len(pos_list),
                                          pos_list=pos_list if self.store_positions else None
                                          ))

    def get_doc_ids(self) -> [int]:
        return list(self.postings_dict.keys())

    def dump(self):
        """Dumps the whole data for this PostingsList to a string for loading directly from index later"""
        dumped_postings_data = PostingsList.delim.join(posting.dump() for posting in self.postings_list)
        return f"{self.term_frequency}{PostingsList.delim}{dumped_postings_data}"

    def dump_raw_postings(self):
        """Dumps only the raw postings to a string for storage in a partial index file, allowing later merging"""
        return PostingsList.delim.join(posting.dump() for posting in self.postings_list)

    def set_page_rankings(self, doc_page_rankings: [int]):
        for posting in self.postings_list:
            posting.page_rank = doc_page_rankings[posting.doc_id]

    def sort(self, page_rank_factor: float, global_tf_idf_factor: float, local_tf_idf_factor: float):
        self.postings_list.sort(
            reverse=True,
            key=lambda x:
                page_rank_factor * x.page_rank +
                local_tf_idf_factor * x.local_tf_idf_score +
                global_tf_idf_factor * x.global_tf_idf_score
        )

    def limit(self, top_k_postings: int):
        del self.postings_list[top_k_postings:]

        self.term_frequency = 0
        self.postings_dict.clear()
        for posting in self.postings_list:
            self.postings_dict[posting.doc_id] = posting
            self.term_frequency += posting.doc_term_frequency

    def __len__(self):
        return len(self.postings_dict)


class Posting:
    delim = ':'

    def __init__(self,
                 doc_id: int = None,
                 pos_list: [int] = None,
                 term_frequency: int = 0,
                 posting_data: str = None):

        self.doc_id: int = doc_id
        self.term_pos_list: [int] = pos_list
        self.doc_term_frequency: int = term_frequency
        self.local_tf_idf_score: float = -1.0
        self.global_tf_idf_score: float = -1.0
        self.page_rank: float = -1.0

        if posting_data is not None:
            data = posting_data.split(Posting.delim)
            self.doc_id = int(data[0])
            self.doc_term_frequency = int(data[1])
            self.local_tf_idf_score = float(data[2])
            self.global_tf_idf_score = float(data[3])
            self.page_rank = float(data[4])
            self.term_pos_list = None
            if len(data) > 5:
                self.term_pos_list = [int(pos) for pos in data[5:]]

    def dump(self) -> str:
        dump_str = \
            f"{self.doc_id}{Posting.delim}" \
            f"{self.doc_term_frequency}{Posting.delim}" \
            f"{round(self.local_tf_idf_score, 3)}{Posting.delim}" \
            f"{round(self.global_tf_idf_score, 3)}{Posting.delim}" \
            f"{round(self.page_rank, 3)}"
        if self.term_pos_list is not None:
            dump_str += f"{Posting.delim}"
            dump_str += Posting.delim.join(str(pos) for pos in self.term_pos_list)
        return dump_str
