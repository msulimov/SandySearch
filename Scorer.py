import math

import Tokenizer
from Indexer.DocList import Posting
from Indexer.Index import Index
from Indexer.TieredIndex import TieredIndex


class Scorer:

    def __init__(self, tiered_index: TieredIndex):
        self.tiered_index = tiered_index
        self.returned_results: {int} = set()
        self.current_results: {int: float} = {}

    def sprint_search(self, query: str, k_results):
        scored_query = self.__score_query(query, self.tiered_index.max_n_grams)
        query_terms = [term for term in scored_query]

        self.current_results.clear()

        self.current_results.update(
            self._search(self.tiered_index.title_index, query_terms, scored_query, 8.0, k_results)
        )
        self.returned_results.update(self.current_results.keys())
        if len(self.current_results) >= k_results:
            return [self.tiered_index.doc_id_to_url_LUT[doc_id] for doc_id in
                    sorted((doc_id for doc_id in self.current_results),
                           key=lambda x: self.current_results[x],
                           reverse=True)
                    ]

        self.current_results.update(
            self._search(self.tiered_index.anchor_index, query_terms, scored_query, 7.0, k_results)
        )
        self.returned_results.update(self.current_results.keys())
        if len(self.current_results) >= k_results:
            return [self.tiered_index.doc_id_to_url_LUT[doc_id] for doc_id in
                    sorted((doc_id for doc_id in self.current_results),
                           key=lambda x: self.current_results[x],
                           reverse=True)
                    ]

        self.current_results.update(
            self._search(self.tiered_index.header_index, query_terms, scored_query, 5.0, k_results)
        )
        self.returned_results.update(self.current_results.keys())
        if len(self.current_results) >= k_results:
            return [self.tiered_index.doc_id_to_url_LUT[doc_id] for doc_id in
                    sorted((doc_id for doc_id in self.current_results),
                           key=lambda x: self.current_results[x],
                           reverse=True)
                    ]

        self.current_results.update(
            self._search(self.tiered_index.bold_index, query_terms, scored_query, 4.0, k_results)
        )
        self.returned_results.update(self.current_results.keys())
        if len(self.current_results) >= k_results:
            return [self.tiered_index.doc_id_to_url_LUT[doc_id] for doc_id in
                    sorted((doc_id for doc_id in self.current_results),
                           key=lambda x: self.current_results[x],
                           reverse=True)
                    ]

        self.current_results.update(
            self._search(self.tiered_index.limited_index, query_terms, scored_query, 1.0, k_results)
        )
        self.returned_results.update(self.current_results.keys())
        return [self.tiered_index.doc_id_to_url_LUT[doc_id] for doc_id in
                sorted((doc_id for doc_id in self.current_results),
                       key=lambda x: self.current_results[x],
                       reverse=True)
                ]

    def complete_search(self, query: str, k_results):
        scored_query = self.__score_query(query, self.tiered_index.max_n_grams)
        query_terms = [term for term in scored_query]

        self.current_results.clear()

        if all(self.tiered_index.complete_index.document_term_counts[term] < 600 for term in query_terms):
            self.current_results.update(
                self._search(self.tiered_index.complete_index, query_terms, scored_query, 1.0, k_results)
            )
        else:
            self.current_results.update(
                self._search(self.tiered_index.limited_index, query_terms, scored_query, 1.0, k_results)
            )
        self.returned_results.update(self.current_results.keys())
        return [self.tiered_index.doc_id_to_url_LUT[doc_id] for doc_id in
                sorted((doc_id for doc_id in self.current_results),
                       key=lambda x: self.current_results[x],
                       reverse=True)
                ]

    def new_search(self):
        self.returned_results.clear()

    def __score_query(self, query: str, max_n_grams: int) -> {str: float}:
        def score(term, count):
            return (1 + math.log10(count)) * \
                   math.log10(
                       len(self.tiered_index.complete_index.document_term_counts) /
                       self.tiered_index.complete_index.document_term_counts[term]
                   )

        query_term_counts = {term: count
                             for term, count in Tokenizer.tokenize_query(query, max_n_grams).items()
                             if term in self.tiered_index.complete_index.document_term_counts
                             }

        query_term_scores = {term: score(term, count) for term, count in query_term_counts.items()}
        normalized_factor = math.sqrt(sum(score ** 2 for score in query_term_scores.values()))
        return {term: term_score / normalized_factor for term, term_score in query_term_scores.items()}

    def _search(self,
                index: Index,
                query_terms: [int],
                scored_query: [float],
                score_weight: float,
                k_results: int) -> [int]:
        term_postings_lists = {term: index.retrieve_posting_list(term) for term in query_terms if term in index}
        doc_ids = {doc_id for postings_list in term_postings_lists.values() for doc_id in postings_list.postings_dict
                   # if doc_id not in self.returned_results
                   }
        sorted_doc_ids = sorted(
            doc_ids,
            key=lambda x: sum(1 for postings_list in term_postings_lists.values() if x in postings_list.postings_dict),
            reverse=True
        )

        results: {int: float} = {}

        global_tf_idf_weight = index.sort_weights["global_tf_idf"]
        local_tf_idf_weight = index.sort_weights["local_tf_idf"]
        page_rank_weight = index.sort_weights["page_rank"]

        doc_id_scores = [0.0] * len(query_terms)

        for doc_id in sorted_doc_ids:
            if len(results) >= k_results:
                return results
            for i, query_term in enumerate(query_terms):

                if query_term not in term_postings_lists or doc_id not in term_postings_lists[query_term].postings_dict:
                    doc_id_scores[i] = 0
                    continue

                doc_posting: Posting = term_postings_lists[query_term].postings_dict[doc_id]
                doc_id_scores[i] = \
                    doc_posting.global_tf_idf_score * scored_query[query_term] * global_tf_idf_weight + \
                    doc_posting.local_tf_idf_score * scored_query[query_term] * local_tf_idf_weight + \
                    doc_posting.page_rank * scored_query[query_term] * page_rank_weight
            normalize_factor = math.sqrt(sum(doc_term_score ** 2 for doc_term_score in doc_id_scores))
            doc_score = score_weight * sum(doc_id_score / normalize_factor for doc_id_score in doc_id_scores)

            results.setdefault(doc_id, 0)
            results[doc_id] += doc_score

        return results
