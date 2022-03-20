import json
import urllib.parse
from pathlib import Path

import Tokenizer
from Indexer.Index import Index
import Tokenizer


class TieredIndex:
    local_store_dir = "./Indexer/Local_Store"
    settings_directory = "./Indexer/Tiered_Indexes_Settings"

    def __enter__(self):

        self.title_index: Index = \
            Index(descriptor="title_index",
                  max_n_gram=3,
                  sort_weights={"page_rank": 0.40, "global_tf_idf": 0.20, "local_tf_idf": 0.40},
                  postings_list_size_limit=70,
                  store_positions=False,
                  )

        self.anchor_index: Index = \
            Index(descriptor="anchor_index",
                  max_n_gram=3,
                  sort_weights={"page_rank": 0.40, "global_tf_idf": 0.00, "local_tf_idf": 0.60},
                  postings_list_size_limit=90,
                  store_positions=False,
                  )

        self.header_index: Index = \
            Index(descriptor="headers_index",
                  max_n_gram=3,
                  sort_weights={"page_rank": 0.40, "global_tf_idf": 0.20, "local_tf_idf": 0.40},
                  postings_list_size_limit=120,
                  store_positions=True,
                  )

        self.bold_index: Index = \
            Index(descriptor="important_text_index",
                  max_n_gram=3,
                  sort_weights={"page_rank": 0.40, "global_tf_idf": 0.20, "local_tf_idf": 0.40},
                  postings_list_size_limit=150,
                  store_positions=True,
                  )

        self.limited_index: Index = \
            Index(descriptor="limited_text_index",
                  max_n_gram=3,
                  sort_weights={"page_rank": 0.40, "global_tf_idf": 0.60, "local_tf_idf": 0.00},
                  postings_list_size_limit=200,
                  store_positions=True,
                  )

        self.complete_index: Index = \
            Index(descriptor="all_text_index",
                  max_n_gram=3,
                  sort_weights={"page_rank": 0.40, "global_tf_idf": 0.60, "local_tf_idf": 0.00},
                  postings_list_size_limit=None,
                  store_positions=True,
                  )

        return self

    def __init__(self, max_n_grams: int, page_rank_iterations: int):

        self.processed_urls = set()
        self.parsed_html_hashes: {int} = {}
        self.doc_fingerprints: {int: {int}} = {}

        self.doc_id_to_url_LUT: {int: str} = {}
        self.url_to_doc_id_LUT: {str: int} = {}
        self.doc_id_counter = 0

        self.max_n_grams: int = max_n_grams

        self.page_rank_iterations = page_rank_iterations
        self.doc_in_edges: {int: {int}} = {}
        self.doc_out_edges: {int: {int}} = {}

        self.local_store_path = Path(TieredIndex.local_store_dir)
        assert self.local_store_path.exists(), f"Local store path {TieredIndex.local_store_dir} does not exist"
        assert self.local_store_path.is_dir(), f"Local store path {TieredIndex.local_store_dir} not a directory"

        self.settings_path: Path = Path(TieredIndex.settings_directory)
        assert self.settings_path.exists(), f"Settings path {Index.settings_directory} does not exist"
        assert self.settings_path.is_dir(), f"Settings path {Index.settings_directory} not a directory"

        self.settings_file_name = f"Tiered_IndexBuilder_settings.json"
        if Path(self.settings_path.joinpath(self.settings_file_name)).is_file():
            print(f"Found settings file, loading settings...", end="")
            self.__load_settings_from_json()
            print(f"Done")
        else:
            print(f"Did not find settings file")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"\nPreparing to close tiered index builder...")
        self.title_index.__exit__(exc_type, exc_val, exc_tb)
        self.anchor_index.__exit__(exc_type, exc_val, exc_tb)
        self.header_index.__exit__(exc_type, exc_val, exc_tb)
        self.bold_index.__exit__(exc_type, exc_val, exc_tb)
        self.limited_index.__exit__(exc_type, exc_val, exc_tb)
        self.complete_index.__exit__(exc_type, exc_val, exc_tb)
        print(f"Closed tiered index builder.")

    def build_tiered_indexes(self):

        print("-" * 120)
        print(f"Starting to build index from local store data")
        print(f"Clearing tiered indexes", end="")
        self.title_index.prep_for_build()
        print(f".", end="")
        self.anchor_index.prep_for_build()
        print(f".", end="")
        self.header_index.prep_for_build()
        print(f".", end="")
        self.bold_index.prep_for_build()
        print(f".", end="")
        self.limited_index.prep_for_build()
        print(f".", end="")
        self.complete_index.prep_for_build()
        print(f"Done\n")

        self.doc_id_counter = 0
        self.doc_in_edges.clear()
        self.doc_out_edges.clear()
        self.url_to_doc_id_LUT.clear()
        self.doc_id_to_url_LUT.clear()

        self.processed_urls.clear()
        self.parsed_html_hashes.clear()
        self.doc_fingerprints.clear()

        exact_duplicates_found = 0
        near_duplicates_found = 0

        print(f"Starting to parse pages in local store")
        for page_file in self.local_store_path.rglob("*.json"):  # iterate all json files in local store
            # print(f"Opening json file: {page_file}")

            # if self.doc_id_counter > 2500:
            #     break

            with open(page_file, "r") as page_json:  # open and read the json file
                data = json.load(page_json)  # load the json data using json.load
                if type(data) is not dict or len(data) != 3:  # fields must be url, content and encoding
                    print(f"Error parsing file {page_file}, json file must have url, content and encoding")
                raw_url, content, encoding = data.values()
                try:
                    url = urllib.parse.urldefrag(raw_url).url
                except ValueError:
                    print(f"Error parsing file {page_file}, url invalid format: {raw_url}")
                    continue

                if content is None or len(content) == 0:
                    print(f"Error parsing file {page_file}, content empty")
                if encoding is None or len(encoding) == 0:
                    print(f"Error parsing file {page_file}, encoding not specified")

                if url in self.processed_urls:  # skip if url already processed
                    print(f"\nAlready parsed url: {url}, ", end="")
                    if url in self.url_to_doc_id_LUT:
                        print(f"which is doc_id: {self.url_to_doc_id_LUT[url]}, skipping document")
                    else:
                        print(f"which was skipped due to duplicated or near duplicated html content")
                    continue

                self.processed_urls.add(url)

                html_hash = crc_hash(content)
                if html_hash in self.parsed_html_hashes:
                    print(f"\nDuplicate html content found between url: {url} "
                          f"and parsed url: {self.parsed_html_hashes[html_hash]}")
                    exact_duplicates_found += 1
                    continue
                self.parsed_html_hashes[html_hash] = url

                doc_simhash = Tokenizer.get_doc_simhash(content)

                near_doc_id = self.find_near_duplicate_doc(doc_simhash)
                if near_doc_id is not None:
                    print(f"\nNear duplicate content found between url: {url} "
                          f"and parsed url: {self.doc_id_to_url_LUT[doc_id]}")
                    near_duplicates_found += 1
                    continue

                doc_id = self.__add_doc(url)
                self.doc_fingerprints[doc_id] = doc_simhash

                print(f"\rParsing doc_id: {doc_id}, url: {url}", end="")

                page_token_dict = Tokenizer.tokenize_html(
                    content, encoding, self.max_n_grams)

                for title_term, positions in page_token_dict["title"].items():
                    self.title_index.add_term(term=title_term, doc_id=doc_id, positions=positions)

                for header_term, positions in page_token_dict["header"].items():
                    self.header_index.add_term(term=header_term, doc_id=doc_id, positions=positions)

                for bold_term, positions in page_token_dict["bold"].items():
                    self.bold_index.add_term(term=bold_term, doc_id=doc_id, positions=positions)

                for term, positions in page_token_dict["text"].items():
                    self.limited_index.add_term(term=term, doc_id=doc_id, positions=positions)
                    self.complete_index.add_term(term=term, doc_id=doc_id, positions=positions)

        print()
        print(f"Finished parsing {doc_id} documents")

        print(f"Starting to compute PageRank and initialize anchor index.", end="")
        self.doc_in_edges, self.doc_out_edges = self.build_anchor_index_and_get_page_directed_edges()
        print(".", end="")
        doc_id_page_rankings: [int] = \
            self.compute_page_rank(self.doc_in_edges, self.doc_out_edges, self.page_rank_iterations)
        print(f"Done\n")

        print(f"Merging full index to get global tf-idf scores...")
        self.complete_index.merge_index(doc_count=self.doc_id_counter,
                                        complete_index=None,
                                        doc_page_rankings=doc_id_page_rankings)
        print(f"Done\n")

        print(f"Starting to merge tiered indexes")
        print(f"Merging title index...", end="")
        self.title_index.merge_index(self.doc_id_counter, self.complete_index, doc_id_page_rankings)
        print(f"Done")
        print(f"Merging anchor index...", end="")
        self.anchor_index.merge_index(self.doc_id_counter, None, doc_id_page_rankings)
        print(f"Done")
        print(f"Merging header index...", end="")
        self.header_index.merge_index(self.doc_id_counter, self.complete_index, doc_id_page_rankings)
        print(f"Done")
        print(f"Merging bold index...", end="")
        self.bold_index.merge_index(self.doc_id_counter, self.complete_index, doc_id_page_rankings)
        print(f"Done")
        print(f"Merging limited index...", end="")
        self.limited_index.merge_index(self.doc_id_counter, self.complete_index, doc_id_page_rankings)
        print(f"Done\n")

        print(f"saving current options to json...", end="")
        self.__save_settings_to_json()
        print(f"Done\n")

        print(f"Done Building all Tiered Indexes. "
              f"Found {exact_duplicates_found} exact duplicate documents and "
              f"{near_duplicates_found} near duplicate documents")

        print(f"Saving settings to file...", end="")
        self.__save_settings_to_json()
        print(f"Done")

        print("-" * 120)

    def __add_doc(self, url) -> int:
        self.doc_id_to_url_LUT[self.doc_id_counter] = url
        self.url_to_doc_id_LUT[url] = self.doc_id_counter
        self.doc_id_counter += 1
        return self.doc_id_counter - 1

    def find_near_duplicate_doc(self, doc_simhash: int):

        for doc_id, simhash in self.doc_fingerprints.items():

            similarity = not (doc_simhash ^ simhash)/32

            if similarity > 31/32:
                return doc_id
        return None

    def build_anchor_index_and_get_page_directed_edges(self):

        url_anchor_text_dict: {int: {str: int}} = {}
        doc_out_edges: {int: {int}} = {}
        doc_in_edges: {int: {int}} = {}

        for page_file in self.local_store_path.rglob("*.json"):  # iterate all json files in local store
            # print(f"Opening json file: {page_file}")
            with open(page_file, "r") as page_json:  # open and read the json file

                data = json.load(page_json)
                raw_url, content, encoding = data.values()
                try:
                    url = urllib.parse.urldefrag(raw_url).url
                except ValueError:
                    print(f"Error parsing file {page_file}, url invalid format: {raw_url}")
                    continue

                if url not in self.url_to_doc_id_LUT:
                    continue

                doc_id = self.url_to_doc_id_LUT[url]

                page_links_dict = Tokenizer.get_page_links(content, self.anchor_index.max_n_grams)

                for target_link, term_frequency_dict in page_links_dict.items():
                    try:
                        target_url = urllib.parse.urldefrag(target_link).url

                    except ValueError:
                        continue

                    if target_url not in self.url_to_doc_id_LUT:
                        continue

                    target_doc_id = self.url_to_doc_id_LUT[target_url]

                    doc_in_edges.setdefault(target_doc_id, set())
                    doc_in_edges[target_doc_id].add(doc_id)
                    doc_out_edges.setdefault(doc_id, set())
                    doc_out_edges[doc_id].add(target_doc_id)

                    for term, count in term_frequency_dict.items():
                        url_anchor_text_dict.setdefault(target_doc_id, {})
                        url_anchor_text_dict[target_doc_id].setdefault(term, 0)
                        url_anchor_text_dict[target_doc_id][term] += 1

        for target_doc_id, term_frequency_dict in url_anchor_text_dict.items():
            for term, count in term_frequency_dict.items():
                self.anchor_index.add_term(term, target_doc_id, [None] * count)

        return doc_in_edges, doc_out_edges

    def compute_page_rank(self, doc_in_edges: {int: {int}}, doc_out_edges: {int: {int}}, iterations: int) -> [int]:

        d = 0.85
        page_rank_values: [float] = [1.0 for _ in range(self.doc_id_counter)]

        for _ in range(iterations):
            for doc_id in range(self.doc_id_counter):
                if doc_id not in doc_in_edges:
                    continue

                page_rank_values[doc_id] = \
                    (1 - d) + d * sum(1 / len(doc_out_edges[target_doc_id])
                                      for target_doc_id in doc_in_edges[doc_id])

        return page_rank_values

    def __load_settings_from_json(self):
        with open(Path(self.settings_path.joinpath(self.settings_file_name)), mode="r") as f:
            data_dict = json.load(f)

            self.doc_id_counter = data_dict["doc_id_counter"]
            self.url_to_doc_id_LUT = data_dict["url_to_doc_id_LUT"]
            self.doc_id_to_url_LUT = {int(k): v for k, v in data_dict["doc_id_to_url_LUT"].items()}

            self.doc_in_edges = {int(k): set(v) for k, v in data_dict["doc_in_edges"].items()}
            self.doc_out_edges = {int(k): set(v) for k, v in data_dict["doc_out_edges"].items()}

    def __save_settings_to_json(self):
        with open(Path(self.settings_path.joinpath(self.settings_file_name)), mode="w") as f:
            json_dict = {

                "doc_id_counter": self.doc_id_counter,
                "url_to_doc_id_LUT": self.url_to_doc_id_LUT,
                "doc_id_to_url_LUT": self.doc_id_to_url_LUT,

                "doc_in_edges": {str(k): list(v) for k, v in self.doc_in_edges.items()},
                "doc_out_edges": {str(k): list(v) for k, v in self.doc_out_edges.items()},

            }

            json.dump(json_dict, f)


def crc_hash(content):
    return hash(content)
