import os
import time
from pathlib import Path
import json
from contextlib import ExitStack
from typing import Optional

from Indexer.DocList import PostingsList, intersection
from Tokenizer import tokenize_html, tokenize_query


class IndexBuilder:

    local_store_dir = "./Indexer/Local_Store"
    index_directory = "./Indexer/Tiered_Indexes"
    partial_index_directory = "./Indexer/Partial_Tiered_Indexes"
    settings_file = "./Indexer/index_builder_settings.json"
    temp_index_file_prefix = "partial_index"
    index_file_prefix = "positional_index"
    MAX_PARTIAL_INDEX_POSITIONS = 10000000  # max number of term positions in partial index before dumping to file
    delim = '='

    # TODO store total terms in each document for tf_idf

    def __enter__(self):
        return self

    def __init__(self):

        print(f"Initializing IndexBuilder object...")

        # verify data paths exist
        self.data_path = Path(IndexBuilder.local_store_dir)
        assert self.data_path.exists(), f"Local store path {IndexBuilder.local_store_dir} does not exist"
        assert self.data_path.is_dir(), f"Local store path {IndexBuilder.local_store_dir} not a directory"
        self.partial_index_path = Path(IndexBuilder.partial_index_directory)
        assert self.partial_index_path.exists(), f"Local store path {IndexBuilder.partial_index_directory} does not exist"
        assert self.partial_index_path.is_dir(), f"Local store path {IndexBuilder.partial_index_directory} not a directory"
        self.index_path = Path(IndexBuilder.index_directory)
        assert self.index_path.exists(), f"Data path {IndexBuilder.index_directory} does not exist"
        assert self.index_path.is_dir(), f"Data path {IndexBuilder.index_directory} not a directory"

        self.index_file_name = f"{IndexBuilder.index_file_prefix}.index"
        print(f"Checking if index file: {self.index_file_name} exists in {self.index_path}")

        if Path(self.index_path.joinpath(self.index_file_name)).is_file():
            print(f"Found index file, opening it...", end="")
            self.index_file_open_object = \
                open(self.index_path.joinpath(self.index_file_name), mode="r", encoding="ascii")
            print("Done")
        else:
            print(f"Did not find index file, creating a new one and opening it...", end="")
            with open(self.index_path.joinpath(self.index_file_name), mode="w", encoding="ascii") as f:
                f.write(" ")
            self.index_file_open_object = \
                open(self.index_path.joinpath(self.index_file_name), mode="r", encoding="ascii")
            print("Done")
        print(f"Checked data and index paths exist")

        # positional index stored in index/positional_index.index

        self.index_file_term_LUT = {}  # dict storing term seek positions in index file
        self.index_terms = set()  # set of all the indexed terms collected

        self.partial_index_terms = set()  # set of terms stored through partial index files
        self.partial_index_file_names = []  # list of all the temp index file names generated in order
        self.partial_index_files_term_LUT = {}  # dict storing filename with dict of term positions in file
        self.partial_index_file_counter = 0  # number of partial index files and used for naming them

        self.doc_id_counter = 1  # counter for giving ids to documents, incremented when new doc added
        self.processed_urls = set()  # urls processed in local store documents
        self.doc_id_to_url_LUT = {}  # lut to find url based on doc_id
        self.url_to_doc_id_LUT = {}  # lut to find doc_id based on url

        if Path(IndexBuilder.settings_file).exists():
            print(f"Found settings file, loading settings...", end="")
            self.__load_options_from_json()
            print(f"Done")
        else:
            print(f"Did not find settings file")



        print(f"IndexBuilder Initialization complete")

    def build_index(self):
        """Constructs a partial index distributed over multiple files from documents in local store"""

        print("-" * 120)
        print(f"Starting to build index from data")
        print(f"Clearing old partial_index variables...", end="")
        # reset the vars keeping track of current partial index
        self.partial_index_terms.clear()
        self.partial_index_file_names.clear()
        self.partial_index_files_term_LUT.clear()
        self.partial_index_file_counter = 0
        self.processed_urls.clear()
        self.doc_id_to_url_LUT.clear()
        self.url_to_doc_id_LUT.clear()
        self.doc_id_counter = 0
        print(f"Done\n")

        current_positions_count = 0  # count of how many postings are in the current partial index file
        partial_index = {}  # the current partial index with token as key and DocPosList (postings) as value

        print(f"Starting to parse pages in local store")
        for page_file in self.data_path.rglob("*.json"):  # iterate all json files in local store
            # print(f"Opening json file: {page_file}")
            with open(page_file, "r") as page_json:  # open and read the json file
                data = json.load(page_json)  # load the json data using json.load
                if type(data) is not dict or len(data) != 3:  # fields must be url, content and encoding
                    print(f"Error parsing file {page_file}, json file must have url, content and encoding")
                url, content, encoding = data.values()
                if url is None or url == "":  # field checking
                    print(f"Error parsing file {page_file}, url is not specified")
                if content is None or len(content) == 0:
                    print(f"Error parsing file {page_file}, content empty")
                if encoding is None or len(encoding) == 0:
                    print(f"Error parsing file {page_file}, encoding not specified")

                if url in self.processed_urls:  # skip if url already processed
                    print(
                        f"Already parsed url: {url} which is "
                        f"doc_id: {self.url_to_doc_id_LUT[url]}, skipping document"
                    )
                    continue  # TODO implement exact/near similarity match

                doc_id = self.__add_doc(url)
                print(f"\rParsing doc_id: {doc_id}, url: {url}", end="")
                page_token_dict = tokenize_html(content, encoding)

                for term in page_token_dict:

                    partial_index.setdefault(term, PostingsList())  # create PostingsList for term if not created yet

                    # add posting for doc_id with term positions
                    partial_index[term].create_posting(doc_id, page_token_dict[term])

                    current_positions_count += len(page_token_dict[term])

                    if current_positions_count >= IndexBuilder.MAX_PARTIAL_INDEX_POSITIONS:

                        print(f"\nPreparing to dump partial index with {current_positions_count} positions to file")
                        self.__dump_partial_index(partial_index)  # dump partial index to file and record it
                        current_positions_count = 0
                        partial_index.clear()  # release partial index from memory

                    self.partial_index_terms.add(term)

        # don't forget to dump one last time after all documents processed
        if len(partial_index) > 0:
            self.__dump_partial_index(partial_index)

        print()
        print(f"Finished building partial index from {doc_id} documents "
              f"distributed over {self.partial_index_file_counter} files")
        print(f"saving current options to json...", end="")
        self.__save_options_to_json()
        print(f"Done")
        print("-"*120)

    def merge_index(self):
        """
        Merges the index from the partial index files into one giant index file,
        recording the seek positions of all the terms. Raises ValueError is no partial index files to process
        """
        if len(self.partial_index_file_names) == 0: # raise error if no partial index files to merge
            raise ValueError(f"No partial index files to process!")

        self.index_file_term_LUT.clear()  # reset index tracking vars
        self.index_terms.clear()
        if self.index_file_open_object is not None:  # close current index file if open
            self.index_file_open_object.close()

        self.index_file_open_object = open(self.index_path.joinpath(self.index_file_name),
                                           mode="w", encoding="ascii"
                                           )  # reopen index file

        # inspiration from src: https://stackoverflow.com/questions/29550290/how-to-open-a-list-of-files-in-python
        with ExitStack() as stack:
            partial_index_open_file_objects = [  # safely open each partial index file and store in list
                (  # NOTE: partial index files opened in sequential order so merging is just appending DocPosList
                   # from previous file to next file since they are filled with postings sequentially
                    partial_index_file_name,
                    stack.enter_context(
                        open(self.partial_index_path.joinpath(partial_index_file_name),
                             mode="r", encoding="ascii")
                    )
                 )
                for partial_index_file_name in self.partial_index_file_names
            ]

            # loop over each term in the partial_index, writing line by line for each term from start in index file
            for term in self.partial_index_terms:

                # store the seek position for the term in the index file
                self.index_file_term_LUT[term] = self.index_file_open_object.tell()
                raw_postings_data_merge_list = []  # list of string data of DocPosLists across all partial index files

                # loop over each partial index file and process it if contains the term
                for partial_index_file_path, partial_index_open_file_object in partial_index_open_file_objects:

                    if term not in self.partial_index_files_term_LUT[partial_index_file_path]:
                        continue  # skip the current partial index file if the term is not in it

                    # use lut to grab the seek start pos and data length of the term entry in the partial index file
                    data_start_pos = self.partial_index_files_term_LUT[partial_index_file_path][term]

                    partial_index_open_file_object.seek(data_start_pos)  # seek to start pos
                    data = partial_index_open_file_object.readline().rstrip('\n')  # read data length bytes

                    # split index term and raw postings from the partial index file
                    index_term, partial_index_raw_postings_data = data.split(IndexBuilder.delim)

                    # add raw postings data to merge list for this term
                    raw_postings_data_merge_list.append(partial_index_raw_postings_data)

                # merge raw postings for this term into a single PostingsList
                merged_postings_list = PostingsList(raw_posting_data_list=raw_postings_data_merge_list)
                merged_postings_list.compute_stats(self.doc_count()) # compute stats like tf-idf before storing in index

                # prepare data string for writing the merged Postings Data to the final index for this term
                write_data = f"{term}{IndexBuilder.delim}{merged_postings_list.dump()}\n"

                self.index_file_open_object.write(write_data)  # write the term postings data to the index
                self.index_terms.add(term)  # add the term to the final index terms

        self.__save_options_to_json()

        self.index_file_open_object.close()  # close the file since done writing
        self.index_file_open_object = open(self.index_path.joinpath(self.index_file_name),
                                           mode="r", encoding="ascii"
                                           )  # reopen index file for reading

    def __dump_partial_index(self, partial_index: {str: PostingsList}):
        """
        Dumps the partial index to a new file with term:DocList separated by newlines
        Records seek positions of terms in partial index file for that partial index file, for later merging
        """



        partial_index_term_seek_pos_lut = {}  # term : (data start pos, data length in bytes)

        # filename for the partial index file: index/partial_index0.dump
        partial_index_file_name = f"{IndexBuilder.temp_index_file_prefix}{self.partial_index_file_counter}.dump"
        partial_index_file_path = self.partial_index_path.joinpath(partial_index_file_name)

        print(f"Dumping partial index to {partial_index_file_path} ...", end="")

        # open partial index file for writing in ascii format for fast random access speeds vs. giant utf-32
        with open(partial_index_file_path, mode="w", encoding="ascii") as partial_index_file_open_object:
            for term, doc_pos_list in partial_index.items():  # loop over each term, doc_pos_list data

                # record the seek pos of the term to be written in this partial index file
                partial_index_term_seek_pos_lut[term] = partial_index_file_open_object.tell()

                # prepare the data string to be written, which just has the raw postings data
                partial_index_write_data = f"{term}{IndexBuilder.delim}{doc_pos_list.dump_raw_postings()}\n"
                partial_index_file_open_object.write(partial_index_write_data)  # write data to the partial index file

        self.partial_index_file_names.append(partial_index_file_name)  # record partial index file path sequentially

        # store the term to seek_pos lut for the partial index file
        self.partial_index_files_term_LUT[partial_index_file_name] = partial_index_term_seek_pos_lut
        self.partial_index_file_counter += 1  # increment global partial index file counter
        print("Done")

    def __add_doc(self, url) -> int:
        self.doc_id_to_url_LUT[self.doc_id_counter] = url
        self.url_to_doc_id_LUT[url] = self.doc_id_counter
        self.doc_id_counter += 1
        return self.doc_id_counter - 1

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.index_file_open_object is not None:
            self.index_file_open_object.close()
        print(f"Closed index file.")

    def doc_count(self):
        return self.doc_id_counter

    def print_stats(self):
        """Prints the stats for Milestone 1"""
        print("-" * 80)
        print(f"Total terms in index: {len(self.index_terms)}")
        print(f"Total documents in index: {self.doc_id_counter-1}")
        print(f"Size of index file in kb: "
              f"{os.path.getsize(self.index_path.joinpath(self.index_file_name)) / 1000}")
        print("-" * 80)

    def boolean_search(self, query: str, k=10) -> [str]:

        term_posting_list_dict = {}
        query_terms = tokenize_query(query)

        for term in query_terms:
            posting_list = self.retrieve_posting_list(term)
            if posting_list is not None:
                term_posting_list_dict[term] = self.retrieve_posting_list(term)

        if len(term_posting_list_dict) == 0:
            return []

        valid_doc_ids = intersection(list(term_posting_list_dict.values()), k=k)
        del query_terms
        del term_posting_list_dict
        return [self.doc_id_to_url_LUT[doc_id] for doc_id in valid_doc_ids]

    def retrieve_posting_list(self, term) -> Optional[PostingsList]:
        if term not in self.index_terms:
            return None

        self.index_file_open_object.seek(self.index_file_term_LUT[term])
        index_term, posting_data = self.index_file_open_object.readline().rstrip('\n').split(IndexBuilder.delim)
        assert term == index_term
        return PostingsList(dump_data=posting_data)

    def __save_options_to_json(self):

        json_dict = {

            "index_file_name": self.index_file_name,
            "index_file_term_LUT": self.index_file_term_LUT,
            "index_terms": list(self.index_terms),

            "partial_index_terms": list(self.partial_index_terms),
            "partial_index_file_names": self.partial_index_file_names,
            "partial_index_files_term_LUT": self.partial_index_files_term_LUT,
            "partial_index_file_counter": self.partial_index_file_counter,

            "doc_id_counter": self.doc_id_counter,
            "processed_urls": list(self.processed_urls),
            "doc_id_to_url_LUT": self.doc_id_to_url_LUT,
            "url_to_doc_id_LUT": self.url_to_doc_id_LUT,

        }

        with open(IndexBuilder.settings_file, mode="w") as f:
            json.dump(json_dict, f)

    def __load_options_from_json(self):
        with open(IndexBuilder.settings_file, mode="r") as f:
            data_dict = json.load(f)

            self.index_file_name = data_dict["index_file_name"]
            self.index_file_term_LUT = data_dict["index_file_term_LUT"]
            self.index_terms = set(data_dict["index_terms"])

            self.partial_index_terms = set(data_dict["partial_index_terms"])
            self.partial_index_file_names = data_dict["partial_index_file_names"]
            self.partial_index_files_term_LUT = data_dict["partial_index_files_term_LUT"]
            self.partial_index_file_counter = data_dict["partial_index_file_counter"]

            self.doc_id_counter = data_dict["doc_id_counter"]
            self.processed_urls = set(data_dict["processed_urls"])
            self.doc_id_to_url_LUT = {int(k): v for k, v in data_dict["doc_id_to_url_LUT"].items()}
            self.url_to_doc_id_LUT = data_dict["url_to_doc_id_LUT"]
