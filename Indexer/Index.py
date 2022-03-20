import os
from pathlib import Path
import json
from contextlib import ExitStack
from typing import Optional

from Indexer.DocList import PostingsList


class Index:

    index_directory = "./Indexer/Tiered_Indexes"
    partial_index_directory = "./Indexer/Partial_Tiered_Indexes"
    settings_directory = "./Indexer/Tiered_Indexes_Settings"
    MAX_PARTIAL_INDEX_POSITIONS = 5000000  # max number of term positions in partial index before dumping to file
    delim = '='

    def __enter__(self):
        return self

    def __init__(self,
                 descriptor: str,
                 max_n_gram: int,
                 sort_weights: {str: float},
                 postings_list_size_limit: Optional[int],
                 store_positions: bool,
                 ):

        print(f"Initializing {descriptor.capitalize()} Index object...")

        self.descriptor = descriptor
        self.max_n_grams: int = max_n_gram
        self.sort_weights: {str: float} = sort_weights
        self.postings_list_size_limit: int = postings_list_size_limit
        self.store_positions: bool = store_positions

        self.settings_file_name: str = f"{self.descriptor}_settings.json"
        self.temp_index_file_prefix: str = f"partial_{self.descriptor}"
        self.index_file_prefix: str = f"{'positional_' if self.store_positions else ''}{self.descriptor}"

        self.index_file_term_LUT: {str: int} = {}  # dict storing term seek positions in index file
        self.document_term_counts: {str: int} = {}  # set of all the indexed terms collected

        self.current_positions_count = 0

        self.partial_index: {str: PostingsList} = {}
        self.partial_index_terms: {str} = set()  # set of terms stored through partial index files
        self.partial_index_file_names: [str] = []  # list of all the temp index file names generated in order
        self.partial_index_files_term_LUT: {str: {str: int}} = {}  # dict storing filename with dict of term positions
        self.partial_index_file_counter: int = 0  # number of partial index files and used for naming them

        # verify data paths exist
        self.partial_index_path: Path = Path(Index.partial_index_directory)
        assert self.partial_index_path.exists(), f"Partial Index path {Index.partial_index_directory} does not exist"
        assert self.partial_index_path.is_dir(), f"Partial Index path {Index.partial_index_directory} not a directory"
        self.index_path: Path = Path(Index.index_directory)
        assert self.index_path.exists(), f"Data path {Index.index_directory} does not exist"
        assert self.index_path.is_dir(), f"Data path {Index.index_directory} not a directory"
        self.settings_path: Path = Path(Index.settings_directory)
        assert self.settings_path.exists(), f"Settings path {Index.settings_directory} does not exist"
        assert self.settings_path.is_dir(), f"Settings path {Index.settings_directory} not a directory"

        self.index_file_name: str = f"{self.index_file_prefix}.index"
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

        if Path(self.settings_path.joinpath(self.settings_file_name)).is_file():
            print(f"Found settings file, loading settings...", end="")
            self.__load_settings_from_json()
            print(f"Done")
        else:
            print(f"Did not find settings file")

        print(f"{self.descriptor.capitalize()} Index Initialization complete")

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.index_file_open_object is not None:
            self.index_file_open_object.close()
        print(f"Closed {self.descriptor} file.")

    def __contains__(self, key: str):
        return key in self.document_term_counts

    def prep_for_build(self):

        self.partial_index_terms.clear()
        self.partial_index_file_names.clear()
        self.partial_index_files_term_LUT.clear()
        self.partial_index_file_counter = 0
        self.current_positions_count = 0

    def add_term(self, term: str, doc_id: int, positions: [int]) -> bool:

        dumped = False
        self.partial_index.setdefault(term, PostingsList(store_positions=self.store_positions))
        self.partial_index[term].create_posting(doc_id, positions)
        self.current_positions_count += len(positions) if self.store_positions else 1

        if self.current_positions_count >= Index.MAX_PARTIAL_INDEX_POSITIONS:
            # print(f"\nPreparing to dump partial index with {current_positions_count} positions to file")
            self.__dump_partial_index(self.partial_index)  # dump partial index to file and record it
            self.current_positions_count = 0
            self.partial_index.clear()  # release partial index from memory
            dumped = True

        self.partial_index_terms.add(term)
        return dumped

    def merge_index(self, doc_count: int, complete_index: Optional['Index'], doc_page_rankings: [int]):
        """
            Merges the index from the partial index files into one giant index file,
            recording the seek positions of all the terms. Raises ValueError is no partial index files to process
        """

        if self.current_positions_count >= 0:
            self.__dump_partial_index(self.partial_index)
            self.partial_index.clear()
            self.current_positions_count = 0

        if len(self.partial_index_file_names) == 0:  # raise error if no partial index files to merge
            raise ValueError(f"No partial index files to process!")

        self.index_file_term_LUT.clear()  # reset index tracking vars
        self.document_term_counts.clear()
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
                    index_term, partial_index_raw_postings_data = data.split(Index.delim)

                    # add raw postings data to merge list for this term
                    raw_postings_data_merge_list.append(partial_index_raw_postings_data)

                # merge raw postings for this term into a single PostingsList
                merged_postings_list = PostingsList(store_positions=self.store_positions,
                                                    raw_posting_data_list=raw_postings_data_merge_list)

                if complete_index is None:
                    merged_postings_list.compute_local_tf_idf(doc_count, copy_to_global=True)
                else:
                    merged_postings_list.compute_local_tf_idf(doc_count, copy_to_global=False)
                    assert term in complete_index.document_term_counts
                    merged_postings_list.add_global_tf_idf(complete_index.retrieve_posting_list(term))
                merged_postings_list.set_page_rankings(doc_page_rankings)

                merged_postings_list.sort(self.sort_weights["page_rank"],
                                          self.sort_weights["local_tf_idf"],
                                          self.sort_weights["global_tf_idf"],
                                          )

                if self.postings_list_size_limit is not None:
                    merged_postings_list.limit(self.postings_list_size_limit)

                # prepare data string for writing the merged Postings Data to the final index for this term
                write_data = f"{term}{Index.delim}{merged_postings_list.dump()}\n"

                self.index_file_open_object.write(write_data)  # write the term postings data to the index

                # store document frequency of term in memory to avoid having to read data from disk
                self.document_term_counts[term] = len(merged_postings_list)

        self.__save_settings_to_json()
        self.__load_settings_from_json()
        self.__save_settings_to_json()

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
        partial_index_file_name = f"{self.temp_index_file_prefix}{self.partial_index_file_counter}.dump"
        partial_index_file_path = self.partial_index_path.joinpath(partial_index_file_name)

        # open partial index file for writing in ascii format for fast random access speeds vs. giant utf-32
        with open(partial_index_file_path, mode="w", encoding="ascii") as partial_index_file_open_object:
            for term, doc_pos_list in partial_index.items():  # loop over each term, doc_pos_list data

                # record the seek pos of the term to be written in this partial index file
                partial_index_term_seek_pos_lut[term] = partial_index_file_open_object.tell()

                # prepare the data string to be written, which just has the raw postings data
                partial_index_write_data = f"{term}{Index.delim}{doc_pos_list.dump_raw_postings()}\n"
                partial_index_file_open_object.write(partial_index_write_data)  # write data to the partial index file

        self.partial_index_file_names.append(partial_index_file_name)  # record partial index file path sequentially

        # store the term to seek_pos lut for the partial index file
        self.partial_index_files_term_LUT[partial_index_file_name] = partial_index_term_seek_pos_lut
        self.partial_index_file_counter += 1  # increment global partial index file counter

    def retrieve_posting_list(self, term) -> Optional[PostingsList]:
        if term not in self.document_term_counts:
            return None

        self.index_file_open_object.seek(self.index_file_term_LUT[term])
        index_term, posting_data = self.index_file_open_object.readline().rstrip('\n').split(Index.delim)
        assert term == index_term
        return PostingsList(self.store_positions, dump_data=posting_data)

    def __load_settings_from_json(self):
        with open(Path(self.settings_path.joinpath(self.settings_file_name)), mode="r") as f:
            data_dict = json.load(f)

            self.index_file_name = data_dict["index_file_name"]
            self.index_file_term_LUT = data_dict["index_file_term_LUT"]
            self.document_term_counts = data_dict["document_term_counts"]

            self.partial_index_terms = set(data_dict["partial_index_terms"])
            self.partial_index_file_names = data_dict["partial_index_file_names"]
            self.partial_index_files_term_LUT = data_dict["partial_index_files_term_LUT"]
            self.partial_index_file_counter = data_dict["partial_index_file_counter"]

    def __save_settings_to_json(self):

        with open(Path(self.settings_path.joinpath(self.settings_file_name)), mode="w") as f:

            json_dict = {

                "index_file_name": self.index_file_name,
                "index_file_term_LUT": self.index_file_term_LUT,
                "document_term_counts": self.document_term_counts,

                "partial_index_terms": list(self.partial_index_terms),
                "partial_index_file_names": self.partial_index_file_names,
                "partial_index_files_term_LUT": self.partial_index_files_term_LUT,
                "partial_index_file_counter": self.partial_index_file_counter,

            }

            json.dump(json_dict, f)
