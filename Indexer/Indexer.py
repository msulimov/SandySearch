from pathlib import Path

from Indexer.DocList import DocList


class Indexer:

    index_file_prefix = "index"

    inverted_index = {}  # term : index file containing postings
    index_file_sizes = {}  # filename : number of postings
    doc_id_to_url_LUT = {}  # docID : url
    url_to_doc_id_LUT = {}

    doc_id_counter = 0

    # Term_ID_1:Frequency_N:Doc_ID_1,Doc_ID_2,Doc_ID_N.Term_ID_2:Frequency_N:Doc_ID_1,Doc_ID_2,Doc_ID_N.Term_ID_N:Frequency_N:Doc_ID_1,Doc_ID_2,Doc_ID_N

    def __init__(self, data_directory, index_directory):

        self.data_path = Path(data_directory)
        assert self.data_path.exists(), f"Data path {data_directory} does not exist"
        assert self.data_path.is_dir(), f"Data path {data_directory} not a directory"
        self.index_path = Path(index_directory)
        assert self.index_path.exists(), f"Data path {index_directory} does not exist"
        assert self.index_path.is_dir(), f"Data path {index_directory} not a directory"

    def index(self):
        pass

    def __add_doc(self, url: str):
        Indexer.doc_id_to_url_LUT[Indexer.doc_id_counter] = url
        Indexer.url_to_doc_id_LUT[url] = Indexer.doc_id_counter
        Indexer.doc_id_counter += 1

    def __add_token_to_index(self, token:str):
        Indexer.inverted_index[token] = DocList(token)