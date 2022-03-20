# SandySearch
A free form search engine that can search through 56,000+ 
documents under harsh operating constraints while maintaining 
a response time below 100ms.  
Utilizes an inverted multi-tiered index comprised of 6 different
indexes that index document titles, anchors, headers, important 
text, limited text and full text. None of the indexes are ever 
fully loaded into limited memory to allow scalability to the web.
Searches for the next page's results from the full index while 
the user looks over the current page's results.

## Features
PageRank  
Inverted Multi-tiered Index  
N-gram Indexing  
Indexing Anchor text  
TF-IDF with cosine similarity  
Indexing term positions


## Requirements:
Python 3.7+
### Packages:
bs4  
lxml  
KrovetzStemmer  

## How to Use:

### Fill the Local store: Indexer/Local_Store
Stores the physical documents representing the pages crawled on the web.
Each document is stored in **JSON** format. The JSON file contains 
a dict with three keys:
#####
"url" : "url string"
#####
"content": "html of page"
#####
"encoding": "encoding of the page"
####

Be sure to fill the Local Store with all the documents to search over 
before building the multi-tiered index.

### Running

Run the driver.py file. The index will start building from all the documents
in the document store. This can take a while if there are many documents. Once
the index has finished building, you will be prompted for a search query where
you can use the command "!Exit" to exit or "!Next" to get the next page's results.  
Since the settings and indexes are stored on the hard disk, you can skip re-building
of the multi-tiered index by commenting out "tiered_index.build_tiered_indexes()" in 
the main method.
