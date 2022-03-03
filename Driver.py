import time

from Indexer.TieredIndexBuilder import IndexBuilder
import sys

if __name__ == "__main__":
    with IndexBuilder() as index:
        # index.build_index()
        # index.merge_index()
        index.print_stats()

        while True:
            print(f"Enter a search query for boolean retrieval. !Exit to exit")
            query = input("Search: ")
            if query == "!Exit":
                break
            print(f"Searching...", end="")
            start_time = time.time()
            results = index.boolean_search(query, k=10)

            end_time = time.time()
            duration = round(end_time - start_time, 4)
            print(f"Top Results retrieved in {duration*1000}ms: ")
            for i, url in enumerate(results, start=1):
                print(f"\n{i}. {url}")
            if len(results) == 0:
                print("It doesn't look like there were any good results found for your phrase.")
            print("-" * 80)
