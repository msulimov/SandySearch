import time

from Indexer.TieredIndex import TieredIndex
import sys

from Scorer import Scorer

if __name__ == "__main__":
    with TieredIndex(max_n_grams=3, page_rank_iterations=5) as tiered_index:
        tiered_index.build_tiered_indexes()

        scorer: Scorer = Scorer(tiered_index)

        k_results = 10
        query = None

        while True:
            print("-" * 80)
            scorer.new_search()
            if query is None:
                print(f"SandySearch: Enter a search query. !Exit to exit")
                query = input("SandySearch: ")
            if query == "!Exit":
                break
            print(f"Searching...", end="")
            start_time = time.time()
            results = scorer.sprint_search(query, k_results=k_results)
            end_time = time.time()
            duration = round(end_time - start_time, 4)
            print(f"Top Results retrieved in {duration*1000}ms: ")

            if len(results) == 0:
                print("It doesn't look like there were any good results found for your phrase.")
                query = None
                continue
            if 0 < len(results) < k_results:
                print("There weren't many relevant results from your search, try searching more general terms.")

            for i, url in enumerate(results, start=1):
                print(f"\n{i}. {url}")
            print()
            page_number = 2
            while True:

                print(f"Searching for next page results...")
                start_time = time.time()
                results = scorer.complete_search(query, k_results=k_results)
                end_time = time.time()
                duration = round(end_time - start_time, 4)
                print(f"Found page {page_number} results in {duration*1000}ms")

                print(f"Enter !Exit to exit, "
                      f"!Next to display next page results, "
                      f"or another search query to search for something else")
                query = input("SandySearch: ")
                if query != "!Next":
                    break

                print(f"Page {page_number} results: ")

                if 0 < len(results) < k_results:
                    print("There weren't many relevant results from your search, try searching for more general terms.")
                    query = None
                    break

                if len(results) == 0:
                    print("It doesn't look like there were any good results found for your phrase.")
                    query = None
                    break

                for i, url in enumerate(results, start=1):
                    print(f"\n{i}. {url}")

                page_number += 1
                print()
        print("-" * 80)
