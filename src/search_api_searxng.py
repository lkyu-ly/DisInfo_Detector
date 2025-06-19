# search_api_searxng.py

import json
import os
import requests
import time
import threading
from urllib.parse import urlencode


class SearchAPISearxng:
    """
    A search API class to interact with a local Searxng instance,
    designed to be a drop-in replacement for the original SearchAPI.
    This version is thread-safe.
    """

    def __init__(self, searxng_url="http://localhost:8080"):
        # The base URL for your local searxng instance.
        self.url = searxng_url

        # Cache setup for searxng results to avoid repeated queries.
        self.cache_file = os.path.join("cache", "search_cache_searxng.json")
        self.cache_dict = self.load_cache()

        # Thread-safety mechanism
        self.cache_lock = threading.RLock()  # Use a re-entrant lock
        self.add_n = 0
        self.save_interval = 10
        print("Initialized SearchAPISearxng to use local searxng instance.")

    def get_snippets(self, claim_lst):
        """
        Takes a list of claims, searches them using searxng, and returns
        the results in a format compatible with the rest of the pipeline.
        This method can be safely called from multiple threads.
        """
        text_claim_snippets_dict = {}
        for query in claim_lst:
            search_result = self.get_search_res(query)

            # Extract results from the 'results' key of the searxng JSON output.
            organic_res = search_result.get("results", [])

            search_res_lst = []
            for item in organic_res:
                # Map searxng fields to the required fields by the pipeline.
                title = item.get("title", "")
                snippet = item.get("content", "")  # searxng 'content' -> 'snippet'
                link = item.get("url", "")  # searxng 'url' -> 'link'

                search_res_lst.append(
                    {"title": title, "snippet": snippet, "link": link}
                )
            text_claim_snippets_dict[query] = search_res_lst
        return text_claim_snippets_dict

    def get_search_res(self, query):
        """
        Queries the local searxng instance for a given query string.
        Implements caching and an unlimited retry loop in a thread-safe manner.
        """
        cache_key = query.strip()

        # --- First, check the cache in a thread-safe way ---
        with self.cache_lock:
            if cache_key in self.cache_dict:
                return self.cache_dict[cache_key]

        # --- If not in cache, perform the network request OUTSIDE the lock ---
        # This allows other threads to perform their own requests concurrently.
        # The '!google' prefix specifies the engine within searxng.
        params = {"q": f"!google {query}", "format": "json", "language": "en"}

        response_json = None
        while True:
            try:
                response = requests.get(
                    self.url, params=params, verify=False, timeout=15
                )
                response.raise_for_status()  # Raise HTTPError for bad responses
                response_json = response.json()

                # Check if search engine was unresponsive
                unresponsive = response_json.get("unresponsive_engines", [])
                if not unresponsive:
                    break  # Success, exit the loop

                # engine error, retry in 10s
                print(
                    f"Query '{query}' failed due to unresponsive engines: {unresponsive}. Retrying in 10s..."
                )
                time.sleep(10)

            except requests.exceptions.RequestException as e:
                # This 'except' block handles cases where the searxng instance itself is down.
                print(
                    f"Could not connect to Searxng for query '{query}': {e}. Retrying in 15s..."
                )
                time.sleep(15)  # Wait longer if the whole service is down
                continue  # Retry the connection attempt

        # If Google search returned empty results, retry with Bing
        if not response_json.get("results"):
            print(
                f"Query '{query}' with !google returned empty results. Retrying with !bing."
            )
            params = {"q": f"!bing {query}", "format": "json", "language": "en"}
            while True:
                try:
                    response = requests.get(
                        self.url, params=params, verify=False, timeout=15
                    )
                    response.raise_for_status()
                    response_json = response.json()

                    unresponsive = response_json.get("unresponsive_engines", [])
                    if not unresponsive:
                        break  # Success

                    print(
                        f"Query '{query}' with !bing failed due to unresponsive engines: {unresponsive}. Retrying in 10s..."
                    )
                    time.sleep(10)

                except requests.exceptions.RequestException as e:
                    print(
                        f"Could not connect to Searxng for query '{query}' with !bing: {e}. Retrying in 15s..."
                    )
                    time.sleep(15)
                    continue

        # --- After a successful query, update and save the cache under a lock ---
        # Only cache if the final result has content
        if response_json and response_json.get("results"):
            with self.cache_lock:
                self.cache_dict[cache_key] = response_json
                self.add_n += 1
                if self.add_n % self.save_interval == 0:
                    # This call is safe because we are using an RLock
                    self.save_cache()

        return response_json

    def save_cache(self):
        """
        Saves the current cache dictionary to a file in a thread-safe manner.
        """
        with self.cache_lock:
            print(f"Saving searxng search cache...")
            # Ensure the directory exists before writing
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

            # Create a copy of the dictionary to ensure consistency during the dump,
            # even though the lock provides strong protection.
            cache_to_save = self.cache_dict.copy()

            with open(self.cache_file, "w") as f:
                json.dump(cache_to_save, f, indent=4)

    def load_cache(self):
        """Loads the cache from a file if it exists."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                try:
                    print(f"Loading searxng cache from {self.cache_file}...")
                    return json.load(f)
                except json.JSONDecodeError:
                    print(
                        f"Warning: Could not decode JSON from {self.cache_file}. Starting with an empty cache."
                    )
                    return {}
        else:
            return {}
