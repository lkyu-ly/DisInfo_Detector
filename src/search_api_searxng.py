# search_api_searxng.py

import json
import os
import requests
from urllib.parse import urlencode


class SearchAPISearxng:
    """
    A search API class to interact with a local Searxng instance,
    designed to be a drop-in replacement for the original SearchAPI.
    """

    def __init__(self, searxng_url="http://localhost:8080"):
        # The base URL for your local searxng instance.
        self.url = searxng_url

        # Cache setup for searxng results to avoid repeated queries.
        self.cache_file = "cache/search_cache_searxng.json"
        self.cache_dict = self.load_cache()
        self.add_n = 0
        self.save_interval = 10
        print("Initialized SearchAPISearxng to use local searxng instance.")

    def get_snippets(self, claim_lst):
        """
        Takes a list of claims, searches them using searxng, and returns
        the results in a format compatible with the rest of the pipeline.
        """
        text_claim_snippets_dict = {}
        for query in claim_lst:
            search_result = self.get_search_res(query)

            # Handle cases where the search fails (e.g., connection error)
            if not search_result:
                text_claim_snippets_dict[query] = []
                continue

            # Extract results from the 'results' key in the searxng JSON output.
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
        Implements caching to speed up repeated searches.
        """
        cache_key = query.strip()
        if cache_key in self.cache_dict:
            return self.cache_dict[cache_key]

        # Construct the query as per your curl example.
        # The '!google' prefix specifies the engine within searxng.
        params = {"q": f"!google {query}", "format": "json"}

        try:
            # The 'verify=False' argument mimics the 'curl -k' flag.
            response = requests.get(self.url, params=params, verify=False, timeout=10)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            response_json = response.json()
        except requests.exceptions.RequestException as e:
            print(
                f"Error connecting to Searxng instance at {self.url}. Please ensure it is running. Details: {e}"
            )
            return None  # Return None to indicate failure

        # Update and save cache
        self.cache_dict[query.strip()] = response_json
        self.add_n += 1
        if self.add_n % self.save_interval == 0:
            self.save_cache()

        return response_json

    def save_cache(self):
        """Saves the current cache to a file."""
        # Ensure the latest cache is merged before saving.
        for k, v in self.load_cache().items():
            if k not in self.cache_dict:
                self.cache_dict[k] = v
        print(f"Saving searxng search cache to {self.cache_file}...")
        with open(self.cache_file, "w") as f:
            json.dump(self.cache_dict, f, indent=4)

    def load_cache(self):
        """Loads the cache from a file if it exists."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                try:
                    cache = json.load(f)
                    print(f"Loading searxng cache from {self.cache_file}...")
                except json.JSONDecodeError:
                    print(
                        f"Warning: Could not decode JSON from {self.cache_file}. Starting with an empty cache."
                    )
                    cache = {}
        else:
            cache = {}
            # Ensure the cache directory exists
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        return cache
