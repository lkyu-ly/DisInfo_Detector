import json
import os
import threading

import tiktoken
from openai import OpenAI


class GetResponse:
    # OpenAI model list: https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    # Claude3 model list: https://www.anthropic.com/claude
    # "claude-3-opus-20240229", gpt-4-0125-preview
    def __init__(
        self,
        cache_file,
        model_name="gpt-4-0125-preview",
        max_tokens=1000,
        temperature=0,
        stop_tokens=None,
        save_interval=5,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_file = cache_file
        self.openai_api_base = os.getenv("OPENAI_API_BASE")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.stop_tokens = stop_tokens

        # Thread-safety mechanism
        self.cache_lock = threading.RLock()

        # invariant variables
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_api_base)
        self.seed = 114115

        # cache related
        self.cache_dict = self.load_cache()
        self.add_n = 0
        self.save_interval = save_interval
        self.print_interval = 20

    # Returns the number of tokens in a text string.
    def tok_count(self, text: str) -> int:
        num_tokens = len(self.tokenizer.encode(text))
        return num_tokens

    def save_cache(self):
        with self.cache_lock:
            # In a multi-threaded environment, it's unsafe to reload the cache before writing.
            # The lock ensures that the current state of self.cache_dict is written atomically.
            with open(self.cache_file, "w") as f:
                json.dump(self.cache_dict, f, indent=4)

    def load_cache(self):
        with self.cache_lock:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    # load a json file
                    cache = json.load(f)
                    print(f"Loading cache from {self.cache_file}...")
            else:
                cache = {}
        return cache

    # Returns the response from the model given a system message and a prompt text.
    def get_response(self, system_message, prompt_text, cost_estimate_only=False):
        prompt_tokens = len(self.tokenizer.encode(prompt_text))
        if cost_estimate_only:
            # count tokens in prompt and response
            response_tokens = 0
            return None, prompt_tokens, response_tokens

        # check if prompt is in cache; if so, return from cache (thread-safe)
        cache_key = prompt_text.strip()
        with self.cache_lock:
            if cache_key in self.cache_dict:
                return self.cache_dict[cache_key], 0, 0

        # --- If not in cache, perform the network request OUTSIDE the lock ---
        if system_message == "":
            message = [{"role": "user", "content": prompt_text}]
        else:
            message = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt_text},
            ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
        )
        response_content = response.choices[0].message.content.strip()

        # --- After a successful query, update and save the cache under a lock ---
        with self.cache_lock:
            self.cache_dict[cache_key] = response_content.strip()
            self.add_n += 1

            # save cache every save_interval times
            if self.add_n % self.save_interval == 0:
                self.save_cache()
            if self.add_n % self.print_interval == 0:
                print(f"Saving # {self.add_n} cache to {self.cache_file}...")

        response_tokens = len(self.tokenizer.encode(response_content))
        return response_content, prompt_tokens, response_tokens
