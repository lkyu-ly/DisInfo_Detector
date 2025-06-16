import json
import os

from get_response import GetResponse


class ClaimVerifier():
    def __init__(self, model_name, label_n=2, cache_dir="./cache/", demon_dir="./demos/"):
        self.model = None
        self.model_name = model_name
        self.label_n = label_n
        self.system_message = None
        self.evaluator = None
        
        if os.path.isdir(model_name):
            from unsloth import FastLanguageModel

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
            self.tokenizer.padding_side = "left"
            # self.alpaca_prompt = open("./prompt/verification_alpaca_template.txt", "r").read()
            # self.instruction = open("./prompt/verification_instruction_binary_no_demo.txt", "r").read()
        else:
            cache_dir = os.path.join(cache_dir, model_name)
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, "claim_verification_cache.json")
            self.demon_path = os.path.join(demon_dir, 'few_shot_examples.jsonl')
            self.get_model_response = GetResponse(cache_file=self.cache_file,
                                                  model_name=model_name,
                                                  max_tokens=1000,
                                                  temperature=0,
                                                  save_interval=20)
            self.system_message = "You are a helpful assistant who can verify the truthfulness of a claim against reliable external world knowledge."
            self.prompt_initial_temp = self.get_initial_prompt_template()

    def get_instruction_template(self):
        prompt_temp = ''
        if self.label_n == 2:
            prompt_temp = open("./prompt/verification_instruction_binary_nli.txt", "r").read()
        elif self.label_n == 3:
            prompt_temp = open("./prompt/verification_instruction_trinary.txt", "r").read()
        else:
            raise ValueError(f"Label number {self.label_n} is not supported.")
        return prompt_temp

    def get_initial_prompt_template(self):
        prompt_temp = self.get_instruction_template()
        with open(self.demon_path, "r") as f:
            example_data = [json.loads(line) for line in f if line.strip()]
        element_lst = []
        for dict_item in example_data:
            claim = dict_item["claim"]
            search_result_str = dict_item["search_result"]
            human_label = dict_item["human_label"]
            if self.label_n == 2:
                if human_label.lower() in ["support.", "supported.", "supported", "support"]:
                    human_label = "Supported."
                else:
                    human_label = "Unsupported."
            element_lst.extend([claim, search_result_str, human_label])

        prompt_few_shot = prompt_temp.format(*element_lst)

        self.your_task = "Your task:\n\n{search_results}\n\nClaim: {claim}\n\nTask: Given the search results above, is the claim supported or unsupported? If supported, return 'Supported'. If not, return 'Unsupported'. For your decision, you must only output the word 'Supported', or the word 'Unsupported', nothing else.\n\nYour decision:"

        if self.label_n == 2:
            self.system_message = "You are a helpful assistant who can judge whether a claim is supported by the search results or not."
        elif self.label_n == 3:
            self.system_message = "You are a helpful assistant who can judge whether a claim is supported or contradicted by the search results, or whether there is no enough information to make a judgement."

        return prompt_few_shot

    def verifying_claim(self, claim_snippets_dict, search_res_num=5, cost_estimate_only=False):
        """
        search_snippet_lst = [{"title": title, "snippet": snippet, "link": link}, ...]
        """
        prompt_tok_cnt, response_tok_cnt = 0, 0
        claim_verify_res_dict = {}
        for claim, search_snippet_lst in claim_snippets_dict.items():
            search_res_str = ""
            search_cnt = 1

            for search_dict in search_snippet_lst[:search_res_num]:
                if 'title' and 'link' in search_dict:
                    search_res_str += f'Search result {search_cnt}\nTitle: {search_dict["title"].strip()}\nLink: {search_dict["link"].strip()}\nContent: {search_dict["snippet"].strip()}\n\n'
                else:
                    search_res_str += f'Search result {search_cnt}\nContent: {search_dict["snippet"].strip()}\n\n'
                search_cnt += 1
            
            if self.model:
                usr_input = f"Claim: {claim.strip()}\n\n{search_res_str.strip()}"
                formatted_input = self.alpaca_prompt.format(self.instruction, usr_input)
                prompt = formatted_input
                inputs = self.tokenizer(formatted_input, return_tensors="pt").to("cuda")
                output = self.model.generate(**inputs,
                                            max_new_tokens=500,
                                            use_cache=True,
                                            eos_token_id=[self.tokenizer.eos_token_id,
                                                        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                                            pad_token_id=self.tokenizer.eos_token_id, )
                response = self.tokenizer.batch_decode(output)
                clean_output = ' '.join(response).split("<|end_header_id|>\n\n")[
                    -1].replace("<|eot_id|>", "").strip()
            else:
                prompt_tail = self.your_task.format(
                    claim=claim,
                    search_results=search_res_str.strip(),
                )
                prompt = f"{self.prompt_initial_temp}\n\n{prompt_tail}"
                response, prompt_tok_num, response_tok_num = self.get_model_response.get_response(self.system_message,
                                                                                                prompt,
                                                                                                cost_estimate_only=cost_estimate_only)
                prompt_tok_cnt += prompt_tok_num
                response_tok_cnt += response_tok_num

                clean_output = response.replace("#", "").split(".")[0].lower() if response is not None else None
                
            claim_verify_res_dict[claim] = clean_output

        return claim_verify_res_dict, prompt_tok_cnt, response_tok_cnt