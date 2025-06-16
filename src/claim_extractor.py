import os
import re
import regex
import spacy

from get_response import GetResponse
from factscore.atomic_facts import AtomicFactGenerator
from prompts import DECOMPOSE_NUM_PROMPT_TEMPLATE, DECOMPOSE_SINGLE_PROMPT_TEMPLATE, EXTRACTION_NON_QA_TEMPLATE, WICE_PROMPT


class ClaimExtractor():
    def __init__(self, model_name, cache_dir="./cache/"):
        self.model = None
        self.model_name = model_name
        self.cache_dir = cache_dir
        if os.path.isdir(model_name):
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=1024,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
            self.model = self.model.to("cuda")
            self.alpaca_prompt = open("./prompt/extraction_alpaca_template.txt", "r").read()
        else:
            cache_dir = os.path.join(cache_dir, model_name)
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, f"claim_extraction_cache.json")
            self.get_model_response = GetResponse(cache_file=self.cache_file,
                                                  model_name=model_name,
                                                  max_tokens=2048,
                                                  temperature=0)
            self.system_message = "You are a helpful assistant who can extract verifiable atomic claims from a piece of text. Each atomic fact should be verifiable against reliable external world knowledge (e.g., via Google Search, Wikipedia, etc.)"

        self.spacy_nlp = spacy.load('en_core_web_sm')

    def fact_extractor(self, snippet, sentence, cost_estimate_only=False):
        if self.model:
            formatted_input = self.alpaca_prompt.format(snippet, "")
            inputs = self.tokenizer(formatted_input, return_tensors="pt").to("cuda")

            outputs = self.model.generate(**inputs, max_new_tokens=1000, use_cache=True)
            output_str = ' '.join(self.tokenizer.batch_decode(outputs))
            clean_output = output_str.split("### Response:")[-1].strip().replace("</s>", "")
            if not clean_output or "No verifiable claim." in clean_output:
                return None, 0, 0
            claims = [x.strip() for x in clean_output.split("\n")]
            return claims, 0, 0
        else:
            prompt_template = EXTRACTION_NON_QA_TEMPLATE
            prompt_text = prompt_template.format(snippet=snippet, sentence=sentence)
            # print("Awating for response...")
            response, prompt_tok_cnt, response_tok_cnt, response_logprobs = self.get_model_response.get_response(self.system_message,
                                                                                                                       prompt_text,
                                                                                                                       cost_estimate_only)
            # print("response: ", response)
            if not response or "No verifiable claim." in response:
                return None, prompt_tok_cnt, response_tok_cnt
            else:
                claims = [x.strip().replace("- ", "") for x in response.split("\n")]
                claims = [regex.sub(r"^\d+\.?\s", "", x) for x in claims]
                return claims, prompt_tok_cnt, response_tok_cnt

    def veriscore_extractor(self, response, cost_estimate_only=False):
        """
        Given a model output
        - split the response into sentences using spaCy
        - snippet = (context1 = 0-3 sentence) <SOS>Sent<EOS> (context2 = 0-1 sentence)
        - call fact_extractor on each snippet
        """
        sentences = [s.text.strip() for s in self.spacy_nlp(response).sents]

        all_facts_lst = []
        # keep track of token counts
        prompt_tok_cnt, response_tok_cnt = 0, 0

        # new return values
        snippet_lst = []
        fact_lst_lst = []

        for i, sentence in enumerate(sentences):
            if self.model:
                input = response.strip()
                snippet = input.replace(sentence, f"<SOS>{sentence}<EOS>")
            else:
                lead_sent = sentences[0]  # 1st sentence of the para
                context1 = " ".join(sentences[max(0, i - 3):i])
                sentence = f"<SOS>{sentences[i].strip()}<EOS>"
                context2 = " ".join(sentences[i + 1:i + 2])

                # if the para is not long
                if len(sentences) <= 5:
                    snippet = f"{context1.strip()} {sentence.strip()} {context2.strip()}".strip()
                # if the para is long, add lead sentence to context1
                else:
                    snippet = f"{lead_sent.strip()} {context1.strip()} {sentence.strip()} {context2.strip()}".strip()
            snippet_lst.append(snippet)
            
            facts, prompt_tok_num, response_tok_num = self.fact_extractor(snippet, sentences[i].strip(), cost_estimate_only=cost_estimate_only)
            
            prompt_tok_cnt += prompt_tok_num
            response_tok_cnt += response_tok_num

            if facts is None:
                fact_lst_lst.append([None])
                continue

            fact_lst = []
            for fact in facts:
                if fact.strip() == "":
                    continue
                elif fact.startswith("Note:"):
                    continue
                elif fact.strip() not in all_facts_lst:
                    all_facts_lst.append(fact.strip())
                fact_lst.append(fact.strip())
            fact_lst_lst.append(fact_lst)

        if len(all_facts_lst) == 0:
            all_facts_lst = [response.strip()]
            fact_lst_lst = [[response.strip()]]

        print(f"Returning facts and token counts for the whole response ...")
        return fact_lst_lst, all_facts_lst, prompt_tok_cnt, response_tok_cnt
    


    def factscore_extractor(self, response, cost_estimate_only=False):
        """
        Given a model output
        - split the response into sentences using spaCy
        - snippet = (context1 = 0-3 sentence) <SOS>Sent<EOS> (context2 = 0-1 sentence)
        - call fact_extractor on each snippet
        """

        generator = AtomicFactGenerator(cache_dir=self.cache_dir, 
                                model_name=self.model_name, 
                                is_bio=False)
        atomic_facts, para_breaks = generator.run(response, cost_estimate=False)
        fact_lst_lst = [item[1] for item in atomic_facts]
        
        # sentences = self.get_sentence(response)

        all_facts_lst = []
        for fact_lst in fact_lst_lst:
            for fact in fact_lst:
                if fact.strip() == "":
                    continue
                if fact.strip().endswith(':'):
                    continue
                # cases where GPT returns its justification
                elif fact.startswith("Note:"):
                    continue
                elif fact.strip() not in all_facts_lst:
                    all_facts_lst.append(fact.strip())

        # keep track of token counts
        prompt_tok_cnt, response_tok_cnt = 0, 0
        
        if len(all_facts_lst) == 0:
            # If no facts are extracted, just use the original response as the only fact
            all_facts_lst = [response.strip()]
            fact_lst_lst = [[response.strip()]]

        print(f"Returning facts from FACTSCORE and token counts (IGNORE THIS token counts) for the whole response ...")
        return fact_lst_lst, all_facts_lst, prompt_tok_cnt, response_tok_cnt


    def wice_extractor(self, response, cost_estimate_only=False):
        """
        Given a model output
        - split the response into sentences using spaCy
        - snippet = (context1 = 0-3 sentence) <SOS>Sent<EOS> (context2 = 0-1 sentence)
        - call fact_extractor on each snippet
        """

        def extract_facts(text):
            # Use regex to find all fact lines and remove leading characters
            facts = re.findall(r'(?:-|\d+\.)\s*(.+)', text)
            if len(facts) == 0 and len(text.split('\n')) > 1:
                facts = text.split('\n')
            return [fact.strip() for fact in facts]

        generator = AtomicFactGenerator(cache_dir=self.cache_dir, 
                                        model_name=self.model_name, 
                                        is_bio=False)
        
        prompt_text = WICE_PROMPT.format(response.strip())
        response_content, prompt_tok_cnt, response_tok_cnt, response_logprobs = self.get_model_response.get_response(system_message="", 
                                                                                                      prompt_text=prompt_text,
                                                                                                      cost_estimate_only=cost_estimate_only)
        extracted_facts = extract_facts(response_content)

        all_facts_lst = []
        for fact in extracted_facts:
            if fact.strip() not in all_facts_lst:
                all_facts_lst.append(fact.strip())
        fact_lst_lst = [all_facts_lst]

        # keep track of token counts
        prompt_tok_cnt, response_tok_cnt = 0, 0
        
        if len(all_facts_lst) == 0:
            # If no facts are extracted, just use the original response as the only fact
            all_facts_lst = [response.strip()]
            fact_lst_lst = [[response.strip()]]

        print(f"Returning facts from WICE and token counts (IGNORE THIS token counts) for the whole response ...")
        return fact_lst_lst, all_facts_lst, prompt_tok_cnt, response_tok_cnt


    def specified_number_extractor(self, response, number_of_claims, cost_estimate_only=False):
        def extract_decomposition(decomposed_claim):
            extracted_claims = []

            pattern = r"```(.*?)```"
            matches = re.finditer(pattern, decomposed_claim, re.DOTALL)
            segments = [match.group(1).strip() for match in matches]
            for segment in segments:
                if segment.strip() == "":
                    continue
                extracted_claims.append(segment)
    
            if len(extracted_claims) == 0:
                extracted_claims = [decomposed_claim]
            
            return extracted_claims
        
        template = DECOMPOSE_NUM_PROMPT_TEMPLATE

        prompt_text = template.format(num_sub_claims=number_of_claims, input_text=response.strip())
        response_content, prompt_tok_cnt, response_tok_cnt = self.get_model_response.get_response(system_message="", 
                                                                                                  prompt_text=prompt_text,
                                                                                                  cost_estimate_only=cost_estimate_only)
        extracted_facts = extract_decomposition(response_content)

        return extracted_facts, prompt_tok_cnt, response_tok_cnt
    

    def summarization_extractor(self, response, cost_estimate_only=False):

        template = DECOMPOSE_SINGLE_PROMPT_TEMPLATE
        prompt_text = template.format(input_text=response.strip())
        response_content, prompt_tok_cnt, response_tok_cnt, = self.get_model_response.get_response(system_message="", 
                                                                                                   prompt_text=prompt_text,
                                                                                                   cost_estimate_only=cost_estimate_only)  
        all_facts_lst = [response_content]
        fact_lst_lst = [all_facts_lst]

        return fact_lst_lst, all_facts_lst, prompt_tok_cnt, response_tok_cnt