import argparse
import json
import logging
import os
from tqdm import tqdm

from claim_extractor import ClaimExtractor
from claim_verifier import ClaimVerifier
from search_API import SearchAPI
from evaluate_result import evaluate
from concurrent.futures import ThreadPoolExecutor


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def run_extraction(
    data, claim_extractor, decompose_method, specified_number_of_claims=8
):
    logging.info(f"Running extraction based on decompose method: {decompose_method}")

    def process_claim(dict_item):
        response = dict_item["response"].strip()

        if "specified_number" in decompose_method.lower():
            claim_extractor.get_model_response.stop_tokens = ["\nInput: ", "\nClaim: "]
            claim_list, prompt_tok_cnt, response_tok_cnt = (
                claim_extractor.specified_number_extractor(
                    response, number_of_claims=specified_number_of_claims
                )
            )
        elif decompose_method == "summarization":
            claim_extractor.get_model_response.stop_tokens = ["\n"]
            claim_list, prompt_tok_cnt, response_tok_cnt = (
                claim_extractor.summarization_extractor(response)
            )
        else:
            raise ValueError(f"Unknown decompose method: {decompose_method}")

        return {
            "prompt_source": dict_item["prompt_source"],
            "response": response,
            "claim_list": claim_list,
            "annot_label": dict_item.get("label", None),
            "prompt_tok_cnt": prompt_tok_cnt,
            "response_tok_cnt": response_tok_cnt,
        }

    ####### Sequential processing #######
    # extracted_claims = []
    # for dict_item in tqdm(data, desc="Processing claims"):
    #     extracted_claims.append(process_claim(dict_item))

    ####### Parallel processing #######
    extracted_claims = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks to the executor and wrap with tqdm for a progress bar
        futures = [executor.submit(process_claim, item) for item in data]
        for future in tqdm(futures, desc="Processing claims"):
            extracted_claims.append(future.result())

    return extracted_claims


def run_searching(data, claim_searcher):
    def process_evidence(dict_item):

        claim_lst = dict_item["claim_list"]
        try:
            claim_snippets = claim_searcher.get_snippets(claim_lst)
            dict_item["claim_search_results"] = claim_snippets
        except Exception as e:
            logger.error(f"Error processing evidence for claim: {e}")
            dict_item["claim_search_results"] = {}

        return dict_item

    ####### Sequential processing #######
    # searched_evidence_dict = []
    # for dict_item in tqdm(data, desc="Processing claims"):
    #     searched_evidence_dict.append(process_evidence(dict_item))

    ####### Parallel processing #######
    searched_evidence_dict = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks to the executor and wrap with tqdm for a progress bar
        futures = [executor.submit(process_evidence, item) for item in data]
        for future in tqdm(futures, desc="Processing claims"):
            searched_evidence_dict.append(future.result())

    return searched_evidence_dict


def run_verification(data, claim_verifier, search_res_num):
    def process_verification(dict_item):
        claim_search_results = dict_item["claim_search_results"]
        claim_verify_res_dict, prompt_tok_cnt, response_tok_cnt = (
            claim_verifier.verifying_claim(
                claim_search_results, search_res_num=search_res_num
            )
        )
        dict_item["claim_verification_result"] = claim_verify_res_dict

        return dict_item, prompt_tok_cnt, response_tok_cnt

    # ####### Sequential processing #######
    # verification_results = []
    # total_prompt_tok_cnt = 0
    # total_resp_tok_cnt = 0
    # for dict_item in tqdm(data, desc="Processing claims"):
    #     verification_result, prompt_tok_cnt, response_tok_cnt = process_verification(dict_item)
    #     verification_results.append(verification_result)
    #     total_prompt_tok_cnt += prompt_tok_cnt
    #     total_resp_tok_cnt += response_tok_cnt

    # ####### Parallel processing #######
    verification_results = []
    total_prompt_tok_cnt = 0
    total_resp_tok_cnt = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks to the executor and wrap with tqdm for a progress bar
        futures = [executor.submit(process_verification, item) for item in data]
        for future in tqdm(futures, desc="Processing claims"):
            verification_result, prompt_tok_cnt, response_tok_cnt = future.result()
            verification_results.append(verification_result)
            total_prompt_tok_cnt += prompt_tok_cnt
            total_resp_tok_cnt += response_tok_cnt

    print(
        f"Claim verification is done! Total cost: {total_prompt_tok_cnt * 10 / 1e6 + total_resp_tok_cnt * 30 / 1e6}"
    )
    logging.info(
        f"Claim verification is done! Total cost: {total_prompt_tok_cnt * 10 / 1e6 + total_resp_tok_cnt * 30 / 1e6}"
    )

    return verification_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--demon_dir", type=str, default="./demos")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--model_name_extraction", type=str, default="gpt-4o")
    parser.add_argument("--model_name_verification", type=str, default="gpt-4o-mini")
    parser.add_argument("--label_n", type=int, default=2, choices=[2, 3])
    parser.add_argument("--search_res_num", type=int, default=10)
    parser.add_argument(
        "--decompose_method",
        type=str,
        default="veriscore",
        choices=[
            "original",
            "veriscore",
            "factscore",
            "wice",
            "specified_number",
            "summarization",
        ],
    )
    parser.add_argument("--specified_number_of_claims", type=int, default=8)
    parser.add_argument(
        "--stage",
        type=str,
        choices=["extraction", "searching", "verification", "evaluation"],
        required=True,
    )
    parser.add_argument(
        "--search_engine",
        type=str,
        default="serper",
        choices=["serper", "searxng"],
        help="The search engine to use. 'serper' for the original API, 'searxng' for the local instance.",
    )

    args = parser.parse_args()

    print("args: ", args)

    # Initialize components
    os.makedirs(args.cache_dir, exist_ok=True)

    # Load and process data
    with open(args.input_file, "r") as f:
        input_data = [json.loads(x) for x in f.readlines() if x.strip()]
    print(f"Data loaded from {args.input_file}, total {len(input_data)} items.")

    if args.stage == "extraction":
        # Initialize claim extractor
        claim_extractor = ClaimExtractor(
            args.model_name_extraction, cache_dir=args.cache_dir
        )
        output_data = run_extraction(
            data=input_data,
            claim_extractor=claim_extractor,
            decompose_method=args.decompose_method,
            specified_number_of_claims=args.specified_number_of_claims,
        )

    elif args.stage == "searching":
        claim_searcher = None
        if args.search_engine == "serper":
            from search_API import SearchAPI

            logger.info("Using 'serper' search engine.")
            claim_searcher = SearchAPI()
        elif args.search_engine == "searxng":
            from search_api_searxng import SearchAPISearxng

            logger.info("Using 'searxng' search engine.")
            claim_searcher = SearchAPISearxng()  # Uses http://localhost:8080 by default

        if claim_searcher:
            output_data = run_searching(data=input_data, claim_searcher=claim_searcher)
        else:
            # argparse 'choices' should prevent this, but it's safe to have
            raise ValueError(
                f"Unknown or unsupported search engine: {args.search_engine}"
            )

    elif args.stage == "verification":
        # Initialize claim verifier
        claim_verifier = ClaimVerifier(
            model_name=args.model_name_verification,
            label_n=args.label_n,
            cache_dir=args.cache_dir,
            demon_dir=args.demon_dir,
        )
        output_data = run_verification(
            data=input_data,
            claim_verifier=claim_verifier,
            search_res_num=args.search_res_num,
        )

    elif args.stage == "evaluation":
        metrics_dict = evaluate(input_data)

    if args.stage != "evaluation":
        with open(args.output_file, "w") as f:
            for dict_item in output_data:
                f.write(json.dumps(dict_item) + "\n")
