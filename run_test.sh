export SERPER_KEY_PRIVATE="0325f2478ebae737e125dafc8d94de5334af1e8d"
export OPENAI_API_BASE="https://api.shubiaobiao.cn/v1"
export OPENAI_API_KEY="sk-agEcX3Su78Bu09c2F49978C6Ba424977B936C8710fAb42E0"

# INPUT="./fact_checking_dataset/fact_checking_dataset_2017.jsonl"
# OUTPUT="./fact_checking_dataset/fact_checking_dataset_2017_result.json"
DATA_PREFIX="./fact_checking_dataset/fact_checking_dataset_2025"

# extraction -> searching -> verification -> evaluation

# python3 src/pipeline_nli.py \
#     --input_file ${DATA_PREFIX}.jsonl \
#     --output_file ${DATA_PREFIX}_extract.json \
#     --model_name_extraction "gpt-4o" \
#     --model_name_verification "gemini-1.5-flash-latest" \
#     --decompose_method "specified_number" \
#     --specified_number_of_claims 8 \
#     --label_n 2 \
#     --stage "extraction"

# python3 src/pipeline_nli.py \
#     --input_file ${DATA_PREFIX}_extract.json \
#     --output_file ${DATA_PREFIX}_search.json \
#     --model_name_extraction "gpt-4o" \
#     --model_name_verification "gemini-1.5-flash-latest" \
#     --decompose_method "specified_number" \
#     --specified_number_of_claims 8 \
#     --label_n 2 \
#     --stage "searching" \
#     --search_engine searxng 

# python3 src/pipeline_nli.py \
#     --input_file ${DATA_PREFIX}_search.json \
#     --output_file ${DATA_PREFIX}_verify.json \
#     --model_name_extraction "gpt-4o" \
#     --model_name_verification "gemini-1.5-flash-latest" \
#     --decompose_method "specified_number" \
#     --specified_number_of_claims 8 \
#     --label_n 2 \
#     --stage "verification"

# python3 src/pipeline_nli.py \
#     --input_file ${DATA_PREFIX}_verify.json \
#     --output_file ${DATA_PREFIX}_evaluate.json \
#     --model_name_extraction "gpt-4o" \
#     --model_name_verification "gemini-1.5-flash-latest" \
#     --decompose_method "specified_number" \
#     --specified_number_of_claims 8 \
#     --label_n 2 \
#     --stage "evaluation"
