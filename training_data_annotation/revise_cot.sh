ORIGINAL_COT_FILE=$1
OUTPUT_COT_DIR=$2
OUTPUT_SHORT_DIR=$3

python3 convert_cot.py \
    --input_file $ORIGINAL_COT_FILE \
    --output_cot_dir $OUTPUT_COT_DIR \
    --output_short_dir $OUTPUT_SHORT_DIR


