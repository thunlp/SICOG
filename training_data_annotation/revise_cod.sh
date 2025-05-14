ORIGINAL_CAPTION_FILE=$1
IMAGE_DIR=$2
OUTPUT_COD_DIR=$3
OUTPUT_DD_DIR=$4

python3 generate_cod_caption.py \
    --input_file $ORIGINAL_CAPTION_FILE \
    --image_dir $IMAGE_DIR \
    --output_cod_dir $OUTPUT_COD_DIR \
    --output_dd_dir $OUTPUT_DD_DIR \

