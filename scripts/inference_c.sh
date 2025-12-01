#!/usr/bin/env bash
CKPT_NAME=$1
TAKE_NAME=$2
FPS=${3:-30}

CAM_ID="16"
# NAME="${FILE_NAME%.*}"
# EXT="${FILE_NAME##*.}"



IMG_PATH=/media/cormond/hdd/data/pilot_oct10/$TAKE_NAME/images/$CAM_ID
OUTPUT_PATH=/media/cormond/hdd/data/pilot_oct10/$TAKE_NAME/smplx_2d

# mkdir -p $IMG_PATH
mkdir -p $OUTPUT_PATH


END_COUNT=$(find "$IMG_PATH" -type f | wc -l)

# inference with smplest_x
PYTHONPATH=../:$PYTHONPATH \
python main/inference_c.py \
    --img_path $IMG_PATH \
    --output_path $OUTPUT_PATH \
    --take_name $TAKE_NAME \
    --cam_id $CAM_ID \
    --num_gpus 1 \
    --ckpt_name $CKPT_NAME \
    --end $END_COUNT \
    # --file_name $NAME \


# convert frames to video
ffmpeg -y -framerate ${FPS} \
    -i ${OUTPUT_PATH}/images_smplx_overlay/${CAM_ID}/%06d.jpg \
    -c:v mpeg4 \
    -q:v 2 \
    -pix_fmt yuv420p \
    /media/cormond/hdd/data/pilot_oct10/$TAKE_NAME/smplx_2d/smplx_overlay_${TAKE_NAME}_cam${CAM_ID}.mp4


# rm -rf ./demo/input_frames
# rm -rf ./demo/output_frames

