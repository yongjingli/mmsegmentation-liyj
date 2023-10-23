#!/usr/bin/bash

img_dir="/home/dyg/project/mmsegmentation/work_dirs/visual_iter_10000_train/vis_data/vis_image"
fps=10

if [ -n "$1" ]; then
  img_dir="$1"
fi

if [ -n "$2" ]; then
  fps=$2
fi

echo "img_dir: ${img_dir}"
echo "fps: ${fps}"

video_file=${img_dir}.mp4

ffmpeg -framerate ${fps} -pattern_type glob -i "${img_dir}/*.png" \
  -c:v libx264 -b:v 8M -pix_fmt yuv420p ${video_file}

echo "Done, saved to ${video_file}"
