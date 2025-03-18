import json
import os
import os.path as osp
import shutil

data_meta = {
    "video_id": "video_id",
    "motion_path": "/path/to/motion.npz",
    "audio_path": "/path/to/audio.wav",
    "mode": "train/val/test",
    "start_idx": 0,
    "end_idx": -1
}

subject_name = 'fretlyn'

base_dir = osp.abspath(osp.join(osp.dirname(__file__), ".."))
work_dir = osp.abspath(osp.dirname(__file__))
json_dir = osp.join(work_dir, "data_json")
target_json = osp.join(json_dir, f"{subject_name}.json")
data_dir = osp.join(work_dir, f"../data/{subject_name}")

motion_dir = osp.join(data_dir, "smplx")
audio_dir = osp.join(data_dir, "audio")

motion_files = os.listdir(motion_dir)
data_info_list = []
for motion_file in motion_files:
    if not motion_file.endswith(".npz"):
        continue
    motion_path = osp.relpath(osp.join(motion_dir, motion_file), base_dir)
    audio_path = osp.relpath(osp.join(audio_dir, f"{motion_file.split('.')[0]}.wav"), base_dir)
    data_info = data_meta.copy()
    data_info["motion_path"] = motion_path
    data_info["audio_path"] = audio_path
    data_info["mode"] = "train"
    data_info_list.append(data_info)
with open(target_json, "w") as f:
    json.dump(data_info_list, f, indent=4)