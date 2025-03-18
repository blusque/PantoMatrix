import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
from sklearn.model_selection import train_test_split

# Example parameters
stride = 20
origin_motion_length = 32
print(origin_motion_length)
aim_fps = 30
# speaker_targets = [12, 13, 22, 23, 24]
speaker_targets = [-1]
use_additional = False

# root_dir = './beat_english_v2.0.0/'
# root_dir = './BEAT2/beat_chinese_v2.0.0/'
root_dir = './data/fretlyn'
output_dir = "./datasets/data_json/"
os.makedirs(output_dir, exist_ok=True)
train_test_split_path = os.path.join(root_dir, 'train_test_split.csv')
if not os.path.exists(train_test_split_path):
  f = open(train_test_split_path, 'w')
  f.write('id,type\n')
  files = [f.strip('.npz') for f in os.listdir(os.path.join(root_dir, "smplxflame_30")) if f.endswith('.npz')]
  train_files, test_files = train_test_split(files, test_size=0.1, random_state=42)
  for file in train_files:
    f.write(f"{file},train\n")
  for file in test_files:
    f.write(f"{file},test\n")
  f.close()
df = pd.read_csv(train_test_split_path)

for speaker_target in speaker_targets:
  print(f"Processing speaker {speaker_target}")
  if speaker_target == -1:
    filtered_df = df[df['type'] != 'additional']
  else:
    filtered_df = df[(df['id'].str.split('_').str[0].astype(int) == speaker_target) & (df['type'] != 'additional')]
  clips = []
  for idx, row_item in tqdm(filtered_df.iterrows()):
      video_id = row_item['id']
      mode = row_item['type'] 
      # check exist
      npz_path = os.path.join(root_dir, "smplxflame_30", video_id + ".npz")
      wav_path = os.path.join(root_dir, "wave16k", video_id + ".wav")

      try:
        motion_data = np.load(npz_path, allow_pickle=True)
      except:
        print(f"cant open {npz_path}")
        continue
      
      try:
        wave_data, _ = librosa.load(wav_path, sr=None)
      except:
        print(f"cant open {wav_path}")
        continue

      motion = motion_data['poses']
      motion_fps = motion_data['mocap_frame_rate']
      motion_length = int(motion_fps / aim_fps) * origin_motion_length
      total_len = motion.shape[0]

      for i in range(0, total_len - motion_length, stride):
          clip = {
              "video_id": video_id,
              "motion_path": npz_path,
              "audio_path": wav_path,
              "mode": mode,
              "start_idx": i,
              "end_idx": i + motion_length
          }
          clips.append(clip)

  # output_json = os.path.join(output_dir, f"beat2_s{stride}_l{motion_length}_speaker{speaker_target}.json")
  if speaker_target == -1:
    output_json = os.path.join(output_dir, f"fretlyn_s{stride}_l{origin_motion_length}_all.json")
  else:
    output_json = os.path.join(output_dir, f"fretlyn_s{stride}_l{origin_motion_length}_speaker{speaker_target}.json")
  with open(output_json, 'w') as f:
      json.dump(clips, f, indent=4)
