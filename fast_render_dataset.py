from emage_utils import fast_render
import os.path as osp
import os

work_dir = osp.abspath(osp.dirname(__file__))
dataset_dir = osp.join(work_dir, 'data/fretlyn')
smplx_dir = osp.join(dataset_dir, 'smplxflame_30')
audio_dir = osp.join(dataset_dir, 'wave16k')

if __name__ == '__main__':
    data_name = set()
    smplx_files = [file for file in os.listdir(smplx_dir) if file.endswith('.npz')]
    audio_files = [file for file in os.listdir(audio_dir) if file.endswith('.wav')]
    smplx_files.sort()
    audio_files.sort()
    for smplx, audio in zip(smplx_files, audio_files):
        print(f"smplx: {smplx}, audio: {audio}")
        data_name = None if smplx.strip('.npz') != audio.strip('.wav') else smplx.strip('.npz')
        if data_name is None:
            continue
        smplx = osp.join(smplx_dir, smplx)
        audio = osp.join(audio_dir, audio)
        print(f"Rendering {data_name}...")
        fast_render.render_one_sequence_no_gt(res_npz_path=smplx, output_dir="./examples/fretlyn_videos", audio_path=audio, model_folder="./emage_evaltools/smplx_models/", remove_transl=True)