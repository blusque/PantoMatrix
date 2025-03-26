import numpy as np
import torch
import torch.nn.functional as F
from models.emage_audio import EmageVQVAEConv, EmageVAEConv, EmageVQModel
import emage_utils.rotation_conversions as rc
from torchvision.io import write_video
from emage_utils import fast_render
from smplx import SMPLX
from emage_utils.motion_io import beat_format_save
from scipy.spatial.transform import Rotation as R

import pandas as pd

data_folder = './data/fretlyn'
save_folder_base = './data/reconstruct'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def evaluate(gt, pred):
    return F.mse_loss(gt, pred)

def reconstruct(motion_model, input_motion, input_expression, gt_trans, video_id):
    motion_index = motion_model.map2index(input_motion, input_expression)
    motion_latent = motion_model.map2latent(input_motion, input_expression)
    recon = motion_model.decode(motion_index['face'], motion_index['upper'], motion_index['hands'],
                                motion_index['lower'], motion_latent['face'], motion_latent['upper'],
                                motion_latent['hands'], motion_latent['lower'], get_global_motion=True, ref_trans=gt_trans)
    motion_pred = recon["motion_axis_angle"]
    t = motion_pred.shape[1]
    motion_pred = motion_pred.cpu().numpy().reshape(t, -1)
    expression_pred = recon["expression"].cpu().numpy().reshape(t, -1)
    trans_pred = recon["trans"].cpu().numpy().reshape(t, -1)
    # print(motion_pred.shape, expression_pred.shape, trans_pred.shape)
    beat_format_save(os.path.join(save_folder, f"{video_id}_output.npz"), motion_pred, upsample=1, expressions=expression_pred, trans=trans_pred)
    # beat_format_save(os.path.join(save_folder, f"{video_id}_output.npz"), motion_pred, upsample=1, expressions=expression_pred)
    re = recon["motion_axis_angle"]
    return re

def visualize_one(save_folder, audio_path, nopytorch3d=False):  
    npz_path = os.path.join(save_folder, f"{os.path.splitext(os.path.basename(audio_path))[0]}_output.npz")
    gt_path = os.path.join(data_folder, 'smplxflame_30', f"{os.path.splitext(os.path.basename(audio_path))[0]}.npz")
    # motion_dict = np.load(npz_path, allow_pickle=True)
    # if not nopytorch3d:
    #     from emage_utils.npz2pose import render2d
    #     v2d_face = render2d(motion_dict, (512, 512), face_only=True, remove_global=True)
    #     write_video(npz_path.replace(".npz", "_2dface.mp4"), v2d_face.permute(0, 2, 3, 1), fps=30)
    #     fast_render.add_audio_to_video(npz_path.replace(".npz", "_2dface.mp4"), audio_path, npz_path.replace(".npz", "_2dface_audio.mp4"))
    #     v2d_body = render2d(motion_dict, (720, 480), face_only=False, remove_global=True)
    #     write_video(npz_path.replace(".npz", "_2dbody.mp4"), v2d_body.permute(0, 2, 3, 1), fps=30)
    #     fast_render.add_audio_to_video(npz_path.replace(".npz", "_2dbody.mp4"), audio_path, npz_path.replace(".npz", "_2dbody_audio.mp4"))
    fast_render.render_one_sequence(npz_path, gt_path, os.path.dirname(npz_path), audio_path, model_folder="./emage_evaltools/smplx_models/", remove_transl=True, rotation=R.from_euler('XYZ', [-90, 0, 0], degrees=True))

if __name__ == '__main__':
    import os
    import os.path as osp
    if not osp.exists(save_folder_base):
        os.mkdir(save_folder_base)
    df = pd.read_csv(osp.join(data_folder, 'train_test_split.csv'))
    train_id = df[df['type'] == 'train']['id'].to_list()
    test_id = df[df['type'] == 'test']['id'].to_list()
    smplx_files = [file for file in os.listdir('data/fretlyn/smplxflame_30') if file.endswith('.npz')]
    eval = np.zeros((len(smplx_files)))
    face_motion_vq = EmageVQVAEConv.from_pretrained("outputs/motion_vae_fretlyn_20250326-1256/checkpoints/best/vq_face").to(device)
    upper_motion_vq = EmageVQVAEConv.from_pretrained("outputs/motion_vae_fretlyn_20250326-1256/checkpoints/best/vq_upper").to(device)
    lower_motion_vq = EmageVQVAEConv.from_pretrained("outputs/motion_vae_fretlyn_20250326-1256/checkpoints/best/vq_lower").to(device)
    hands_motion_vq = EmageVQVAEConv.from_pretrained("outputs/motion_vae_fretlyn_20250326-1256/checkpoints/best/vq_hands").to(device)
    global_motion_ae = EmageVAEConv.from_pretrained("outputs/motion_vae_fretlyn_20250326-1256/checkpoints/best/global").to(device)
    motion_model = EmageVQModel(
        face_model=face_motion_vq, upper_model=upper_motion_vq,
        hands_model=hands_motion_vq, lower_model=lower_motion_vq,
        global_model=global_motion_ae
    )
    for param in motion_model.parameters():
        param.requires_grad_(False)

    for id, smplx_file in enumerate(smplx_files):
        print(f"{id+1}/{len(smplx_files)} Processing: {smplx_file}...")
        if smplx_file.strip('.npz') in test_id:
            save_folder = osp.join(save_folder_base, 'test')
        elif smplx_file.strip('.npz') in train_id:
            save_folder = osp.join(save_folder_base, 'train')
        if not osp.exists(save_folder):
            os.mkdir(save_folder)
        video_id = smplx_file.strip('.npz')
        smplx = np.load(osp.join('data/fretlyn/smplxflame_30', smplx_file), allow_pickle=True)
        input_motion = smplx['poses']
        input_trans = smplx['trans']
        t, _ = input_motion.shape
        input_motion = input_motion.reshape(1, t, -1, 3)
        input_trans = input_trans.reshape(1, t, 3)
        input_motion = torch.from_numpy(input_motion).to(torch.float32).to(device)
        input_motion_6d = rc.axis_angle_to_rotation_6d(input_motion)
        input_motion_6d = input_motion_6d.reshape(1, t, -1)
        input_trans = torch.from_numpy(input_trans).to(torch.float32).to(device)
        input_expression = torch.zeros((1, t, 100), dtype=torch.float32).to(device)
        recon = reconstruct(motion_model, input_motion_6d, input_expression, input_trans, video_id)
        recon = recon.reshape(1, t, -1, 3)
        eval[id] = evaluate(input_motion, recon)
        visualize_one(save_folder, os.path.join(data_folder, 'wave16k', video_id+'.wav'), True)

    np.save('evaluation_vq.npy', eval)
    
    print(f'max: {np.max(eval)}, min: {np.min(eval)}, avg: {np.mean(eval)}')