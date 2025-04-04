import os
import shutil
import argparse
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
import importlib
import copy
import librosa
from pathlib import Path
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import math
import emage_utils.rotation_conversions as rc

from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf

from emage_evaltools.mertic import FGD, BC, L1div, LVDFace, MSEFace
from emage_utils.motion_io import beat_format_load, beat_format_save, SMPLX_MASK_DICT, recover_from_mask
import emage_utils.rotation_conversions as rc
from emage_utils import fast_render
from emage_utils.motion_rep_transfer import get_motion_rep_numpy
from models.emage_audio import EmageVQVAEConv, EmageVAEConv, EmageRVQModel, EmageAudioModel
from peft import LoraConfig, get_peft_model, TaskType
from smplx import SMPLX
from smplx.lbs import lbs

# ---------------------------------  train,val,test fn here --------------------------------- #
def inference_fn(cfg, motion_vq: EmageRVQModel, device, test_path, save_path, **kwargs):
    actual_model = motion_vq.module if isinstance(motion_vq, torch.nn.parallel.DistributedDataParallel) else motion_vq
    actual_model.eval()
    test_list = []
    for data_meta_path in test_path:
        test_list.extend(json.load(open(data_meta_path, "r")))
    test_list = [item for item in test_list if item.get("mode") == "test"]
    seen_ids = set()
    test_list = [item for item in test_list if not (item["video_id"] in seen_ids or seen_ids.add(item["video_id"]))]

    save_list = []
    start_time = time.time()
    total_length = 0
    for test_file in tqdm(test_list, desc="Testing"):
        # motion seed
        motion_data = np.load(test_file["motion_path"], allow_pickle=True)
        poses = torch.from_numpy(motion_data["poses"]).unsqueeze(0).to(device).float()
        bs, t, _ = poses.shape
        foot_contact = torch.from_numpy(np.load(test_file["motion_path"].replace("smplxflame_30", "footcontact").replace(".npz", ".npy"))).unsqueeze(0).to(device).float()
        trans = torch.from_numpy(motion_data["trans"]).unsqueeze(0).to(device).float()
        if "expressions" in motion_data:
            expression = torch.from_numpy(motion_data["expressions"]).unsqueeze(0).to(device).float()
        else:
            expression = torch.zeros(poses.shape[1], 100).unsqueeze(0).to(device).float()
        poses_6d = rc.axis_angle_to_rotation_6d(poses.reshape(bs, t, -1, 3)).reshape(bs, t, -1)
        masked_motion = torch.cat([poses_6d, trans, foot_contact], dim=-1) # bs t 337

        downsample_factor = math.ceil(motion_data['mocap_frame_rate'] // cfg.data.pose_fps)
        masked_motion = masked_motion[:, ::downsample_factor]

        # reconstrcution check
        motion_index = actual_model.map2index(poses_6d, expression, tar_contact=foot_contact, tar_trans=trans)
        face_index = motion_index['face']
        upper_index = motion_index['upper']
        lower_index = motion_index['lower']
        hands_index = motion_index['hands']

        latent_dict = actual_model.map2latent(poses_6d, expression, tar_contact=foot_contact, tar_trans=trans)
        face_latent = latent_dict["face"]
        upper_latent = latent_dict["upper"]
        lower_latent = latent_dict["lower"]
        hands_latent = latent_dict["hands"]

        motion_all = actual_model.decode(
            face_latent=face_latent, upper_latent=upper_latent, lower_latent=lower_latent, hands_latent=hands_latent,
            face_index=face_index, upper_index=upper_index, lower_index=lower_index, hands_index=hands_index,
            get_global_motion=True, ref_trans=trans[:, 0])
       
        motion_pred = motion_all["motion_axis_angle"]
        t = motion_pred.shape[1]
        motion_pred = motion_pred.cpu().numpy().reshape(t, -1)
        expression_pred = motion_all["expression"].cpu().numpy().reshape(t, -1)
        trans_pred = motion_all["trans"].cpu().numpy().reshape(t, -1)
        # print(motion_pred.shape, expression_pred.shape, trans_pred.shape)
        beat_format_save(os.path.join(save_path, f"{test_file['video_id']}_output.npz"), motion_pred, upsample=30//cfg.data.pose_fps, expressions=expression_pred, trans=trans_pred)
        save_list.append(
            {
                "audio_path": test_file["audio_path"],
                "motion_path": os.path.join(save_path, f"{test_file['video_id']}_output.npz"),
                "video_id": test_file["video_id"],
            }
        )
        total_length+=t
    time_cost = time.time() - start_time
    print(f"\n cost {time_cost:.2f} seconds to generate {total_length / cfg.data.pose_fps:.2f} seconds of motion")
    return test_list, save_list

def get_ang_vel_from_rot(local_rotation: torch.Tensor, parents):
    local_rotation_mat = rc.axis_angle_to_matrix(local_rotation)
    orientation_mat = torch.zeros_like(local_rotation_mat)
    for i in range(len(parents)):
        pi = parents[i]
        if pi == -1:
            orientation_mat[:, :, i, ...] = local_rotation_mat[:, :, i, ...].clone()
        else:
            orientation_mat[:, :, i, ...] = orientation_mat[:, :, pi, ...].clone() @ local_rotation_mat[:, :, i, ...]
    ang_vel_mat = orientation_mat[:, :-1, ...].transpose(-1, -2) @ orientation_mat[:, 1:, ...]
    ang_vel = rc.matrix_to_axis_angle(ang_vel_mat)
    return ang_vel

def get_rec_loss(motion_pred, motion_gt):
    return F.mse_loss(motion_pred, motion_gt)

def get_trans_loss(trans_pred, trans_gt):
    return F.mse_loss(trans_pred, trans_gt)
    
def get_vel_loss(motion_pred_vel):
    '''
    :motion_pred_vel: (bs, t, joint, 3)
    '''
    zeros = torch.zeros_like(motion_pred_vel).to(motion_pred_vel.device)
    return F.mse_loss(motion_pred_vel, zeros)

def get_vq_loss(ze: dict, zq: dict):
    total_loss = 0
    for key in zq.keys():
        sg_ze = ze[key].detach().clone()
        total_loss += F.mse_loss(sg_ze, zq[key])
    return total_loss

def get_commitment_loss(ze: dict, zq: dict, beta):
    total_loss = 0
    for key in zq.keys():
        sg_zq = zq[key].detach().clone()
        total_loss += F.mse_loss(ze[key], sg_zq)
    return total_loss

def train_val_fn(cfg, batch, motion_vq: EmageRVQModel, device, mode="train", **kwargs):
    if mode == "train":
        actual_model = motion_vq.module if isinstance(motion_vq, torch.nn.parallel.DistributedDataParallel) else motion_vq
        actual_model.train()
        if isinstance(kwargs['optimizer'], dict):
            for key in kwargs['optimizer'].keys():
                kwargs['optimizer'][key].zero_grad()
        else:
            kwargs["optimizer"].zero_grad()
    else:
        actual_model = motion_vq.module if isinstance(motion_vq, torch.nn.parallel.DistributedDataParallel) else motion_vq
        actual_model.eval()

    motion_gt = batch["motion"].to(device)
    expressions_gt = batch["expressions"].to(device)
    betas = batch['betas'].to(device)
    trans = batch["trans"].to(device)
    foot_contact = batch["foot_contact"].to(device)

    bs, t, jc = motion_gt.shape
    betas = betas.unsqueeze(1).repeat(1, t, 1)
    smplx = kwargs.get('smplx', SMPLX(batch_size=bs*t, **cfg.smplx).to(device))
    j = jc // 3
    motion_gt_axis_angle = motion_gt.clone()
    motion_gt_trans = trans.clone()
    motion_gt = rc.axis_angle_to_rotation_6d(motion_gt.reshape(bs, t, j, 3)).reshape(bs, t, j*6)
    
    motion_pred_dict = actual_model(motion_gt, expressions_gt, tar_contact=foot_contact, get_global_motion=True, tar_trans=trans)

    perplexity = motion_pred_dict['perplexity']
    indices = motion_pred_dict['indices']

    motion_pred_axis_angle = motion_pred_dict['motion_axis_angle'].reshape(bs, t, 165)

    betas = betas.reshape(bs * t, -1)
    expressions = expressions_gt.reshape(bs * t, -1)
    motion_gt_axis_angle = motion_gt_axis_angle.reshape(bs * t, -1)
    motion_pred_axis_angle = motion_pred_axis_angle.reshape(bs * t, -1)
    shape_components = torch.cat([betas, expressions], dim=-1)
    shape_dirs = torch.cat([smplx.shapedirs, smplx.expr_dirs], dim=-1)

    motion_pred_trans = motion_pred_dict['trans']

    # print(motion_pred_trans)

    _, motion_gt_pos = lbs(shape_components, motion_gt_axis_angle, smplx.v_template,
                        shape_dirs, smplx.posedirs, smplx.J_regressor, smplx.parents,
                        smplx.lbs_weights)
    motion_gt_pos = motion_gt_pos.reshape(bs, t, 55, 3)

    _, motion_pred_pos = lbs(shape_components, motion_pred_axis_angle, smplx.v_template,
                         shape_dirs, smplx.posedirs, smplx.J_regressor, smplx.parents,
                         smplx.lbs_weights)
    motion_pred_pos = motion_pred_pos.reshape(bs, t, 55, 3)

    motion_pred_vel = torch.diff(motion_pred_pos, dim=1)

    motion_pred_trans_vel = torch.diff(motion_pred_trans, dim=1)
    motion_pred_ang_vel = get_ang_vel_from_rot(motion_pred_axis_angle.reshape(bs, t, 55, 3), smplx.parents)
    loss_dict = {
        'rec_rot_seed': get_rec_loss(motion_pred_axis_angle, motion_gt_axis_angle),
        "rec_pos_seed": get_rec_loss(motion_pred_pos, motion_gt_pos),
        'trans_seed': get_trans_loss(motion_pred_trans, motion_gt_trans),
        'embedding_seed': motion_pred_dict['embedding_loss'],
        "vel_seed": get_vel_loss(motion_pred_vel),
        "trans_vel_seed": get_vel_loss(motion_pred_trans_vel),
        "ang_vel_seed": get_vel_loss(motion_pred_ang_vel)
    }
    
    all_loss = 0
    for key, loss in loss_dict.items():
        all_loss += cfg.loss_weights[key] * loss
    loss_dict["all"] = all_loss
  
    if mode == "train":
        if cfg.solver.max_grad_norm > 0:
          torch.nn.utils.clip_grad_norm_(actual_model.parameters(), cfg.solver.max_grad_norm)
        all_loss.backward()
        if isinstance(kwargs['optimizer'], dict):
            for optimizer in kwargs["optimizer"].values():
                optimizer.step()
            actual_model.update_ema()
        else:
            kwargs["optimizer"].step()
            actual_model.update_ema()
            actual_model.update_codebook()
        if isinstance(kwargs['lr_scheduler'], dict):
            for lr_scheduler in kwargs["lr_scheduler"].values():
                lr_scheduler.step()
        else:
            kwargs["lr_scheduler"].step()
    elif mode == 'val':
        latent_index_dict = actual_model.map2index(motion_gt, expressions_gt, tar_contact=foot_contact, tar_trans=trans)
        latent_dict = actual_model.map2latent(motion_gt, expressions_gt, tar_contact=foot_contact, tar_trans =trans)
        decode_dict = actual_model.decode(
            face_index=latent_index_dict['face'], upper_index=latent_index_dict['upper'],
            hands_index=latent_index_dict['hands'], lower_index=latent_index_dict['lower'],
            face_latent=latent_dict['face'], upper_latent=latent_dict['upper'],
            lower_latent=latent_dict['lower'], get_global_motion=True, ref_trans=trans)
        motion_pred_rot6d = decode_dict["all_motion4inference"][:, :, :-7]
        motion_gt_rot6d = rc.axis_angle_to_rotation_6d(motion_gt_axis_angle.reshape(bs, t, j, 3)).reshape(bs, t, -1)
        kwargs["fgd_evaluator"].update(motion_pred_rot6d, motion_gt_rot6d)
    return loss_dict, perplexity, indices


# ---------------------------------  main train loop here --------------------------------- #
def main(cfg):
    seed_everything(cfg.seed)
    os.environ["WANDB_API_KEY"] = cfg.wandb_key
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(backend="nccl")
    log_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    experiment_ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(experiment_ckpt_dir, exist_ok=True)

    if local_rank == 0 and cfg.validation.wandb:  
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        wandb.init(
            project=cfg.wandb_project,
            name=f"{cfg.exp_name}_{run_time}",
            entity=cfg.wandb_entity,
            dir=log_dir,
            config=OmegaConf.to_container(cfg)
        )

    # init
    face_motion_vq = init_hf_class(cfg.face_model.name_pyfile, cfg.face_model.class_name, cfg.face_model)
    upper_motion_vq = init_hf_class(cfg.upper_model.name_pyfile, cfg.upper_model.class_name, cfg.upper_model)
    lower_motion_vq = init_hf_class(cfg.lower_model.name_pyfile, cfg.lower_model.class_name, cfg.lower_model)
    hands_motion_vq = init_hf_class(cfg.hands_model.name_pyfile, cfg.hands_model.class_name, cfg.hands_model)
    global_motion_ae = init_hf_class(cfg.global_model.name_pyfile, cfg.global_model.class_name, cfg.global_model)

    # face_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/face").to(device)
    # upper_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/upper").to(device)
    # lower_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/lower").to(device)
    # hands_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/hands").to(device)
    # global_motion_ae = EmageVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/global").to(device)

    motion_vq = EmageRVQModel(
      face_model=face_motion_vq, upper_model=upper_motion_vq,
      lower_model=lower_motion_vq, hands_model=hands_motion_vq,
      global_model=global_motion_ae).to(device)
    
    if cfg.peft and str.lower(cfg.peft) == 'lora':
        config = LoraConfig(peft_type=TaskType.FEATURE_EXTRACTION,
                            inference_mode=False,
                            r=8,
                            target_modules=['out_proj', 'motion_proj', 'audio_body_motion_proj'],
                            lora_alpha=32,
                            lora_dropout=0.1)
        motion_vq = get_peft_model(motion_vq, config)
  
    motion_vq = nn.SyncBatchNorm.convert_sync_batchnorm(motion_vq)
    # for name, param in motion_vq.named_parameters():
    #     param.requires_grad = True
    motion_vq = DDP(motion_vq, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)

    # optimizer
    enc_dec_params = []
    quantizer_params = []
    for name, p in motion_vq.named_parameters():
        if p.requires_grad:
            if 'quantizer' in name:
                quantizer_params.append(p)
            else:
                enc_dec_params.append(p)
    enc_dec_optimizer_cls = torch.optim.AdamW
    quantizer_optimizer_cls = torch.optim.SGD
    optimizers = {}
    if len(enc_dec_params) > 0:
        enc_dec_optimizer = enc_dec_optimizer_cls(
            enc_dec_params,
            lr=cfg.solver.enc_dec.learning_rate,
            betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
            weight_decay=cfg.solver.adam_weight_decay,
            eps=cfg.solver.adam_epsilon
        )
        optimizers['enc_dec'] = enc_dec_optimizer
    if len(quantizer_params) > 0:
        quantizer_optimizer = quantizer_optimizer_cls(
            quantizer_params,
            lr=cfg.solver.quantizer.learning_rate
        )
        optimizers['quantizer'] = quantizer_optimizer
    assert len(optimizers.keys()) > 0, "No optimizer"
    lr_schedulers = {}
    for key, opt in optimizers.items():
        lr_cfg = cfg.solver.__getattr__(key)
        lr_schedulers[key] = get_scheduler(
            lr_cfg.lr_scheduler,
            optimizer=opt,
            num_warmup_steps=lr_cfg.lr_warmup_steps * cfg.solver.gradient_accumulation_steps,
            num_training_steps=cfg.solver.max_train_steps * cfg.solver.gradient_accumulation_steps
        )

    # loss
    ClsFn = nn.NLLLoss()

    # dataset
    train_dataset = init_class(cfg.data.name_pyfile, cfg.data.class_name, cfg, split='train')
    test_dataset = init_class(cfg.data.name_pyfile, cfg.data.class_name, cfg, split='test')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.train_bs, sampler=train_sampler, drop_last=True, num_workers=1, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.train_bs, sampler=test_sampler, drop_last=False, num_workers=1, pin_memory=False)

    # resume
    if cfg.resume_from_checkpoint:
        checkpoint = torch.load(cfg.resume_from_checkpoint, map_location="cpu")
        motion_vq.load_state_dict(checkpoint["model_state_dict"])
        for key, optim in optimizers.items():
            optim.load_state_dict(checkpoint["optimizer_state_dict"][key])
        for key, scheduler in optimizers.items():
            scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'][key])
        iteration = checkpoint["iteration"]
    else:  
        iteration = 0
    if cfg.test:
        iteration = 0

    max_epochs = (cfg.solver.max_train_steps // len(train_loader)) + (1 if cfg.solver.max_train_steps % len(train_loader) != 0 else 0)
    start_epoch = iteration // len(train_loader)
    start_step_in_epoch = iteration % len(train_loader)
    fgd_evaluator = FGD(download_path="./emage_evaltools/")
    bc_evaluator = BC(download_path="./emage_evaltools/", sigma=0.3, order=7)
    l1div_evaluator= L1div()
    lvd_evaluator = LVDFace()
    mse_evaluator = MSEFace()
    loss_meters = {}
    loss_meters_val = {}
    best_fgd_val = np.inf
    best_fgd_iteration_val= 0
    best_fgd_test = np.inf
    best_fgd_iteration_test = 0

    # train loop
    epoch = start_epoch
    while iteration < cfg.solver.max_train_steps:
        train_sampler.set_epoch(epoch)
        data_start = time.time()
        pbar = tqdm(train_loader, leave=True)
        for i, batch in enumerate(pbar):
            # for correct resume, if the dataset is very large. since we fixed the seed, we can skip the data
            if i < start_step_in_epoch: 
              iteration += 1
              continue
           
            # test
            if iteration % cfg.validation.test_steps == 0 and local_rank == 0:
                test_save_path = os.path.join(log_dir, f"test_{iteration}")
                os.makedirs(test_save_path, exist_ok=True)
                with torch.no_grad():
                    test_list, save_list = inference_fn(cfg, motion_vq, device, cfg.data.test_meta_paths, test_save_path)
                if cfg.validation.evaluation:
                    metrics = evaluation_fn([True]*55, test_list, save_list, fgd_evaluator, bc_evaluator, l1div_evaluator, device, lvd_evaluator, mse_evaluator)
                if cfg.validation.visualization: visualization_fn(save_list, test_save_path, test_list, only_check_one=True)
                if cfg.validation.evaluation: best_fgd_test, best_fgd_iteration_test =  log_test(motion_vq, metrics, iteration, best_fgd_test, best_fgd_iteration_test, cfg, local_rank, experiment_ckpt_dir, test_save_path)
                if cfg.test: return 0

            # validation
            if iteration % cfg.validation.validation_steps == 0:
                loss_meters = {}
                loss_meters_val = {}
                fgd_evaluator.reset()
                pbar_val = tqdm(test_loader, leave=True)

                data_start_val = time.time()
                total_indices = {
                    'face': [],
                    'upper': [],
                    'hands': [],
                    'lower': []
                }
                total_val_loss_dict = {}
                for j, batch in enumerate(pbar_val):
                    data_time_val = time.time() - data_start_val
                    with torch.no_grad():
                        val_loss_dict, perplexity, indices = train_val_fn(cfg, batch, motion_vq, device, mode="val", fgd_evaluator=fgd_evaluator, ClsFn=ClsFn, iteration=iteration)
                    net_time_val = time.time() - data_start_val
                    val_loss_dict["fgd"] = fgd_evaluator.compute() if j == len(test_loader) - 1 else 0
                    data_start_val = time.time()
                    for key in total_indices.keys():
                        total_indices[key].append(indices[key]) # list(list())
                    for key in val_loss_dict.keys():
                        if key not in total_val_loss_dict.keys():
                            total_val_loss_dict[key] = 0
                        total_val_loss_dict[key] += val_loss_dict[key]  # list(list())
                    if cfg.debug and j > 1: break
                usage_rate = {
                    'face': None,
                    'upper': None,
                    'hands': None,
                    'lower': None
                }
                for key, value in total_indices.items():
                    vae_codebook_size = cfg.__getattr__(f'{key}_model').vae_codebook_size
                    usage_rate[key] = []
                    if isinstance(value, list):
                        for indices in value:
                            indices = torch.cat(indices, dim=0)
                            embedding_onehot = F.one_hot(indices, vae_codebook_size).to(torch.float32)
                            e_mean = torch.mean(embedding_onehot, dim=0)
                            usage_rate[key].append(torch.sum((e_mean != 0).to(torch.float32)) / float(vae_codebook_size))
                    else:
                        indices = torch.cat(value, dim=0)
                        embedding_onehot = F.one_hot(indices, vae_codebook_size).to(torch.float32)
                        e_mean = torch.mean(embedding_onehot, dim=0)
                        usage_rate[key].append(torch.sum((e_mean != 0).to(torch.float32)) / float(vae_codebook_size))
                log_train_val(cfg, total_val_loss_dict, local_rank, loss_meters_val, pbar_val, epoch, max_epochs, iteration, net_time_val, data_time_val, optimizers, "Val", perplexity, usage_rate)

                if local_rank == 0:
                    best_fgd_val, best_fgd_iteration_val = save_last_and_best_ckpt(
                        motion_vq, optimizers, lr_schedulers, iteration, experiment_ckpt_dir, best_fgd_val, best_fgd_iteration_val, val_loss_dict["fgd"], lower_is_better=True, mertic_name="fgd")

            # train
            data_time = time.time() - data_start
            loss_dict, perplexity, _ = train_val_fn(cfg, batch, motion_vq, device, mode="train", optimizer=optimizers, lr_scheduler=lr_schedulers, ClsFn=ClsFn, iteration=iteration)
            net_time = time.time() - data_start - data_time
            log_train_val(cfg, loss_dict, local_rank, loss_meters, pbar, epoch, max_epochs, iteration, net_time, data_time, optimizers, "Train", perplexity)
            data_start = time.time()

            iteration += 1
   
        start_step_in_epoch = 0
        epoch += 1
        time.sleep(0.003)

    if local_rank == 0 and cfg.validation.wandb:
        wandb.finish()
    torch.distributed.destroy_process_group()


# ---------------------------------  utils fn here --------------------------------- #
def evaluation_fn(joint_mask, gt_list, pred_list, fgd_evaluator, bc_evaluator, l1_evaluator, device, lvd_evaluator, mse_evaluator):
    fgd_evaluator.reset()
    bc_evaluator.reset()
    l1_evaluator.reset()
    lvd_evaluator.reset()
    mse_evaluator.reset()

    for test_file in tqdm(gt_list, desc="Evaluation"):
        # only load selective joints
        pred_file = [item for item in pred_list if item["video_id"] == test_file["video_id"]][0]
        if not pred_file:
            print(f"Missing prediction for {test_file['video_id']}")
            continue
        # print(test_file["motion_path"], pred_file["motion_path"])
        gt_dict = beat_format_load(test_file["motion_path"], joint_mask)
        pred_dict = beat_format_load(pred_file["motion_path"], joint_mask)

        motion_gt = gt_dict["poses"]
        motion_pred = pred_dict["poses"]
        expressions_gt = gt_dict["expressions"]
        expressions_pred = pred_dict["expressions"]
        betas = gt_dict["betas"]
        # motion_gt = recover_from_mask(motion_gt, joint_mask) # t1*165
        # motion_pred = recover_from_mask(motion_pred, joint_mask) # t2*165
        
        t = min(motion_gt.shape[0], motion_pred.shape[0])
        motion_gt = motion_gt[:t]
        motion_pred = motion_pred[:t]
        expressions_gt = expressions_gt[:t]
        expressions_pred = expressions_pred[:t]

        # bc and l1 require position representation
        motion_position_pred = get_motion_rep_numpy(motion_pred, device=device, betas=betas)["position"] # t*55*3
        motion_position_pred = motion_position_pred.reshape(t, -1)
        # ignore the start and end 2s, this may for beat dataset only
        audio_beat = bc_evaluator.load_audio(test_file["audio_path"], t_start=2 * 16000, t_end=int((t-60)/30*16000))
        motion_beat = bc_evaluator.load_motion(motion_position_pred, t_start=60, t_end=t-60, pose_fps=30, without_file=True)
        bc_evaluator.compute(audio_beat, motion_beat, length=t-120, pose_fps=30)
        # audio_beat = bc_evaluator.load_audio(test_file["audio_path"], t_start=0 * 16000, t_end=int((t-0)/30*16000))
        # motion_beat = bc_evaluator.load_motion(motion_position_pred, t_start=0, t_end=t-0, pose_fps=30, without_file=True)
        # bc_evaluator.compute(audio_beat, motion_beat, length=t-0, pose_fps=30)

        l1_evaluator.compute(motion_position_pred)

        face_position_pred = get_motion_rep_numpy(motion_pred, device=device, expressions=expressions_pred, expression_only=True, betas=betas)["vertices"] # t -1
        face_position_gt = get_motion_rep_numpy(motion_gt, device=device, expressions=expressions_gt, expression_only=True, betas=betas)["vertices"]
        lvd_evaluator.compute(face_position_pred, face_position_gt)
        mse_evaluator.compute(face_position_pred, face_position_gt)

        # fgd requires rotation 6d representaiton
        motion_gt = torch.from_numpy(motion_gt).to(device).unsqueeze(0)
        motion_pred = torch.from_numpy(motion_pred).to(device).unsqueeze(0)
        motion_gt = rc.axis_angle_to_rotation_6d(motion_gt.reshape(1, t, 55, 3)).reshape(1, t, 55*6)
        motion_pred = rc.axis_angle_to_rotation_6d(motion_pred.reshape(1, t, 55, 3)).reshape(1, t, 55*6)
        fgd_evaluator.update(motion_pred.float(), motion_gt.float())
       
    metrics = {}
    metrics["fgd"] = fgd_evaluator.compute()
    metrics["bc"] = bc_evaluator.avg()
    metrics["l1"] = l1_evaluator.avg()
    metrics["lvd"] = lvd_evaluator.avg()
    metrics["mse"] = mse_evaluator.avg()
    return metrics


def visualization_fn(pred_list, save_path, gt_list=None, only_check_one=True):
    if gt_list is None: # single visualization
        for i in range(len(pred_list)):
            fast_render.render_one_sequence(
                pred_list[i]["motion_path"],
                save_path,
                pred_list[i]["audio_path"],
                model_folder="./evaluation/smplx_models/",
            )
            if only_check_one: break
    else: # paired visualization, pad the translation
        for i in range(len(pred_list)):
            npz_pred = np.load(pred_list[i]["motion_path"], allow_pickle=True)
            gt_file = [item for item in gt_list if item["video_id"] == pred_list[i]["video_id"]][0]
            if not gt_file:
                print(f"Missing prediction for {pred_list[i]['video_id']}")
                continue
            npz_gt = np.load(gt_file["motion_path"], allow_pickle=True)
            t  = npz_gt["poses"].shape[0]
            np.savez(
                os.path.join(save_path, f"{pred_list[i]['video_id']}_transpad.npz"),
                betas=npz_pred['betas'][:t],
                poses=npz_pred['poses'][:t],
                expressions=npz_pred['expressions'][:t],
                trans=npz_pred["trans"][:t],
                model='smplx2020',
                gender='neutral',
                mocap_frame_rate=30,
            )
            fast_render.render_one_sequence(
                os.path.join(save_path, f"{pred_list[i]['video_id']}_transpad.npz"),
                gt_file["motion_path"],
                save_path,
                pred_list[i]["audio_path"],
                model_folder="./evaluation/smplx_models/",
            )
            if only_check_one: break
     

def log_test(model, metrics, iteration, best_mertics, best_iteration, cfg, local_rank, experiment_ckpt_dir, video_save_path=None):
    if local_rank == 0:
        print(f"\n Test Results at iteration {iteration}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.10f}")
        if cfg.validation.wandb:
            for key, value in metrics.items():
                wandb.log({f"test/{key}": value}, step=iteration)
        if cfg.validation.wandb and cfg.validation.visualization:
            videos_to_log = []
            for filename in os.listdir(video_save_path):
                if filename.endswith(".mp4"):
                    videos_to_log.append(wandb.Video(os.path.join(video_save_path, filename)))
            if videos_to_log:
                wandb.log({"test/videos": videos_to_log}, step=iteration)
        if metrics["fgd"] < best_mertics:
            best_mertics = metrics["fgd"]
            best_iteration = iteration
            model.module.save_pretrained(os.path.join(experiment_ckpt_dir, "test_best"))
        # print(metrics, best_mertics, best_iteration)
        message = f"Current Test FGD: {metrics['fgd']:.4f} (Best: {best_mertics:.4f} at iteration {best_iteration})"
        log_metric_with_box(message)
    return best_mertics, best_iteration


def log_metric_with_box(message):
    box_width = len(message) + 2
    border = "-" * box_width
    print(f"\n{border}")
    print(f"|{message}|")
    print(f"{border}\n")


def log_train_val(cfg, loss_dict, local_rank, loss_meters, pbar, epoch, max_epochs, iteration, net_time, data_time, optimizer, ptype="Train", perplexity=None, usage_rate=None):
    new_loss_dict = {}
    for k, v in loss_dict.items():
        if "fgd" in k: continue
        v_cpu = torch.as_tensor(v).float().cpu().item()
        if k not in loss_meters:
            loss_meters[k] = {"sum":0,"count":0}
        loss_meters[k]["sum"] += v_cpu
        loss_meters[k]["count"] += 1
        new_loss_dict[k] = v_cpu
    mem_used = torch.cuda.memory_reserved() / 1E9
    loss_str = " ".join([f"{k}: {new_loss_dict[k]:.4f}({loss_meters[k]['sum']/loss_meters[k]['count']:.4f})" for k in new_loss_dict])
    if isinstance(optimizer, dict):
        lr_str = ' '.join([f'{k}: {optimizer[k].param_groups[0]["lr"]:.2E}' for k in optimizer.keys()])
    else:
        lr_str = f'{optimizer.param_groups[0]["lr"]:.2E}'
    desc = f"{ptype}: Epoch[{epoch}/{max_epochs}] Iter[{iteration}] {loss_str} lr: {lr_str} data_time: {data_time:.3f} net_time: {net_time:.3f} mem: {mem_used:.2f}GB "
    pbar.set_description(desc)
    pbar.bar_format = "{desc} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    if cfg.validation.wandb and local_rank == 0:
        for k, v in new_loss_dict.items():
            wandb.log({f"loss/{ptype}/{k}": v}, step=iteration)
        if perplexity is not None:
            for k, v in perplexity.items():
                if isinstance(v, list):
                    for i, p in enumerate(v):
                        wandb.log({f"perplexity/{ptype}/{k}_layer_{i}": p}, step=iteration)
                else:
                    wandb.log({f'wandb/{ptype}/{k}': v}, step=iteration)
        if usage_rate is not None:
            for k, v in usage_rate.items():
                if isinstance(v, list):
                    for i, u in enumerate(v):
                        wandb.log({f'usage_rate/{ptype}/{k}_layer_{i}': u}, step=iteration)
                else:
                    wandb.log({f'usage_rate/{ptype}/{k}': v}, step=iteration)
        if 'fgd' in loss_dict.keys():
            wandb.log({f'FGD': loss_dict['fgd']}, step=iteration)


def save_last_and_best_ckpt(model, optimizer, lr_scheduler, iteration, save_dir, previous_best, best_iteration, current, lower_is_better=True, mertic_name="fgd"):
    checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {k: optimizer[k].state_dict() for k in optimizer.keys()} if isinstance(optimizer, dict) else optimizer.state_dict(),
            "lr_scheduler_state_dict": {k: lr_scheduler[k].state_dict() for k in lr_scheduler.keys()} if isinstance(lr_scheduler, dict) else lr_scheduler.state_dict(),
            "iteration": iteration,
        }
    torch.save(checkpoint, os.path.join(save_dir, "last.bin"))
    model.module.save_pretrained(os.path.join(save_dir, "last"))
    if (lower_is_better and current < previous_best) or (not lower_is_better and current > previous_best):
        previous_best = current
        best_iteration = iteration
        shutil.copy(os.path.join(save_dir, "last.bin"), os.path.join(save_dir, "best.bin"))
        model.module.save_pretrained(os.path.join(save_dir, "best"))
    message = f"Current interation {iteration} {mertic_name}: {current:.4f} (Best: {previous_best:.4f} at iteration {best_iteration})"
    log_metric_with_box(message)
    return previous_best, best_iteration


def init_hf_class(module_name, class_name, config, **kwargs):
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    config_class = model_class.config_class
    config = config_class(config_obj=config)
    instance = model_class(config, **kwargs)
    return instance


def init_class(module_name, class_name, config, **kwargs):
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    instance = model_class(config, **kwargs)
    return instance


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def init_env():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/stage2.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--visualization", action="store_true")
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--resume_local_from", type=str, default=None)
    parser.add_argument("--peft", type=str, default=None)
    parser.add_argument('overrides', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config.exp_name = os.path.splitext(os.path.basename(args.config))[0]

    if args.overrides: config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    if args.debug:
        config.wandb_project = "debug"
        config.exp_name = "debug"
        config.solver.max_train_steps = 4
    else:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        config.exp_name = config.exp_name + "_" + run_time
    if args.wandb:
        config.validation.wandb = True
    if args.visualization:
        config.validation.visualization = True
    if args.evaluation:
        config.validation.evaluation = True
    if args.test:
        config.test = True
    if args.finetune:
        config.finetune = True
    if args.resume_local_from:
        config.local_pretrained = f"./emage_weights/{args.resume_local_from}/best"
    if args.peft:
        config.peft = args.peft
    assert config.peft is None or str.lower(config.peft) in ["lora"], f"Unsupported PEFT method {config.peft}"
    save_dir = os.path.join(config.output_dir, config.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    sanity_check_dir = os.path.join(save_dir, 'sanity_check')
    os.makedirs(sanity_check_dir, exist_ok=True)
    with open(os.path.join(sanity_check_dir, f'{config.exp_name}.yaml'), 'w') as f:
        OmegaConf.save(config, f)
    current_dir = Path.cwd()
    py_files = []
    for py_file in current_dir.rglob('*.py'):
        if save_dir.split('/')[1] not in str(py_file):
            py_files.append(py_file)
    for py_file in py_files:
        if sanity_check_dir not in str(py_file):
            dest_path = Path(sanity_check_dir) / py_file.relative_to(current_dir)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(py_file, dest_path)
    return config


if __name__ == "__main__":
    config = init_env()
    main(config)