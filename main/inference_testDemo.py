import os
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2
import datetime
from tqdm import tqdm
from pathlib import Path
from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.visualization_utils import render_mesh
from utils.inference_utils import non_max_suppression
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--file_name', type=str, default='test')
    parser.add_argument('--ckpt_name', type=str, default='model_dump')
    parser.add_argument('--start', type=str, default=1)
    parser.add_argument('--end', type=str, default=1)
    parser.add_argument('--multi_person', action='store_true')
    args = parser.parse_args()
    return args

def save_smplx_params(out, frame_idx, output_folder, person_idx=0):
    """
    Save SMPL-X parameters for one person in one frame to a PKL.
    We **ignore cam_trans** (set transl=0) and also save joints_cam
    so we can realign to world later using registration.
    """

    def extract(key):
        if key not in out:
            return None
        v = out[key]
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        if v.ndim >= 2:
            v = v[person_idx]
        return v.astype(np.float32)

    # body params from SMPLest-X
    global_orient = extract("smplx_root_pose")   # (3,)
    body_pose     = extract("smplx_body_pose")   # (63,)
    jaw_pose      = extract("smplx_jaw_pose")    # (3,)
    lhand_pose    = extract("smplx_lhand_pose")  # (45,)
    rhand_pose    = extract("smplx_rhand_pose")  # (45,)
    betas_raw     = extract("smplx_shape")       # (10,)

    # simplest: use zero-betas (average body) first:
    # betas = np.zeros_like(betas_raw, dtype=np.float32)
    betas= betas_raw
    # later you can replace with per-subject mean betas from registration

    # kill noisy camera translation
    transl = np.zeros(3, dtype=np.float32)
    transl = extract("cam_trans")

    # also save joints in camera frame for later world alignment
    joints_cam = extract("smplx_joint_cam")  # (137,3)

    params = {
        "global_orient":   global_orient,
        "body_pose":       body_pose,
        "jaw_pose":        jaw_pose,
        "left_hand_pose":  lhand_pose,
        "right_hand_pose": rhand_pose,
        "betas":           betas,
        "transl":          transl,
        "joints_cam":      joints_cam,  # extra, for later alignment
        "frame_idx":       int(frame_idx),
    }

    # minimal sanity check
    for k in ["global_orient", "body_pose", "left_hand_pose",
              "right_hand_pose", "betas", "transl"]:
        if params[k] is None:
            raise RuntimeError(f"Missing key '{k}' in out for frame {frame_idx}")

    smplx_out_dir = osp.join(output_folder)
    os.makedirs(smplx_out_dir, exist_ok=True)
    fname = f"mesh-f{frame_idx:05d}_smplx.pkl"
    out_path = osp.join(smplx_out_dir, fname)

    with open(out_path, "wb") as f:
        pickle.dump(params, f, protocol=2)

    
def main():
    args = parse_args()
    cudnn.benchmark = True

    # init config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    root_dir = Path(__file__).resolve().parent.parent
    config_path = osp.join('./pretrained_models', args.ckpt_name, 'config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join('./pretrained_models', args.ckpt_name, f'{args.ckpt_name}.pth.tar')
    img_folder = osp.join(root_dir, 'demo', 'input_frames', args.file_name)
    output_folder = osp.join(root_dir, 'demo', 'output_frames', args.file_name)
    os.makedirs(output_folder, exist_ok=True)
    exp_name = f'inference_{args.file_name}_{args.ckpt_name}_{time_str}'

    new_config = {
        "model": {
            "pretrained_model_path": checkpoint_path,
        },
        "log":{
            'exp_name':  exp_name,
            'log_dir': osp.join(root_dir, 'outputs', exp_name, 'log'),  
            }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()
    
    # init human models
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init tester
    demoer = Tester(cfg)
    demoer.logger.info(f"Using 1 GPU.")
    demoer.logger.info(f'Inference [{args.file_name}] with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # init detector
    bbox_model = getattr(cfg.inference.detection, "model_path", 
                        './pretrained_models/yolov8x.pt')
    detector = YOLO(bbox_model)

    start = int(args.start)
    end = int(args.end) + 1

    for frame in tqdm(range(start, end)):
        
        # prepare input image
        img_path =osp.join(img_folder, f'{int(frame):06d}.jpg')

        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]
        
        # detection, xyxy
        yolo_bbox = detector.predict(original_img, 
                                device='cuda', 
                                classes=00, 
                                conf=cfg.inference.detection.conf, 
                                save=cfg.inference.detection.save, 
                                verbose=cfg.inference.detection.verbose
                                    )[0].boxes.xyxy.detach().cpu().numpy()

        if len(yolo_bbox)<1:
            # save original image if no bbox
            num_bbox = 0
        if not args.multi_person:
            # only select the largest bbox
            num_bbox = 1
            # yolo_bbox = yolo_bbox[0]
        else:
            # keep bbox by NMS with iou_thr
            yolo_bbox = non_max_suppression(yolo_bbox, cfg.inference.detection.iou_thr)
            num_bbox = len(yolo_bbox)

        # loop all detected bboxes
        for bbox_id in range(num_bbox):
            yolo_bbox_xywh = np.zeros((4))
            yolo_bbox_xywh[0] = yolo_bbox[bbox_id][0]
            yolo_bbox_xywh[1] = yolo_bbox[bbox_id][1]
            yolo_bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
            yolo_bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])
            
            # xywh
            bbox = process_bbox(bbox=yolo_bbox_xywh, 
                                img_width=original_img_width, 
                                img_height=original_img_height, 
                                input_img_shape=cfg.model.input_img_shape, 
                                ratio=getattr(cfg.data, "bbox_ratio", 1.25))                
            img, _, _ = generate_patch_image(cvimg=original_img, 
                                                bbox=bbox, 
                                                scale=1.0, 
                                                rot=0.0, 
                                                do_flip=False, 
                                                out_shape=cfg.model.input_img_shape)
                
            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')

            # save SMPL-X parameters for this frame/person
            save_smplx_params(out, frame_idx=frame,
                              output_folder="./demo/smplx_params/test_video",
                              person_idx=bbox_id)
            

            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

            # render mesh
            focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2], 
                     cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
            princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0], 
                       cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]
            
            # draw the bbox on img
            vis_img = cv2.rectangle(vis_img, (int(yolo_bbox[bbox_id][0]), int(yolo_bbox[bbox_id][1])), 
                                    (int(yolo_bbox[bbox_id][2]), int(yolo_bbox[bbox_id][3])), (0, 255, 0), 1)
            # draw mesh
            vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, mesh_as_vertices=False)

        # save rendered image
        frame_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_folder, frame_name), vis_img[:, :, ::-1])
    for k, v in out.items(): 
        print(k, v.shape)

if __name__ == "__main__":
    main()
