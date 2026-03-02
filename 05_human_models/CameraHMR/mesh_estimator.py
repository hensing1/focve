import cv2
import os
import json
import torch
import smplx
import trimesh
import numpy as np
from glob import glob
from torchvision.transforms import Normalize
from detectron2.config import LazyConfig
from core.utils.utils_detectron2 import DefaultPredictor_Lazy

from core.camerahmr_model import CameraHMR
from core.constants import CHECKPOINT_PATH, CAM_MODEL_CKPT, SMPL_MODEL_PATH, DETECTRON_CKPT, DETECTRON_CFG
from core.datasets.dataset import Dataset
from core.utils.renderer_pyrd import Renderer
from core.utils import recursive_to
from core.utils.geometry import batch_rot2aa
from core.cam_model.fl_net import FLNet
from core.constants import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, NUM_BETAS

def resize_image(img, target_size):
    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Calculate the new size while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image using OpenCV
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new blank image with the target size
    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Paste the resized image onto the blank image, centering it
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y:start_y + new_height, start_x:start_x + new_width] = resized_img

    return aspect_ratio, final_img

class HumanMeshEstimator:
    def __init__(self, smpl_model_path=SMPL_MODEL_PATH, threshold=0.25):
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = self.init_model()
        self.detector = self.init_detector(threshold)
        self.cam_model = self.init_cam_model()
        self.smpl_model = smplx.SMPLLayer(model_path=smpl_model_path, num_betas=NUM_BETAS).to(self.device)
        self.normalize_img = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)

    def init_cam_model(self):
        model = FLNet()
        checkpoint = torch.load(CAM_MODEL_CKPT)['state_dict']
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def init_model(self):
        model = CameraHMR.load_from_checkpoint(CHECKPOINT_PATH, strict=False, map_location='cpu')
        model = model.to(self.device)
        model.eval()
        return model
    
    def init_detector(self, threshold):

        detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
        detectron2_cfg.train.init_checkpoint = DETECTRON_CKPT
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = threshold
        detector = DefaultPredictor_Lazy(detectron2_cfg)
        return detector

    
    def convert_to_full_img_cam(self, pare_cam, bbox_height, bbox_center, img_w, img_h, focal_length):
        s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
        tz = 2. * focal_length / (bbox_height * s)
        cx = 2. * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
        cy = 2. * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)
        cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
        return cam_t

    def get_output_mesh(self, params, pred_cam, batch):
        smpl_output = self.smpl_model(**{k: v.float() for k, v in params.items()})
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        img_h, img_w = batch['img_size'][0]
        cam_trans = self.convert_to_full_img_cam(
            pare_cam=pred_cam,
            bbox_height=batch['box_size'],
            bbox_center=batch['box_center'],
            img_w=img_w,
            img_h=img_h,
            focal_length=batch['cam_int'][:, 0, 0]
        )
        return pred_vertices, pred_keypoints_3d, cam_trans

    def get_cam_intrinsics(self, img):
        img_h, img_w, c = img.shape
        aspect_ratio, img_full_resized = resize_image(img, IMAGE_SIZE)
        img_full_resized = np.transpose(img_full_resized.astype('float32'),
                            (2, 0, 1))/255.0
        img_full_resized = self.normalize_img(torch.from_numpy(img_full_resized).float())

        estimated_fov, _ = self.cam_model(img_full_resized.unsqueeze(0))
        vfov = estimated_fov[0, 1]
        fl_h = (img_h / (2 * torch.tan(vfov / 2))).item()
        # fl_h = (img_w * img_w + img_h * img_h) ** 0.5
        cam_int = np.array([[fl_h, 0, img_w/2], [0, fl_h, img_h / 2], [0, 0, 1]]).astype(np.float32)
        return cam_int


    def remove_pelvis_rotation(self, smpl):
        """We don't trust the body orientation coming out of bedlam_cliff, so we're just going to zero it out."""
        smpl.body_pose[0][0][:] = np.zeros(3)


    def _get_calib_data(self, img_path):
        calib_dome_path = os.path.join(os.path.dirname(img_path), 'calibration_dome.json')
        if os.path.exists(calib_dome_path):
            with open(calib_dome_path, 'rt') as f:
                calib_data = json.load(f)
            calib_cameras = calib_data['cameras']
            basename = os.path.splitext(os.path.basename(img_path))[0]
            for calib_cam in calib_cameras:
                if calib_cam['camera_id'] in basename:
                    print(f"Found camera {calib_cam['camera_id']} for {img_path}")
                    return calib_cam
        print("No calibration data found, using HumanFoV estimate...")
        return None

    def _get_cam_intrinsics_from_calib_data(self, img, calib_data, upright_images=True, up=np.array([-1, 0, 0])):
        # Actual image size (usually 2x downsampled)
        img_h, img_w, c = img.shape

        camera_matrix = np.array(calib_data['intrinsics']['camera_matrix'], dtype=np.float32).reshape(3, 3)
        view_matrix = np.array(calib_data['extrinsics']['view_matrix'], dtype=np.float32).reshape(4, 4)

        # Reference image size for camera matrix from calibration
        res_w, res_h = calib_data['intrinsics']['resolution']

        # The cameras are more or less randomly rotated by +-90°.
        # Detect the rotation of the camera, and adjust the camera matrix here, to match the (already un-rotated!) input images.
        if upright_images:
            # Swap reference image width/height
            res_h, res_w = res_w, res_h

            # Check if we have to rotate left or right?
            local_up = view_matrix[0:3, 0:3] @ up
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            if local_up[0] < 0:
                # Build new rotated camera matrix
                camera_matrix = np.array([
                    [fy, 0, img_h-cy],
                    [0, fx, cx],
                    [0, 0, 1]
                ], dtype=np.float32)
            else:
                # Build new rotated camera matrix
                camera_matrix = np.array([
                    [fy, 0, cy],
                    [0, fx, img_w-cx],
                    [0, 0, 1]
                ], dtype=np.float32)

        # Account for image size in camera matrix (it is now wrt. img_w, img_h)
        res_scale = np.array([ img_w/res_w, img_h/res_h, 1 ], dtype=np.float32)
        camera_matrix = camera_matrix * res_scale[:, None] # Scale each row separately

        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        return camera_matrix


    def process_image(self, img_path, output_img_folder, i):
        img_cv2 = cv2.imread(str(img_path))
        
        fname, img_ext = os.path.splitext(os.path.basename(img_path))
        overlay_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}{img_ext}') #_{i:06d}
        json_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_{{:04d}}.json') #_{i:06d}
        mesh_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}.obj') #_{i:06d}

        # Detect humans in the image
        det_out = self.detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0 
        bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0

        calib_data = self._get_calib_data(str(img_path))
        if calib_data is not None:
            # Get Camera intrinsics from dome calibration data
            cam_int = self._get_cam_intrinsics_from_calib_data(img_cv2, calib_data)
        else:
            # Get Camera intrinsics using HumanFoV Model
            cam_int = self.get_cam_intrinsics(img_cv2)
        dataset = Dataset(img_cv2, bbox_center, bbox_scale, cam_int, False, img_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=10)

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            img_h, img_w = batch['img_size'][0]
            with torch.no_grad():
                out_smpl_params, out_cam, focal_length_ = self.model(batch)

            output_vertices, output_joints, output_cam_trans = self.get_output_mesh(out_smpl_params, out_cam, batch)

            smpl_global_orient = out_smpl_params['global_orient']
            smpl_betas = out_smpl_params['betas']
            smpl_joints = out_smpl_params['body_pose']
            for person_index in range(smpl_joints.shape[0]):
                with open(json_fname.format(person_index), 'wt') as f:
                    # Convert parameters into Python arrays
                    betas_list = smpl_betas[person_index].tolist()
                    joints_list = [
                        smpl_joints[person_index, joint_index].flatten().tolist()
                        for joint_index in range(smpl_joints.shape[1])
                    ]
                    global_orient_list = smpl_global_orient[person_index, 0].flatten().tolist()
                    cam_trans_list = output_cam_trans[person_index].tolist()
                    # Assemble output dictionary
                    out_dict = {
                        'betas': betas_list,
                        'joints': joints_list,
                        'root_rot': global_orient_list,
                        'root_trans': cam_trans_list,
                    }
                    # Dump parameters to file
                    json.dump(out_dict, f, indent=2)


            mesh = trimesh.Trimesh(output_vertices[0].cpu().numpy() , self.smpl_model.faces,
                            process=False)
            mesh.export(mesh_fname)

            # Render overlay
            focal_length = (focal_length_[0], focal_length_[0])
            pred_vertices_array = (output_vertices + output_cam_trans.unsqueeze(1)).detach().cpu().numpy()
            renderer = Renderer(focal_length=focal_length[0], img_w=img_w, img_h=img_h, faces=self.smpl_model.faces, same_mesh_color=True)
            front_view = renderer.render_front_view(pred_vertices_array, bg_img_rgb=img_cv2.copy())
            final_img = front_view
            # Write overlay
            cv2.imwrite(overlay_fname, final_img)
            renderer.delete()


    def run_on_images(self, image_folder, out_folder):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
        images_list = [image for ext in image_extensions for image in glob(os.path.join(image_folder, ext))]
        images_list.sort()
        for ind, img_path in enumerate(images_list):
            self.process_image(img_path, out_folder, ind)
