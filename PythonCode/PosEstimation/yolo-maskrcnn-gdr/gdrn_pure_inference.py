"""
GDR-Net "Pure" inference shim (single-file)
Place this script in the root of the GDR-Net repository (alongside `core/`)
Run in your `deeplearning` conda env: python gdrn_pure_inference.py --cfg configs/... --weights path/to/gdrn_lm.pth --image path/to/img.png --ply models/obj_01.ply

What this script does:
- Provides tiny shims for `detectron2` and `mmcv.runner.optimizer` used by older GDR-Net code so you don't need to install Detectron2.
- Loads the GDR-Net model class (tries several common paths) and the provided .pth weights (will try to adapt key prefixes).
- Runs a minimal forward pass (inference) and expects the model to return R,t,mask or a dict with these keys. Adapt as needed.
- Loads a .ply file and visualizes projected point cloud / bbox / axes on the image using Open3D + OpenCV.

LIMITATIONS / WARNINGS:
- I cannot run or test this on your machine. You may need to tweak imports, class names, and key mapping depending on the exact GDR-Net fork you have.
- This is a best-effort shim to avoid installing Detectron2. If the original code relies heavily on D2 features, further edits will be required.

Requirements (install in conda env):
pip install torch torchvision opencv-python open3d numpy pyyaml

"""

import os
import sys
import argparse
import importlib
import types
import torch
import numpy as np
import cv2

# ---------------------------
# 1) Lightweight shims
# ---------------------------
# If detectron2 is imported by the GDR-Net code, provide minimal fake module(s)
# and mmcv.runner.optimizer shim to satisfy imports used by GDR-Net's solver_utils.

def make_detectron2_shim():
    """Create very small detectron2 shim in sys.modules to satisfy imports.
    The shim exposes: model_zoo-like config helper, model registry placeholders if needed.
    """
    if 'detectron2' in sys.modules:
        return
    d2 = types.ModuleType('detectron2')
    # Provide submodules commonly imported
    d2.modeling = types.SimpleNamespace()
    d2.config = types.SimpleNamespace()
    d2.utils = types.SimpleNamespace()
    d2.data = types.SimpleNamespace()
    # Minimal registry placeholder
    class DummyRegistry(dict):
        def register(self, *args, **kwargs):
            def _wrap(x):
                return x
            return _wrap
    d2.modeling.META_ARCH_REGISTRY = DummyRegistry()

    sys.modules['detectron2'] = d2
    sys.modules['detectron2.modeling'] = d2.modeling
    sys.modules['detectron2.config'] = d2.config
    sys.modules['detectron2.utils'] = d2.utils
    sys.modules['detectron2.data'] = d2.data


def make_mmcv_runner_optimizer_shim():
    """Provide mmcv.runner.optimizer with a lightweight build_optimizer implementation.
    Some GDR-Net forks import: from mmcv.runner.optimizer import OPTIMIZERS, DefaultOptimizerConstructor, build_optimizer
    We implement a simple build_optimizer that returns torch.optim.Adam/SGD.
    """
    if 'mmcv' in sys.modules:
        # If a real mmcv exists, don't override
        return
    mmcv = types.ModuleType('mmcv')
    runner = types.ModuleType('mmcv.runner')

    OPTIMIZERS = {}

    class DefaultOptimizerConstructor:
        def __init__(self, params):
            self._params = params
        def __call__(self, model):
            # trivial: return an optimizer on model.parameters()
            return torch.optim.Adam(model.parameters(), lr=0.001)

    def build_optimizer(model, cfg=None):
        # cfg is ignored; choose Adam by default
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=1e-4)

    runner.OPTIMIZERS = OPTIMIZERS
    runner.DefaultOptimizerConstructor = DefaultOptimizerConstructor
    runner.build_optimizer = build_optimizer

    sys.modules['mmcv'] = mmcv
    sys.modules['mmcv.runner'] = runner


# apply shims before importing GDR-Net modules
make_detectron2_shim()
make_mmcv_runner_optimizer_shim()

# ---------------------------
# 2) Utilities: load PLY, project, draw
# ---------------------------

def load_ply_points(ply_path):
    """Read vertex positions from a simple PLY (ascii or binary) using Open3D if available, else parse minimal PLY.
    Returns Nx3 numpy array.
    """
    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(ply_path)
        if mesh.is_empty():
            pcd = o3d.io.read_point_cloud(ply_path)
            pts = np.asarray(pcd.points)
        else:
            pts = np.asarray(mesh.vertices)
        return pts
    except Exception as e:
        # naive PLY parser for vertex lists (ASCII)
        pts = []
        with open(ply_path, 'r', encoding='utf-8', errors='ignore') as f:
            header = True
            num_verts = 0
            while header:
                line = f.readline()
                if line.startswith('element vertex'):
                    num_verts = int(line.split()[-1])
                if line.strip() == 'end_header':
                    break
            for i in range(num_verts):
                line = f.readline().strip()
                if not line:
                    break
                vals = line.split()
                if len(vals) >= 3:
                    pts.append([float(vals[0]), float(vals[1]), float(vals[2])])
        return np.array(pts, dtype=np.float32)


def project_points(pts3d, K, R, t):
    """Project Nx3 pts (model coordinates) onto image plane using camera intrinsics K and pose (R,t).
    returns Nx2 float coords and a depth vector.
    """
    if pts3d.shape[1] != 3:
        pts3d = pts3d.reshape(-1, 3)
    pts_cam = (R @ pts3d.T) + t.reshape(3,1)
    zs = pts_cam[2, :].copy()
    pts_proj = (K @ pts_cam).T
    pts2 = pts_proj[:, :2] / pts_proj[:, 2:3]
    return pts2, zs


def draw_projection(img, pts2, color=(0,255,0)):
    img = img.copy()
    for (x,y) in pts2.astype(np.int32):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (int(x),int(y)), 1, color, -1)
    return img


def draw_axes(img, K, R, t, length=0.05):
    # axes in model frame
    axes = np.array([[0,0,0], [length,0,0], [0,length,0], [0,0,length]], dtype=np.float32)
    pts2, _ = project_points(axes, K, R, t)
    img = img.copy()
    o = tuple(pts2[0].astype(int))
    x_p = tuple(pts2[1].astype(int))
    y_p = tuple(pts2[2].astype(int))
    z_p = tuple(pts2[3].astype(int))
    cv2.line(img, o, x_p, (0,0,255), 3)
    cv2.line(img, o, y_p, (0,255,0), 3)
    cv2.line(img, o, z_p, (255,0,0), 3)
    return img

# ---------------------------
# 3) Model loader helpers
# ---------------------------

def try_import_model_class():
    """Try several likely import paths for the GDRN class used by different forks.
    Returns model class or raises ImportError.
    """
    names = [
        'core.gdrn_modeling.models.GDRN.GDRN',
        'core.gdrn_modeling.models.GDRN.GDRNModel',
        'core.gdrn_modeling.models.gdrn.GDRN',
        'gdrn.core.models.GDRN.GDRN',
        'models.GDRN.GDRN',
    ]
    for n in names:
        module_name, _, cls = n.rpartition('.')
        try:
            mod = importlib.import_module(module_name)
            model_cls = getattr(mod, cls)
            print(f"Found model class: {n}")
            return model_cls
        except Exception:
            continue
    # fallback: try to import any class named GDRN in repo
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read()
                    if 'class GDRN' in txt or 'class GDRNModel' in txt:
                        rel = os.path.relpath(path, '.')[:-3].replace(os.sep, '.')
                        try:
                            mod = importlib.import_module(rel)
                            if hasattr(mod, 'GDRN'):
                                print('Found GDRN in', rel)
                                return getattr(mod, 'GDRN')
                            if hasattr(mod, 'GDRNModel'):
                                return getattr(mod, 'GDRNModel')
                        except Exception:
                            pass
    raise ImportError('Could not locate GDRN model class automatically. Please edit the script to point to your model class.')


def load_state_dict_adapt(model, weights_path, device):
    sd = torch.load(weights_path, map_location='cpu')
    # sd might be a dict with 'model' or 'state_dict'
    if isinstance(sd, dict):
        if 'state_dict' in sd:
            sd = sd['state_dict']
        elif 'model' in sd and isinstance(sd['model'], dict):
            sd = sd['model']
    # remove possible 'module.' prefix
    new_sd = {}
    for k,v in sd.items():
        new_k = k
        if k.startswith('module.'):
            new_k = k[len('module.'):]
        # sometimes detectron2 stores model weights under 'backbone.' etc; try to keep
        new_sd[new_k] = v
    # attempt to load
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print('Loaded weights. missing_keys:', missing, 'unexpected_keys:', unexpected)
    return model

# ---------------------------
# 4) Main inference flow
# ---------------------------

def run_inference(args):
    # load image
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError('Image not found: ' + args.image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # camera intrinsics: user can pass YAML or a simple comma list
    if args.K is not None:
        K = np.array(args.K.split(','), dtype=float).reshape(3,3)
    else:
        # fallback standard LM intrinsics (example) - please change
        K = np.array([[572.4114, 0, 325.2611],[0,573.57043,242.04899],[0,0,1]], dtype=float)
    # load model class
    ModelCls = try_import_model_class()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Constructing model...')
    try:
        model = ModelCls()  # many forks use default constructor
    except Exception as e:
        # try to call with cfg
        print('Default constructor failed, trying with cfg path...')
        try:
            import yaml
            cfg = yaml.safe_load(open(args.cfg)) if args.cfg else {}
            model = ModelCls(cfg)
        except Exception as e2:
            print('Could not instantiate model automatically:', e, e2)
            raise
    model.to(device)
    model.eval()

    # load weights
    load_state_dict_adapt(model, args.weights, device)

    # prepare input tensor - many implementations accept crop x of size HxW; we pass whole image
    inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    inp_t = torch.from_numpy(inp).permute(2,0,1).unsqueeze(0).to(device).float()

    with torch.no_grad():
        # heuristic: try calling model(inp_t) or model.forward_test / model.inference
        out = None
        try:
            out = model(inp_t)
        except Exception as e:
            print('model(input) failed, trying model.forward_test...')
            try:
                out = model.forward_test(inp_t)
            except Exception:
                print('model.forward_test failed, trying model.inference...')
                try:
                    out = model.inference(inp_t)
                except Exception as e2:
                    print('All model call attempts failed. Error:', e, e2)
                    raise
    print('Raw model output type:', type(out))

    # Expect out to be dict with keys 'R', 't' or similar. Try to recover.
    if isinstance(out, (list, tuple)):
        # pick first element
        out = out[0]
    if isinstance(out, torch.Tensor):
        print('Model returned tensor; cannot interpret as pose. Please adapt the script to your model')
        return
    # normalize keys
    outd = {}
    if isinstance(out, dict):
        for k in out.keys():
            outd[k.lower()] = out[k]
    else:
        # unknown structure
        raise RuntimeError('Unsupported model output type; please inspect and adapt script')

    # try to get R and t
    if 'r' in outd and 't' in outd:
        R = outd['r']
        t = outd['t']
    elif 'rot' in outd and 'trans' in outd:
        R = outd['rot']
        t = outd['trans']
    elif 'pred_rot' in outd or 'pred_r' in outd:
        R = outd.get('pred_rot', outd.get('pred_r'))
        t = outd.get('pred_t', outd.get('pred_trans'))
    else:
        print('Could not find R/t in model output keys:', list(outd.keys()))
        # maybe model returns pose as quaternion+trans
        if 'quat' in outd and 'trans' in outd:
            quat = outd['quat']
            t = outd['trans']
            # convert quat to R
            q = quat.cpu().numpy().reshape(4)
            # w,x,y,z or x,y,z,w ? User must adapt
            R = quat_to_rotmat(q)
        else:
            raise RuntimeError('No recognizable pose in model output; please adapt script')

    # ensure tensors -> numpy
    if isinstance(R, torch.Tensor):
        R = R.cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy().reshape(3)
    # if R is shape (N,3,3) pick first
    if R.ndim == 4 and R.shape[0] == 1:
        R = R[0]
    if R.ndim == 3 and R.shape[0] == 1:
        R = R[0]
    t = t.reshape(3,)

    print('Recovered pose R shape', R.shape, 't', t)

    # load ply
    pts3d = load_ply_points(args.ply)
    pts2, zs = project_points(pts3d, K, R, t)
    vis = draw_projection(img, pts2)
    vis = draw_axes(vis, K, R, t)

    outpath = os.path.splitext(args.image)[0] + '_gdrn_vis.png'
    cv2.imwrite(outpath, vis)
    print('Wrote visualization to', outpath)

# helper quat->rot

def quat_to_rotmat(q):
    # expecting [w,x,y,z]
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=float)
    return R

# ---------------------------
# CLI
# ---------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--ply', required=True)
    p.add_argument('--cfg', required=False, help='optional cfg file path for model constructor')
    p.add_argument('--K', required=False, help='camera intrinsics as comma list, 9 values')
    args = p.parse_args()
    run_inference(args)
