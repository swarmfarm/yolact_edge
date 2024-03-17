import os
import sys
import subprocess
# implement pip as a subprocess:
# subprocess.check_call(['python', '-m', 'pip', 'install', 'git+https://github.com/haotian-liu/cocoapi.git'])
# os.system('pip install git+https://github.com/haotian-liu/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"')

import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from eval import *

parse_args(["--config=yolact_edge_config", "--calib_images=../calib_images"])

from eval import args


def load_image(im_filename):
    im_path = Path(im_dir).joinpath(im_filename)
    # print(im_path)
    
    im = Image.open(str(im_path))
    # print(im.size)
    
    im = im.resize(IM_SIZE)
    # print(im.size)
    
    im = np.array(im)
    # print(im.shape)
    
    im = im[:,:,0:3]
    # print(im.shape)
    assert im.shape[2] == 3
    
    return im


def generate_predictions(image):
    
    frame = torch.from_numpy(image).cuda().float()
    # print(frame.shape)

    batch = FastBaseTransform()(frame.unsqueeze(0))

    extras = {"backbone": "full", "interrupt": False, "keep_statistics": False,
              "moving_statistics": None}

    with torch.no_grad():
        preds = net(batch, extras=extras)["pred_outs"]   
    
    return preds, frame


def draw_predictions(predictions, frame):
    # Visualize the predictions
    disp_image = prep_display(predictions, frame, None, None, undo_transform=False)
    # print(disp_image.shape)
    
    return disp_image


logger = logging.getLogger("yolact.eval")
logger.setLevel(logging.INFO)

args.trained_model = "./weights/yolact_edge_54_800000.pth"
args.yolact_transfer = True

torch.set_default_tensor_type('torch.cuda.FloatTensor')

logger.info('Loading model...')
net = Yolact(training=False)
net.load_weights(args.trained_model, args=args)
net.eval()
logger.info('Model loaded.')

net.detect.use_fast_nms = args.fast_nms
cfg.mask_proto_debug = args.mask_proto_debug

args.score_threshold = 0.30
args.top_k = 15

args.output_coco_json = True


IM_DIR_COCO = "/notebooks/yolact_edge/data/coco/images"  # coco
IM_DIR_SF = "/notebooks/yolact_edge/data/swarmfarm/annotation"  # swarmfarm
IM_SIZE = (640, 480)

im_dir = IM_DIR_SF

if im_dir == IM_DIR_COCO:
    file_type = ".jpg"
elif im_dir == IM_DIR_SF:
    file_type = ".png"

image_list = [str(x) for x in list(Path(im_dir).glob(f"**/*{file_type}"))]
# image_list = [x for x in test2017_images]
image_list.sort()
image_list[0:10]

image_dims = Image.open(str(image_list[0])).size


DISP_SCALE = 50
w, h = IM_SIZE
display_size = (h/DISP_SCALE, w/DISP_SCALE)

results_dir = "./results/inference_pretrained_coco/"
Path(results_dir).mkdir(exist_ok=True)

for filename in tqdm(image_list, total=len(image_list)):
    im = load_image(filename)
    
    if image_dims is None: 
        image_dims = im.shape[0:2]
    
    predictions, frame = generate_predictions(im)
    
    output = draw_predictions(predictions, frame)
    
    output_path = Path(results_dir).joinpath(Path(filename).name)
    # Image.fromarray(output).save(output_path)  # .png
    Image.fromarray(output).save(str(output_path).replace(".png", ".jpg"))  # .jpg
    
    break