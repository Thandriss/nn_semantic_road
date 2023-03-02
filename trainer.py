import os
import torch
import argparse
from collections import OrderedDict
from core.config import cfg
from core.modelling.model import build_model
from core.data import make_data_loader
from core.engine.train import do_train
from core.data.datasets import build_dataset


def train_model(cfg, args):
    os.environ['CUDA_VISIBLE_DEVICES'] ='0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_ENABLE_COREDUMP_ON_EXCEPTION'] = '1'

    device = torch.device(cfg.MODEL.DEVICE)

    # Create model
    model = build_model(cfg)
    model.to(device)

    # Create training data
    data_loader = make_data_loader(cfg, is_train=True)


    # Calculate class weigths
    class_weights = None
# Create checkpointer
    arguments = {"epoch": 0}

    # Train model
    model = do_train(cfg, model, data_loader, class_weights, device,arguments, args)

    return model


def str2bool(s):
    return s.lower() in ('true', '1')


def main() -> int:
    # Create argument parser
    parser = argparse.ArgumentParser(description='Semantic Segmentation Model Training With PyTorch')
    parser.add_argument("--config-file", dest="config_file", required=True, type=str, default="", metavar="FILE",
                        help="path to config file")
    parser.add_argument('--log-step', dest="log_step", required=False, type=int, default=1,
                        help='Print logs every log_step')
    parser.add_argument('--save-step', dest="save_step", required=False, type=int, default=1,
                        help='Save checkpoint every save_step')
    parser.add_argument('--eval-step', dest="eval_step", required=False, type=int, default=1,
                        help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--use-tensorboard', dest="use_tensorboard", required=False, default=False, type=str2bool,
                        help='Use tensorboard summary writer')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()
    NUM_GPUS = 1
    args.distributed = False
    args.num_gpus = NUM_GPUS

    # Enable cudnn auto-tuner to find the best algorithm to use for your hardware.
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Create config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


    # Create config backup
    with open(os.path.join(cfg.OUTPUT_DIR, 'cfg.yaml'), "w") as cfg_dump:
        cfg_dump.write(str(cfg))

    # Train model
    model = train_model(cfg, args)

    return 0

if __name__ == '__main__':
    exit(main())