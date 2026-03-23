import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_Physio
from dataset_packetloss import get_packet_loss_dataloaders
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:1', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.0)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)

args = parser.parse_args()
print(args)

path = "config/" + args.config
print(path)
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/physio_fold" + str(args.nfold) + "_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

CLEAN_DIR  = "/home/ubuntu/projects/ia.novikov/CSDIweather/CSDI/train/clean_signals"
LOSSY_DIR  = "/home/ubuntu/projects/ia.novikov/CSDIweather/CSDI/train/lossy_signals"
TRACES_DIR = "/home/ubuntu/projects/ia.novikov/CSDIweather/CSDI/train/traces"

train_loader, valid_loader, test_loader = get_packet_loss_dataloaders(
    clean_dir=CLEAN_DIR,
    lossy_dir=LOSSY_DIR,
    traces_dir=TRACES_DIR,
    batch_size=8,
    seed=args.seed,
    n_folds=5,
    fold_idx=args.nfold,
    missing_ratio=config["model"]["test_missing_ratio"],  # если хочешь доп. скрытие
    sr=16000,
    n_fft=512,
    hop_length=160,
    win_length=512,
    L=1000,
    return_clean=True,  # включи True если хочешь clean_spec в batch
)

# # Проверяем один батч
# batch = next(iter(train_loader))
# print(f"Batch keys: {batch.keys()}")
# print(f"Observed data shape: {batch['observed_data'].shape}")
# print(f"Observed mask shape: {batch['observed_mask'].shape}")
# print(f"Target data shape: {batch['target_data'].shape}")
# batch = next(iter(train_loader))
# print(f"\n=== ПРОВЕРКА МАСК ИЗ ДАТАСЕТА ===")
# print(f"observed_mask unique values: {torch.unique(batch['observed_mask'])}")
# print(f"observed_mask range: [{batch['observed_mask'].min()}, {batch['observed_mask'].max()}]")
# print(f"gt_mask unique values: {torch.unique(batch['gt_mask'])}")

model = CSDI_Physio(config, args.device, target_dim=256).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(model, valid_loader, nsample=args.nsample, scaler=1, foldername=foldername)
