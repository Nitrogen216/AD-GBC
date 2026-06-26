import argparse
import os
from glob import glob
from pathlib import Path
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import archs_GBC
from dataset import Dataset
from metrics import iou_score, indicators
from utils import AverageMeter
from simple_transforms import SimpleSegTransform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='busi_RU_GBC_L_div0.01_sca0.1', help='model name')
    parser.add_argument('--device', default='auto',
                        help='device to use: auto, cuda, mps, or cpu')
    parser.add_argument('--data_root', default='auto',
                        help='dataset root. auto resolves to ../../Dataset/segmentation when available')
    parser.add_argument('--split_root', default='auto',
                        help='split root. auto resolves to ../../Dataset/splits/segmentation when available')
    parser.add_argument('--artifact_root', default='auto',
                        help='artifact root. auto resolves to ../../Artifacts/AD-GBC when available')
    args = parser.parse_args()
    return args


def select_device(name='auto'):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def empty_device_cache(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()


def find_workspace_root():
    code_root = Path(__file__).resolve().parent
    for candidate in [Path.cwd().resolve(), code_root, *code_root.parents]:
        if (candidate / 'Dataset' / 'segmentation').exists():
            return candidate
        if (candidate / '.aris').exists():
            return candidate
    return code_root.parent.parent


def resolve_data_root(data_root='auto'):
    if data_root and data_root != 'auto':
        return Path(data_root).expanduser().resolve()

    env_root = os.environ.get('AD_GBC_DATA_ROOT')
    if env_root:
        return Path(env_root).expanduser().resolve()

    workspace_root = find_workspace_root()
    code_root = Path(__file__).resolve().parent
    candidates = [
        workspace_root / 'Dataset' / 'segmentation',
        code_root / 'inputs',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_split_root(split_root='auto'):
    if split_root and split_root != 'auto':
        return Path(split_root).expanduser().resolve()

    env_root = os.environ.get('AD_GBC_SPLIT_ROOT')
    if env_root:
        return Path(env_root).expanduser().resolve()

    return (find_workspace_root() / 'Dataset' / 'splits' / 'segmentation').resolve()


def resolve_artifact_root(artifact_root='auto'):
    if artifact_root and artifact_root != 'auto':
        return Path(artifact_root).expanduser().resolve()

    env_root = os.environ.get('AD_GBC_ARTIFACT_ROOT')
    if env_root:
        return Path(env_root).expanduser().resolve()

    return (find_workspace_root() / 'Artifacts' / 'AD-GBC').resolve()


def read_split_file(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_val_split_ids(split_root, dataset, seed, available_ids):
    val_path = Path(split_root) / dataset / f'seed_{seed}_val.txt'
    if not val_path.exists():
        return None

    val_ids = read_split_file(val_path)
    available = set(available_ids)
    missing = [img_id for img_id in val_ids if img_id not in available]
    if missing:
        raise FileNotFoundError(
            f'split file {val_path} references missing image ids, e.g. {missing[:5]}'
        )

    print('Using fixed validation split: %s' % val_path)
    return val_ids


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    args = parse_args()
    device = select_device(args.device)
    print('Using device: %s' % device)

    artifact_root = resolve_artifact_root(args.artifact_root)
    models_root = artifact_root / 'models'
    outputs_root = artifact_root / 'outputs'
    model_dir = models_root / args.name

    with open(model_dir / 'config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    configured_data_root = args.data_root
    if configured_data_root == 'auto':
        configured_data_root = config.get('data_root', 'auto')
    configured_split_root = args.split_root
    if configured_split_root == 'auto':
        configured_split_root = config.get('split_root', 'auto')
    data_root = resolve_data_root(configured_data_root)
    split_root = resolve_split_root(configured_split_root)
    print('Using data root: %s' % data_root)
    print('Using split root: %s' % split_root)
    print('Using artifact root: %s' % artifact_root)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    gbc_kwargs = {
        'gbc_num_balls': config.get('gbc_num_balls', 32),
        'gbc_proj_dim': None if config.get('gbc_proj_dim', 0) == 0 else config.get('gbc_proj_dim'),
        'use_diag_cov': config.get('use_diag_cov', True),
        'tau': config.get('tau', 1.0),
        'gbc_mode': config.get('gbc_mode', 'static'),
    }
    model = archs_GBC.__dict__[config['arch']](num_classes=config['num_classes'],
                                           input_channels=config['input_channels'],
                                           deep_supervision=config['deep_supervision'],
                                           **gbc_kwargs)

    model = model.to(device)

    if config['dataset'] == 'ISIC17':
        img_ext = '.jpg'
    else:
        img_ext = '.png'

    if config['dataset'] == 'busi':
        mask_ext = '_mask.png'
    elif config['dataset'] == 'glas':
        mask_ext = '.png'
    elif config['dataset'] == 'cvc':
        mask_ext = '.png'
    elif config['dataset'] == 'ISIC17':
        mask_ext = '_segmentation.png'
    else:
        mask_ext = config.get('mask_ext', '.png')

    # Data loading code
    dataset_root = data_root / config['dataset']
    img_ids = glob(str(dataset_root / 'images' / ('*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    val_img_ids = load_val_split_ids(split_root, config['dataset'], config['dataseed'], img_ids)
    if val_img_ids is None:
        print('No fixed validation split found; falling back to train_test_split with dataseed=%s' % config['dataseed'])
        _, val_img_ids = train_test_split(
            img_ids, test_size=0.2, random_state=config['dataseed']
        )

    model.load_state_dict(torch.load(model_dir / 'model.pth', map_location=device))
    model.eval()

    val_transform = SimpleSegTransform(config['input_h'], config['input_w'], training=False)

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=str(dataset_root / 'images'),
        mask_dir=str(dataset_root / 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()
    recall_avg_meter = AverageMeter()
    specificity_avg_meter = AverageMeter()
    precision_avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(outputs_root / args.name / str(c), exist_ok=True)

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)
            # compute output
            output = model(input)

            # iou, dice = iou_score(output, target)
            iou, dice, hd, hd95, recall, specificity, precision = indicators(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            hd_avg_meter.update(hd, input.size(0))
            hd95_avg_meter.update(hd95, input.size(0))
            recall_avg_meter.update(recall, input.size(0))
            specificity_avg_meter.update(specificity, input.size(0))
            precision_avg_meter.update(precision, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    Image.fromarray((output[i, c] * 255).astype('uint8')).save(
                        outputs_root / args.name / str(c) / (meta['img_id'][i] + '.png')
                    )

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    print('Hd: %.4f' % hd_avg_meter.avg)
    print('Hd95: %.4f' % hd95_avg_meter.avg)
    print('Recall: %.4f' % recall_avg_meter.avg)
    print('Specificity: %.4f' % specificity_avg_meter.avg)
    print('Precision: %.4f' % precision_avg_meter.avg)

    empty_device_cache(device)


if __name__ == '__main__':
    main()
