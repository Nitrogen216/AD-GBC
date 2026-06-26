import argparse
import os
from collections import OrderedDict
from glob import glob
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import archs_GBC
import losses
from dataset import Dataset
from metrics import iou_score
from simple_transforms import SimpleSegTransform
from utils import AverageMeter, str2bool
import time
from tensorboardX import SummaryWriter

ARCH_NAMES = archs_GBC.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N', help='mini-batch size(default: 8)')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='auto',
                        help='device to use: auto, cuda, mps, or cpu')

    parser.add_argument('--dataseed', default=41, type=int,
                        help='') # default 2981

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH',
                        default='GBC_Rolling_Unet_S')  ### GBC_Rolling_Unet_S, GBC_Rolling_Unet_M, GBC_Rolling_Unet_L
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--input_w', default=256, type=int, help='image width(default: 256)')
    parser.add_argument('--input_h', default=256, type=int, help='image height(default: 256)')

    # loss
    parser.add_argument('--loss', default='BCEDiceWithGeometryLoss', choices=LOSS_NAMES,
                        help='loss: ' + ' | '.join(LOSS_NAMES) + ' (default: BCEDiceWithGeometryLoss)')
    parser.add_argument('--div_weight', type=float, default=0.01, help='Weight of diversity loss')
    parser.add_argument('--scale_weight', type=float, default=0.01, help='Weight of scale consistency loss')

    # data
    parser.add_argument('--dataset', default='isic', help='dataset name')  ### isic, busi, chasedb1, glas
    parser.add_argument('--data_root', default='auto',
                        help='dataset root. auto resolves to ../../Dataset/segmentation when available')
    parser.add_argument('--split_root', default='auto',
                        help='split root. auto resolves to ../../Dataset/splits/segmentation when available')
    parser.add_argument('--artifact_root', default='auto',
                        help='artifact root. auto resolves to ../../Artifacts/AD-GBC when available')
    parser.add_argument('--img_ext', default='.png', help='image file extension')
    parser.add_argument('--mask_ext', default='.png', help='masks file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help='loss: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR',
                        help='initial learning rate(default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay(default: 1e-4)')
    parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')

    # GBC
    parser.add_argument('--gbc_lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate for GBC module (centers and logit_scale)')
    parser.add_argument('--gbc_weight_decay', default=1e-4, type=float,
                        help='weight decay for GBC centers')
    parser.add_argument('--gbc_num_balls', default=32, type=int, help='number of granular balls (shared)')
    parser.add_argument('--gbc_proj_dim', default=0, type=int,
                        help='projection dim for GBC; 0 means use kan_input_dim (auto)')
    parser.add_argument('--use_diag_cov', default=True, type=str2bool,
                        help='Use diagonal covariance (K×D scales). Fixed Prompt 1: needs str2bool '
                             'so --use_diag_cov False actually selects isotropic.')
    parser.add_argument('--tau', default=1.0, type=float,
                        help='softmax temperature')
    parser.add_argument('--gbc_mode', default='static',
                        choices=['static', 'paper_sum', 'mean'],
                        help='GBC region update: static (legacy) | paper_sum (diagnostic) | mean')
    parser.add_argument('--wdiv_mode', default='legacy',
                        choices=['legacy', 'paper', 'rank_aware'],
                        help='Wasserstein diversity loss variant')
    parser.add_argument('--train_seed', default=1029, type=int,
                        help='seed for model init / data order (separate from --dataseed split seed)')
    parser.add_argument('--deterministic', default=True, type=str2bool,
                        help='cudnn deterministic + benchmark off for reproducibility')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N', help='early stopping (default: -1)')

    config = parser.parse_args()

    return config


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


def load_split_ids(split_root, dataset, seed, available_ids):
    split_dir = Path(split_root) / dataset
    train_path = split_dir / f'seed_{seed}_train.txt'
    val_path = split_dir / f'seed_{seed}_val.txt'
    if not train_path.exists() or not val_path.exists():
        return None

    train_ids = read_split_file(train_path)
    val_ids = read_split_file(val_path)
    available = set(available_ids)
    missing = [img_id for img_id in train_ids + val_ids if img_id not in available]
    if missing:
        raise FileNotFoundError(
            f'split files under {split_dir} reference missing image ids, e.g. {missing[:5]}'
        )

    print('Using fixed split files:')
    print('  train: %s' % train_path)
    print('  val: %s' % val_path)
    return train_ids, val_ids


def train(config, train_loader, model, criterion, optimizer, device):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.to(device)
        target = target.to(device)

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                if config['loss'] == 'BCEDiceWithGeometryLoss':
                    loss += criterion(output, target, model)
                else:
                    output, _ = output
                    loss += criterion(output, target)
            loss /= len(outputs)

            iou, dice = iou_score(outputs[-1], target)
            
        else:
            output = model(input)
            if config['loss'] == 'BCEDiceWithGeometryLoss':
                loss = criterion(output, target, model)
            else:
                output, _ = output
                loss = criterion(output, target)
            iou, dice = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion, device):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    if config['loss'] == 'BCEDiceWithGeometryLoss':
                        loss += criterion(output, target, model)
                    else:
                        loss += criterion(output, target)

                loss /= len(outputs)
                iou, dice = iou_score(outputs[-1], target)
            else:
                output = model(input)
                if config['loss'] == 'BCEDiceWithGeometryLoss':
                    loss = criterion(output, target, model)
                else:
                    loss = criterion(output, target)
                iou, dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def seed_torch(seed=1029, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def main():
    seed_torch(config.get('train_seed', 1029), deterministic=config.get('deterministic', True))
    config = vars(parse_args())
    device = select_device(config['device'])
    data_root = resolve_data_root(config['data_root'])
    split_root = resolve_split_root(config['split_root'])
    artifact_root = resolve_artifact_root(config['artifact_root'])
    models_root = artifact_root / 'models'
    runs_root = artifact_root / 'runs'
    config['data_root'] = str(data_root)
    config['split_root'] = str(split_root)
    config['artifact_root'] = str(artifact_root)
    print('Using device: %s' % device)
    print('Using data root: %s' % data_root)
    print('Using split root: %s' % split_root)
    print('Using artifact root: %s' % artifact_root)

    current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

    model_dir = models_root / config['name']
    model_dir.mkdir(parents=True, exist_ok=True)
    my_writer = SummaryWriter(log_dir=str(runs_root / config['name']))

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(model_dir / 'config.yml', 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    elif config['loss'] == 'BCEDiceWithGeometryLoss':
        criterion = losses.__dict__[config['loss']](div_weight=config.get('div_weight'),scale_weight=config.get('scale_weight'),wdiv_mode=config.get('wdiv_mode','legacy')).to(device)
    else:
        criterion = losses.__dict__[config['loss']]().to(device)

    # Prompt 1 fix: do not unconditionally re-enable benchmark (it overrode the
    # deterministic setting from seed_torch). Respect --deterministic.
    cudnn.benchmark = not config.get('deterministic', True)

    gbc_kwargs = {
        'gbc_num_balls': config.get('gbc_num_balls'),
        'gbc_proj_dim': None if config.get('gbc_proj_dim') == 0 else config.get('gbc_proj_dim'),
        'use_diag_cov': config.get('use_diag_cov'),
        'tau': config.get('tau'),
        'gbc_mode': config.get('gbc_mode', 'static'),
    }

    # create model
    model = archs_GBC.__dict__[config['arch']](num_classes=config['num_classes'],
                                           input_channels=config['input_channels'],
                                           deep_supervision=config['deep_supervision'],
                                           **gbc_kwargs)

    model = model.to(device)

    #params = filter(lambda p: p.requires_grad, model.parameters())
    # === parameter groups: separate kan params and gbc centers if desired ===
    params = []
    gbc_centers_params = []
    others = []

    # Prompt 1 fix: geometry group = ONLY the granular-ball geometry parameters
    # (centers / log_sigma / log_radius). The previous rule put every parameter
    # whose name contained 'gbc' — including the GBC refine conv/BN and the
    # projection conv/BN — into the high-LR geometry group.
    geometry_suffixes = ('centers', 'log_sigma', 'log_radius')
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.split('.')[-1] in geometry_suffixes:
            gbc_centers_params.append(param)
        else:
            others.append(param)

    if len(others) > 0:
        params.append({'params': others, 'lr': config['lr'], 'weight_decay': config['weight_decay']})
    if len(gbc_centers_params) > 0:
        # give GBC centers same lr as kan (you can change)
        params.append({'params': gbc_centers_params, 'lr': config['gbc_lr'], 'weight_decay': config['gbc_weight_decay']})

    # fallback if nothing matched (shouldn't happen)
    if len(params) == 0:
        params = [{'params': model.parameters(), 'lr': config['lr'], 'weight_decay': config['weight_decay']}]

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    if config['dataset'] == 'busi':
        config['mask_ext'] = '_mask.png'
    if config['dataset'] == 'ISIC17':
        config['img_ext'] = '.jpg'
        config['mask_ext'] = '_segmentation.png'
        
    dataset_root = data_root / config['dataset']
    img_ids = glob(str(dataset_root / 'images' / ('*' + config['img_ext'])))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    split_ids = load_split_ids(split_root, config['dataset'], config['dataseed'], img_ids)
    if split_ids is not None:
        train_img_ids, val_img_ids = split_ids
    else:
        print('No fixed split found; falling back to train_test_split with dataseed=%s' % config['dataseed'])
        train_img_ids, val_img_ids = train_test_split(
            img_ids, test_size=0.2, random_state=config['dataseed']
        )

    train_transform = SimpleSegTransform(config['input_h'], config['input_w'], training=True)
    val_transform = SimpleSegTransform(config['input_h'], config['input_w'], training=False)

    
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=str(dataset_root / 'images'),
        mask_dir=str(dataset_root / 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=str(dataset_root / 'images'),
        mask_dir=str(dataset_root / 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True,
        pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer, device)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion, device)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv(model_dir / 'log.csv', index=False)

        my_writer.add_scalar('loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('val_loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val_iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val_dice', val_log['dice'], global_step=epoch)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), model_dir / 'model.pth')
            best_iou = val_log['iou']
            best_dice = val_log['dice']
            print("=> saved best model")
            print('IoU: %.4f' % best_iou)
            print('Dice: %.4f' % best_dice)
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        empty_device_cache(device)


if __name__ == '__main__':
    main()
