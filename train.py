from im2mesh.checkpoints import CheckpointIO
from im2mesh import config, data
import torch
from torch import nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time

import yaml


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


# > The learning rate is initialized to `initial` and then every `interval` epochs, the learning rate
# is multiplied by `factor`
class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor**(epoch // self.interval))


def adjust_learning_rate(lr_schedules, optimizer, epoch):
    """
    It takes a list of learning rate schedules and an optimizer, and adjusts the learning rate of each
    parameter group in the optimizer according to the corresponding learning rate schedule
    
    :param lr_schedules: a list of learning rate schedules, one for each parameter group
    :param optimizer: the optimizer to use
    :param epoch: the current epoch number
    """
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)


def get_lr(optimizer):
    """
    It returns the learning rate of the optimizer
    
    :param optimizer: the optimizer used to train the model
    :return: The learning rate of the optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


# Arguments
parser = argparse.ArgumentParser(description='Train a 4D model.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
batch_size_vis = cfg['training']['batch_size_vis']
batch_size_val = cfg['training']['batch_size_val']
backup_every = cfg['training']['backup_every']
# exit_after = args.exit_after
lr = cfg['training']['learning_rate']

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

## Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

## Copy current config
with open(os.path.join(out_dir, 'cfg.yaml'), 'w') as f:
    yaml.dump(cfg, f)

## Dataset
train_dataset = config.get_dataset('train', cfg)
val_dataset = config.get_dataset('val', cfg)
print(len(train_dataset), len(val_dataset))

## Dataloader
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           num_workers=16,
                                           shuffle=True,
                                           collate_fn=data.collate_remove_none,
                                           worker_init_fn=data.worker_init_fn)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size_val,
                                         num_workers=4,
                                         shuffle=False,
                                         collate_fn=data.collate_remove_none,
                                         worker_init_fn=data.worker_init_fn)

## For visualizations
vis_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size_vis,
                                         shuffle=True,
                                         collate_fn=data.collate_remove_none,
                                         worker_init_fn=data.worker_init_fn)

## Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# if torch.cuda.device_count() > 1:
#     print("Let's use %d GPUs!" % (torch.cuda.device_count()))
#     model = nn.DataParallel(model)

### Get optimizer and trainer
scheduler = StepLearningRateSchedule(
    lr,  # Initial
    5000,  # Interval
    0.5,  # Factor
)
optimizer = optim.Adam(model.parameters(), lr=scheduler.get_learning_rate(0))
# optimizer = optim.Adam(model.parameters(), lr=lr)

trainer = config.get_trainer(model, optimizer, cfg, device=device)

# Load pre-trained model is existing
kwargs = {
    'model': model,
    'optimizer': optimizer,
}
checkpoint_io = CheckpointIO(
    out_dir,
    initialize_from=cfg['model']['initialize_from'],
    initialization_file_name=cfg['model']['initialization_file_name'],
    **kwargs)
try:
    load_dict = checkpoint_io.load('model.pt')
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)
metric_val_best = load_dict.get('loss_val_best',
                                -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf

print('Current best validation metric (%s): %.8f' %
      (model_selection_metric, metric_val_best))

logger = SummaryWriter(os.path.join(out_dir, 'logs'))

## Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

## Print model
nparameters = sum(p.numel() for p in model.parameters())
# print(model)
print('Total number of parameters: %d' % nparameters)

## Training loop
while True:
    epoch_it += 1
    mean_loss = 0

    # if epoch_it > 0:
    #     print("Starting faster training ..........")
    #     # loader.dataset.set_use_cache(use_cache=True)
    #     train_loader.num_workers = 16
    #     val_loader.num_workers = 4

    for batch in train_loader:
        # print(batch)
        it += 1
        loss, loss_recon, loss_corr = trainer.train_step(cfg, batch)
        logger.add_scalar('train/loss_iter', loss, it)
        logger.add_scalar('train/loss_recon', loss_recon, it)
        logger.add_scalar('train/loss_corr', loss_corr, it)
        mean_loss += loss

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            print('[Epoch %02d] it=%03d, loss=%.4f, lr=%.6f' %
                  (epoch_it, it, loss, get_lr(optimizer)))

        # # Visualize output
        # if visualize_every > 0 and (it % visualize_every) == 0:
        #     print('Visualizing')
        #     trainer.visualize(data_vis)

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io.save('model.pt',
                               epoch_it=epoch_it,
                               it=it,
                               loss_val_best=metric_val_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it,
                               epoch_it=epoch_it,
                               it=it,
                               loss_val_best=metric_val_best)
        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
            eval_dict = trainer.evaluate(val_loader)
            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f' %
                  (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt',
                                   epoch_it=epoch_it,
                                   it=it,
                                   loss_val_best=metric_val_best)
        if it > 396000:
            print('Iter limit reached. Exiting.')
            checkpoint_io.save('model.pt',
                               epoch_it=epoch_it,
                               it=it,
                               loss_val_best=metric_val_best)
            exit(3)

    logger.add_scalar('train/loss_epoch', mean_loss, epoch_it)
    logger.add_scalar('train/lr_epoch', get_lr(optimizer), epoch_it)
