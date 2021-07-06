import sys
import os
import numpy as np
from typing import List, Tuple
import pandas as pd
from datetime import datetime
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

from tycho_env.residual_estimator import quat_shift_to_transformation, ResEstimator

from calibration import optimized_FK, measured_R

HEBI_MAX_BACKLASH = 0.25 * np.pi / 180. * 1.1  # in radians (10% fudge factor)

# x is joint angles, y is chopstick tip position
def load_csv_data(path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    optitrack2robot = quat_shift_to_transformation(measured_R)
    csv = pd.read_csv(path)
    x_data = [np.fromstring(r[1:-1], dtype=np.float32, sep=' ')[0:6] for r in
              csv['joint_position'].to_list()]  # keep only 6 joints
    ee_data = [np.fromstring(r[1:-1], dtype=np.float32, sep=' ') for r in
               csv['ball_loc'].to_list()]  # [1:-1] to exclude '['']'
    # Remember to transform from optitrack frame to robot frame
    y_data = [(optitrack2robot @
               np.hstack((pos, [1])).reshape((4, 1))).astype(np.float32)[:-1]
               for pos in ee_data]
    return x_data, y_data

def load_dataset(datapath, split_percentage, device, batch_size, seed):
    data_x, data_y = load_csv_data(datapath)
    x = torch.tensor(data_x).to(device)
    y = torch.tensor(data_y).to(device)
    dataset = TensorDataset(x, y)
    dataset_size = len(dataset)
    val_size = int(split_percentage * dataset_size // 100)
    train_size = dataset_size - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    print("Train set contains {} datapoints; Val has {}".format(
        train_size, val_size))
    return train_loader, val_loader

def train(epochs: int, model: ResEstimator, data: DataLoader,
          loss_fn, optimizer: torch.optim.Optimizer,
          log_interval=1, writer=None,
          checkpoint_interval=-1, checkpoint_dir=None,
          val_interval=-1, val_dataloader: DataLoader=None):
    model.train()
    epoch_digits = 1 + np.ceil(np.log10(epochs))
    for epoch in range(1, epochs + 1):
        total_loss = 0
        n_batches = 0
        for batch, (x, y) in enumerate(data):
            y = y.squeeze(-1)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if writer is not None:
            writer.add_scalar("Loss/train", total_loss / n_batches, epoch)
        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            state = {"optimizer_state_dict": optimizer.state_dict()}
            model.save(os.path.join(checkpoint_dir,
                f"ckpt_epoch-{epoch}_loss-{total_loss / n_batches:.6f}.tar"),
                state, backwards_compatible=True)
        if val_interval != -1 and epoch % val_interval == 0:
            evaluate(epoch, model, val_dataloader, writer)
        if epoch % log_interval == 0:
            mean_loss = total_loss / n_batches
            epoch_str = ("%% %dd" % epoch_digits) % epoch
            # format the format string, then insert epoch
            print(f"Progress: {epoch_str}/{epochs} - Loss: {mean_loss:.6f}")
    model.train(False)


def evaluate(epoch: int, model: ResEstimator, data: DataLoader, writer):
    model.eval()
    total_loss = 0
    n_data = 0
    for _,(x,y) in enumerate(data):
        y = y.squeeze(-1)
        y_hat = model(x)
        loss = torch.norm(y_hat-y, dim=1).sum()
        total_loss += loss
        n_data += x.shape[0]
    avg_loss = total_loss / n_data
    writer.add_scalar("Loss/val", avg_loss, epoch)
    print(f"Val performance: {avg_loss:.6f}")
    model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data", type=str, help="Path to the data file")
    parser.add_argument("-s", "--seed", type=int, default=13, help="Seed")
    parser.add_argument("--val_split", type=int, default=20,
                        help="Portion of data to use for validation (0-100)")
    parser.add_argument("--val_interval", type=int, default=-1,
                        help="Eval performance on val dataset every X epochs")
    parser.add_argument("-n", "--n_epoch", type=int, default=100,
                        help="Number of epochs to train (default 100)")
    parser.add_argument("-b", "--batch_size", type=int, default=256,
                        help="Batch size for training (default 256)")
    parser.add_argument("--log_interval", type=int, default=1,
                        help="Print training information every X epochs")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save location of trained model")
    parser.add_argument("-cp", "--checkpoint_interval", type=int, default=-1,
                        help="Save checkpoints every X epochs")
    parser.add_argument("-m", "--model_path", type=str, default=None,
                        help="Path to a model checkpoint to resume training")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    return parser.parse_args()

def main():
    args = parse_args()
    assert args.val_interval == -1 or args.val_split > 0, \
           "To eval the agent performance, need to have val_split > 0!"

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_name}")
    device = torch.device(device_name)

    train_dataloader, val_dataloader = load_dataset(
        args.train_data, args.val_split, device, args.batch_size, args.seed)

    fk_params = optimized_FK[:-7].reshape((6, 4))
    last_joint = optimized_FK[-7:]
    net = ResEstimator.create(device, HEBI_MAX_BACKLASH, fk_params, last_joint)
    loss = lambda a, b: torch.norm(a - b, dim=1).mean() # mean of norms across batch
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    writer = SummaryWriter()

    if args.model_path is not None:
        print("Loading checkpoint from file...")
        state = net.load(args.model_path)
        optimizer.load_state_dict(state["optimizer_state_dict"])

    folder = str(datetime.now().strftime("results/residual-%m%d-%H-%M-%S"))
    os.mkdir(folder)

    if args.checkpoint_interval != -1:
      checkpoint_dir = os.path.join(folder, "checkpoints")
      os.mkdir(checkpoint_dir)
    else:
      checkpoint_dir = None

    print("Starting training...")
    train(args.n_epoch, net, train_dataloader, loss, optimizer,
          log_interval=args.log_interval, writer=writer,
          checkpoint_interval=args.checkpoint_interval,
          checkpoint_dir=checkpoint_dir,
          val_interval=args.val_interval, val_dataloader=val_dataloader)
    print("Finished training!")

    net.save(args.output if args.output is not None else os.path.join(folder, "model.pt"), backwards_compatible=True)


if __name__ == "__main__":
    main()
