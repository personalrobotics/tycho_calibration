import sys
import numpy as np
import torch
from hebi_env.residual_estimator import quat_shift_to_transformation, ResEstimator
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple
import pandas as pd
from datetime import datetime
import os
from math import pi
import argparse
from torch.utils.tensorboard import SummaryWriter

HEBI_MAX_BACKLASH = 0.25 * pi / 180. * 1.1  # in radians (10% fudge factor)

# See the explanation in calibration.py
optimized_FK = np.array([ # 2021 June 25
    0.00000197751938338511, 0.00000000000000000000, 0.10146636705958071711, -0.01995108501982497534,
    1.57078973509802510833, 0.00000000000000000000, 0.08232539360652703364, -0.00860234628959506871,
    3.14159663371207376059, 0.32629066314809224147, 0.04552785300369301819, 0.00673997924040529734,
    3.13627271999935741675, 0.32731971536747545004, 0.07100628183517872227, -0.00561447135824789598,
    1.57079775505823837634, 0.00000000000000000000, 0.11402510192401031641, -0.00000001036213024073,
    1.57079909083616686694, 0.00000000000000000000, 0.11373140497245554092, -0.00000118215296507181,
    -0.70699999999999996181, 0.00000000000000000000, 0.00000000000000000000, 0.70699999999999996181,
    0.11942109387735543036, 0.07954729440645284810, 0.02443140499777806188
])

# first 4 are quat, last 3 are translation
cage_to_hebi_quat_shift = np.array([
    # Update on 2021 June 21
    0.00714843, 0.00664545, -0.00000020, 0.99995237, -1.07350576, 0.17235089,
    -0.02163 + 0.0162])  # Should match hebi_calibration measured_R


# x is joint angles, y is chopstick tip position
def load_data(path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    cage_to_hebi = quat_shift_to_transformation(cage_to_hebi_quat_shift)
    csv = pd.read_csv(path)
    x_data = [np.fromstring(r[1:-1], dtype=np.float32, sep=' ')[0:6] for r in
              csv['joint_position'].to_list()]  # keep only 6 joints
    ee_data = [np.fromstring(r[1:-1], dtype=np.float32, sep=' ') for r in
               csv['ball_loc'].to_list()]  # [1:-1] to exclude '['']'
    # Remember to transform from cage frame to hebi frame
    y_data = [(cage_to_hebi @ np.hstack((pos, [1])).reshape((4, 1))).astype(np.float32)[:-1] for pos in ee_data]
    return x_data, y_data


def create_dataloader(device: torch.device, data_x: List[np.ndarray], data_y: List[np.ndarray],
                      batch_size) -> DataLoader:
    x = torch.tensor(data_x).to(device)
    y = torch.tensor(data_y).to(device)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(epochs: int, model: ResEstimator, data: DataLoader, loss_fn, optimizer: torch.optim.Optimizer,
          log_interval=1, writer=None, checkpoint_interval=-1, checkpoint_dir=None,
          test_interval=-1, test_data: DataLoader=None):
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
        if checkpoint_interval != -1 and checkpoint_dir is not None and epoch % checkpoint_interval == 0:
            state = {"optimizer_state_dict": optimizer.state_dict()}
            model.save(os.path.join(checkpoint_dir, f"checkpoint_epoch-{epoch}_loss-{total_loss / n_batches:.6f}.tar"),
                       state, backwards_compatible=True)
        if test_interval != -1 and test_data is not None and epoch % test_interval == 0:
            test(epoch, model, test_data, writer)
        if epoch % log_interval == 0:
            mean_loss = total_loss / n_batches
            epoch_str = ("%% %dd" % epoch_digits) % epoch  # format the format string, then insert epoch
            print(f"Progress: {epoch_str}/{epochs} - Loss: {mean_loss:.6f}")
    model.train(False)


def test(epoch: int, model: ResEstimator, data: DataLoader, writer):
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
    writer.add_scalar("Loss/test", avg_loss, epoch)
    print(f"Test performance: {avg_loss:.6f}")
    model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data", type=str, help="Path to the train data file")
    parser.add_argument("--test_data", type=str, default=None, help="Path to the test data file")
    parser.add_argument("--test_interval", type=int, default=-1, help="Interval between evaluating test dataset")
    parser.add_argument("-n", "--n_epoch", type=int, default=100,
                        help="Number of epochs to train the model (default 100)")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for training (default 32)")
    parser.add_argument("--log_interval", type=int, default=1,
                        help="Print training information every set number of epochs")
    parser.add_argument("-o", "--output", type=str, default=None, help="Save location of trained model")
    parser.add_argument("-cp", "--checkpoint_interval", type=int, default=-1,
                        help="Number of epochs to wait between saving checkpoints")
    parser.add_argument("-m", "--model_path", type=str, default=None,
                        help="Path to model checkpoint to resume training")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4,
                        help="Learning rate to use for optimization")
    return parser.parse_args()


def validate_args(args):
    if args.test_interval != -1 and args.test_data is None:
        print("No test data was passed, but a test interval was specified!", file=sys.stderr)
        exit(1)


def main():
    args = parse_args()
    validate_args(args)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_name}")
    device = torch.device(device_name)
    fk_params = optimized_FK[:-7].reshape((6, 4))
    last_joint = optimized_FK[-7:]
    net = ResEstimator.create(device, HEBI_MAX_BACKLASH, fk_params, last_joint)
    loss = lambda a, b: torch.norm(a - b, dim=1).mean() # mean of norms across batch
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    x_train, y_train = load_data(args.train_data)
    x_test, y_test = (None, None) if args.test_data is None else load_data(args.test_data)
    dataloader = create_dataloader(device, x_train, y_train, args.batch_size)
    test_dataloader = None if x_test is None else create_dataloader(device, x_test, y_test, args.batch_size)
    writer = SummaryWriter()

    if args.model_path is not None:
        print("Loading checkpoint from file...")
        state = net.load(args.model_path)
        optimizer.load_state_dict(state["optimizer_state_dict"])

    folder = str(datetime.now().strftime("results/res-estim-%m%d-%H-%M"))
    # Handle the case where the folder already exists
    if os.path.isdir(folder):
        old_folder_name = folder
        n = 1
        while os.path.isdir(folder):
            folder = f"{old_folder_name}_{n}"
            n += 1
    os.mkdir(folder)
    checkpoint_interval = args.checkpoint_interval
    checkpoint_dir = None if checkpoint_interval == -1 else os.path.join(folder, "checkpoints")
    if checkpoint_dir is not None:
        os.mkdir(checkpoint_dir)

    print("Starting training...")
    train(args.n_epoch, net, dataloader, loss, optimizer, log_interval=args.log_interval, writer=writer,
          checkpoint_interval=checkpoint_interval, checkpoint_dir=checkpoint_dir,
          test_interval=args.test_interval, test_data=test_dataloader)
    print("Finished training!")

    net.save(args.output if args.output is not None else os.path.join(folder, "model.pt"), backwards_compatible=True)


if __name__ == "__main__":
    main()
