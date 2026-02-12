import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cldm.test_callback import EpochTestCallback


def parse_args():
    parser = argparse.ArgumentParser(description='Step2 Training')
    # ==================== Configs ====================
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the pretrained checkpoint (.ckpt)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--logger_freq', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--sd_locked', action='store_true', default=True)
    parser.add_argument('--only_mid_control', action='store_true', default=False)

    # ==================== Dataset Configs ====================
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Root directory of the dataset (e.g. /root/autodl-tmp/dreamnav/)')
    parser.add_argument('--dataset_type', type=str, default='try_train',
                        help='Dataset subdirectory name (e.g. try_train)')

    # ==================== Test Callback Configs ====================
    # Path to step1 prediction JSON file
    parser.add_argument('--step1_json_path', type=str, required=True,
                        help='Path to step1 prediction JSON file')
    # Correction offsets: will try prediction ± these values and 0
    parser.add_argument('--heading_offset', type=float, default=10.0,
                        help='heading will try ±this value')
    parser.add_argument('--range_offset', type=float, default=1.5,
                        help='range will try ±this value')
    # DDIM sampling steps for test image generation
    parser.add_argument('--test_ddim_steps', type=int, default=50)
    # Maximum test samples per epoch (None for all, set smaller for faster testing)
    parser.add_argument('--max_test_samples', type=int, default=100,
                        help='Set to -1 to test all samples')
    # Number of test samples to process simultaneously (each has 9 offset combos)
    parser.add_argument('--test_batch_size', type=int, default=3,
                        help='Number of test samples to run in parallel (total batch = this * 9)')
    # Directory to save test results
    parser.add_argument('--test_save_dir', type=str, required=True,
                        help='Directory to save test results')
    # Output root directory for trainer
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root directory for training outputs (checkpoints, logs)')

    return parser.parse_args()


def main():
    args = parse_args()

    max_test_samples = None if args.max_test_samples < 0 else args.max_test_samples

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./cldm_v15_numeric.yaml').cpu()
    model.load_state_dict(load_state_dict(args.checkpoint_path, location='cpu'), strict=False)
    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control


    # Misc
    dataset = MyDataset(root_dir=args.root_dir, dataset_type=args.dataset_type)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)

    # 每12200保存+测试一次
    # Callbacks
    logger = ImageLogger(batch_frequency=args.logger_freq)
    test_callback = EpochTestCallback(
        data_root=args.root_dir,
        step1_json_path=args.step1_json_path,
        heading_offset=args.heading_offset,
        range_offset=args.range_offset,
        ddim_steps=args.test_ddim_steps,
        max_test_samples=max_test_samples,
        save_dir=args.test_save_dir,
        test_batch_size=args.test_batch_size
    )

    trainer = pl.Trainer(
        gpus=1, 
        precision=32, 
        max_epochs=args.max_epochs,
        callbacks=[logger, test_callback], 
        default_root_dir=args.output_dir,
        #limit_train_batches=1,  # 取消注释可用于快速测试
    )


    # Train!
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    main()
