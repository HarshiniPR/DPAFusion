import argparse
from configs.stage1_config import Stage1Config
from configs.stage2_config import Stage2Config
from configs.stage4_config import Stage4Config

from trainers.stage1_trainer import run_stage1_training
from trainers.stage2_trainer import run_stage2_training
from trainers.stage4_trainer import run_stage4_training

def main():
    parser = argparse.ArgumentParser(description="DPAFusion Unified Training Pipeline")
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 4], help="Stage to train: 1, 2, or 4")
    parser.add_argument('--epochs', type=int, default=None, help="Override epochs")
    parser.add_argument('--batch_size', type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    if args.stage == 1:
        config = Stage1Config()
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        run_stage1_training(config)

    elif args.stage == 2:
        config = Stage2Config()
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        run_stage2_training(config)

    elif args.stage == 4:
        config = Stage4Config()
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        run_stage4_training(config)

if __name__ == '__main__':
    main()