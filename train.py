import argparse
from configs.stage1_config import Stage1Config
from configs.stage2_config import Stage2Config
from trainers.stage1_trainer import run_stage1_training
from trainers.stage2_trainer import run_stage2_training

def main():
    parser = argparse.ArgumentParser(description="DAPFusion Modular Runner")
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2], help='Pipeline Stage (1 or 2)')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    
    args = parser.parse_args()

    if args.stage == 1:
        config = Stage1Config()
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.epochs is not None:
            config.epochs = args.epochs
        run_stage1_training(config)
        
    elif args.stage == 2:
        config = Stage2Config()
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.epochs is not None:
            config.epochs = args.epochs
        run_stage2_training(config)

if __name__ == '__main__':
    main()