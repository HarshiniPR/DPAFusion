import argparse
from configs.stage1_config import Stage1Config
from trainers.stage1_trainer import run_stage1_training

def main():
    parser = argparse.ArgumentParser(description="DAPFusion Training Entrypoint")
    parser.add_argument('--stage', type=int, default=1, help='Pipeline Stage to execute (default: 1)')
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
    else:
        raise NotImplementedError(f"Stage {args.stage} is not implemented yet.")

if __name__ == '__main__':
    main()