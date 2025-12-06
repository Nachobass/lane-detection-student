import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset path")
    parser.add_argument("--model_type", help="Model type", default='ENet')
    parser.add_argument("--loss_type", help="Loss type", default='FocalLoss')
    parser.add_argument("--save", required=False, help="Directory to save model", default="./log")
    parser.add_argument("--epochs", required=False, type=int, help="Training epochs", default=25)
    parser.add_argument("--width", required=False, type=int, help="Resize width", default=512)
    parser.add_argument("--height", required=False, type=int, help="Resize height", default=256)
    parser.add_argument("--bs", required=False, type=int, help="Batch size", default=4)
    parser.add_argument("--val", required=False, type=bool, help="Use validation", default=False)
    parser.add_argument("--lr", required=False, type=float, help="Learning rate", default=0.0001)
    parser.add_argument("--pretrained", required=False, default=None, help="pretrained model path")
    parser.add_argument("--image", default="./output", help="output image folder")
    parser.add_argument("--net", help="backbone network")
    parser.add_argument("--json", help="post processing json")
    
    # Temporal training arguments
    parser.add_argument("--use_temporal", action='store_true', default=False, 
                       help="Enable temporal training with ConvLSTM (default: False)")
    parser.add_argument("--sequence_length", type=int, default=3,
                       help="Number of frames in temporal sequence (default: 3)")
    parser.add_argument("--freeze_encoder", action='store_true', default=True,
                       help="Freeze encoder during Phase 1 training (default: True)")
    parser.add_argument("--num_epochs_phase1", type=int, default=5,
                       help="Number of epochs for Phase 1 (frozen encoder) (default: 5)")
    parser.add_argument("--num_epochs_phase2", type=int, default=20,
                       help="Number of epochs for Phase 2 (unfrozen encoder) (default: 20)")
    
    # Data augmentation arguments
    parser.add_argument("--use_augmentation", action='store_true', default=False,
                       help="Enable data augmentations using Albumentations (default: False, but True for temporal training)")
    parser.add_argument("--strong_augmentation", action='store_true', default=False,
                       help="Use stronger/more aggressive augmentations (default: False)")
    
    return parser.parse_args()
