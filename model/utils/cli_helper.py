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
    
    # LaneNetPlus specific arguments
    parser.add_argument("--use_lanenet_plus", action='store_true', 
                       help="Use LaneNetPlus instead of LaneNet")
    parser.add_argument("--use_attention", action='store_true',
                       help="Use self-attention blocks in encoder")
    parser.add_argument("--use_multitask", action='store_true',
                       help="Enable multi-task learning (lane + drivable area)")
    parser.add_argument("--use_rectification", action='store_true',
                       help="Use homography rectification preprocessing")
    parser.add_argument("--lambda_drivable", type=float, default=0.5,
                       help="Weight for drivable area loss in multi-task learning")
    parser.add_argument("--drivable_dir", type=str, default=None,
                       help="Directory containing drivable area masks")
    parser.add_argument("--save_visualizations", action='store_true',
                       help="Save visualization images during training")
    
    return parser.parse_args()
