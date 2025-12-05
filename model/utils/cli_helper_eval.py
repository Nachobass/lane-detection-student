import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset path")
    parser.add_argument("--model_type", help="Model type", default='ENet')
    parser.add_argument("--model", help="Model path", default='./log/best_model.pth')
    parser.add_argument("--width", required=False, type=int, help="Resize width", default=512)
    parser.add_argument("--height", required=False, type=int, help="Resize height", default=256)
    parser.add_argument("--save", help="Directory to save output", default="./test_output")
    # Temporal evaluation arguments
    parser.add_argument("--use_temporal", action='store_true', default=False,
                       help="Enable temporal evaluation with ConvLSTM")
    parser.add_argument("--sequence_length", type=int, default=3,
                       help="Number of frames in sequence")
    return parser.parse_args()
