import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", help="Img path or video path")
    parser.add_argument("--model_type", help="Model type", default='ENet')
    parser.add_argument("--model", help="Model path", default='./log/best_model.pth')
    parser.add_argument("--width", required=False, type=int, help="Resize width", default=512)
    parser.add_argument("--height", required=False, type=int, help="Resize height", default=256)
    parser.add_argument("--save", help="Directory to save output", default="./test_output")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze encoder backbone")
    parser.add_argument("--use_kalman", action="store_true", help="Use Kalman filter for lane tracking")
    parser.add_argument("--kalman_process_noise", type=float, default=0.03, help="Kalman process noise")
    parser.add_argument("--kalman_measurement_noise", type=float, default=0.3, help="Kalman measurement noise")
    parser.add_argument("--max_lanes", type=int, default=4, help="Maximum number of lanes to track")
    parser.add_argument("--sequence", action="store_true", help="Process multiple images as sequence (for Kalman)")
    parser.add_argument("--img_pattern", help="Image pattern for sequence (e.g., './data/test_kalman/*.jpg')")
    
    # LaneNetPlus specific arguments
    parser.add_argument("--use_lanenet_plus", action='store_true',
                       help="Use LaneNetPlus instead of LaneNet")
    parser.add_argument("--use_attention", action='store_true',
                       help="Use self-attention blocks (must match training)")
    parser.add_argument("--use_multitask", action='store_true',
                       help="Enable multi-task output (must match training)")
    parser.add_argument("--use_rectification", action='store_true',
                       help="Apply homography rectification preprocessing")
    
    return parser.parse_args()
