import argparse
from pathlib import Path

from ultralytics import YOLO

from Tools.train_custom_yolo import clear_dataset_caches, prepare_dataset_yaml, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO model for multi-color laser point detection.")
    parser.add_argument("--model", type=str, default="YOLO/yolo11n.pt", help="Base weights to fine-tune from.")
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/laser_red_only_v2/data.yaml",
        help="Path to the Ultralytics dataset YAML.",
    )
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--project", type=str, default="runs/detect", help="Output project directory.")
    parser.add_argument("--name", type=str, default="laser_points_train", help="Run name inside the project directory.")
    parser.add_argument("--device", type=str, default=None, help="Training device, for example cpu, 0, or 0,1.")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = resolve_path(args.model)
    data_path = resolve_path(args.data)
    project_path = resolve_path(args.project)

    if not model_path.exists():
        raise FileNotFoundError(f"Base model weights not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    print(f"Loading model: {model_path}")
    print(f"Using laser dataset: {data_path}")

    model = YOLO(str(model_path))
    prepared_data_path = prepare_dataset_yaml(data_path)
    clear_dataset_caches(data_path)

    train_kwargs = {
        "data": str(prepared_data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(project_path),
        "name": args.name,
        "degrees": 0.0,
        "translate": 0.03,
        "scale": 0.2,
        "fliplr": 0.0,
        "mosaic": 0.2,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.02,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "perspective": 0.0,
        "erasing": 0.0,
        "close_mosaic": 10,
    }
    if args.device:
        train_kwargs["device"] = args.device

    try:
        model.train(**train_kwargs)
    finally:
        if prepared_data_path != data_path and prepared_data_path.exists():
            prepared_data_path.unlink()

    save_dir = Path(model.trainer.save_dir)
    weights_dir = save_dir / "weights"
    print("\nTraining finished.")
    print(f"Best weights: {weights_dir / 'best.pt'}")
    print(f"Last weights: {weights_dir / 'last.pt'}")
    print(
        "\nRun the drone app with your custom model using:\n"
        f"python autonomous_drone_main.py --model-path {weights_dir / 'best.pt'} "
        "--conf-threshold 0.12 --min-box-area-ratio 0.00001 --imgsz 960"
    )


if __name__ == "__main__":
    main()
