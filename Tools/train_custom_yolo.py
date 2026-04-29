import argparse
import tempfile
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train a custom YOLO model.")
    parser.add_argument("--model", type=str, default="YOLO/yolo11n.pt", help="Base weights to fine-tune from.")
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/my_objects/data.yaml",
        help="Path to the Ultralytics dataset YAML.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--project", type=str, default="runs/detect", help="Output project directory.")
    parser.add_argument("--name", type=str, default="custom_train", help="Run name inside the project directory.")
    parser.add_argument("--device", type=str, default=None, help="Training device, for example cpu, 0, or 0,1.")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def prepare_dataset_yaml(data_path: Path) -> Path:
    if data_path.suffix.lower() not in {".yaml", ".yml"}:
        return data_path

    yaml_text = data_path.read_text(encoding="utf-8")
    dataset_root = data_path.parent.as_posix()
    yaml_lines = yaml_text.splitlines()
    updated_lines = []
    path_replaced = False

    for line in yaml_lines:
        if line.strip().startswith("path:"):
            updated_lines.append(f"path: {dataset_root}")
            path_replaced = True
        else:
            updated_lines.append(line)

    if not path_replaced:
        updated_lines.insert(0, f"path: {dataset_root}")

    temp_file = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    with temp_file:
        temp_file.write("\n".join(updated_lines) + "\n")
    return Path(temp_file.name)


def clear_dataset_caches(data_path: Path) -> None:
    dataset_root = data_path.parent
    labels_dir = dataset_root / "labels"
    for split_name in ("train", "val", "test"):
        cache_path = labels_dir / f"{split_name}.cache"
        if cache_path.exists():
            cache_path.unlink()


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
    print(f"Using dataset: {data_path}")

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
        f"python autonomous_drone_main.py --model-path {weights_dir / 'best.pt'}"
    )


if __name__ == "__main__":
    main()
