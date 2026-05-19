import os
import warnings

import numpy as np
from prettytable import PrettyTable

from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info


def get_weight_size(path):
    stats = os.stat(path)
    return f"{stats.st_size / 1024 / 1024:.1f}"


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    model_path = "runs/train/exp/weights/best.pt"
    data_yaml = "dataset/data.yaml"

    model = RTDETR(model_path)
    results = model.val(
        data=data_yaml,
        split="val",
        imgsz=640,
        batch=4,
        project="runs/val",
        name="exp",
    )

    if model.task != "detect":
        raise RuntimeError(f"Unsupported task for this script: {model.task}")

    class_count = results.box.p.size
    class_names = list(results.names.values())

    preprocess_ms = results.speed["preprocess"]
    inference_ms = results.speed["inference"]
    postprocess_ms = results.speed["postprocess"]
    total_ms = preprocess_ms + inference_ms + postprocess_ms

    _, n_params, _, flops = model_info(model.model)

    info_table = PrettyTable()
    info_table.title = "Model Info"
    info_table.field_names = [
        "GFLOPs",
        "Parameters",
        "Preprocess (s/img)",
        "Inference (s/img)",
        "Postprocess (s/img)",
        "FPS (total)",
        "FPS (inference)",
        "Model File Size",
    ]
    info_table.add_row(
        [
            f"{flops:.1f}",
            f"{n_params:,}",
            f"{preprocess_ms / 1000:.6f}s",
            f"{inference_ms / 1000:.6f}s",
            f"{postprocess_ms / 1000:.6f}s",
            f"{1000 / total_ms:.2f}",
            f"{1000 / inference_ms:.2f}",
            f"{get_weight_size(model_path)}MB",
        ]
    )
    print(info_table)

    metrics_table = PrettyTable()
    metrics_table.title = "Detection Metrics"
    metrics_table.field_names = ["Class", "Precision", "Recall", "F1", "mAP50", "mAP75", "mAP50-95"]
    for i in range(class_count):
        metrics_table.add_row(
            [
                class_names[i],
                f"{results.box.p[i]:.4f}",
                f"{results.box.r[i]:.4f}",
                f"{results.box.f1[i]:.4f}",
                f"{results.box.ap50[i]:.4f}",
                f"{results.box.all_ap[i, 5]:.4f}",
                f"{results.box.ap[i]:.4f}",
            ]
        )

    metrics_table.add_row(
        [
            "all",
            f"{results.results_dict['metrics/precision(B)']:.4f}",
            f"{results.results_dict['metrics/recall(B)']:.4f}",
            f"{np.mean(results.box.f1[:class_count]):.4f}",
            f"{results.results_dict['metrics/mAP50(B)']:.4f}",
            f"{np.mean(results.box.all_ap[:class_count, 5]):.4f}",
            f"{results.results_dict['metrics/mAP50-95(B)']:.4f}",
        ]
    )
    print(metrics_table)

    out_file = results.save_dir / "paper_data.txt"
    with open(out_file, "w+", errors="ignore", encoding="utf-8") as f:
        f.write(str(info_table))
        f.write("\n")
        f.write(str(metrics_table))

    print(f"Saved results to: {out_file}")
