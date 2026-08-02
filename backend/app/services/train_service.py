import os
import time
import asyncio
import threading
from typing import List, Dict, Any, Optional

class TrainService:
    def __init__(self):
        self.is_running = False
        self.logs: List[str] = []
        self.listeners = set()
        self.latest_metrics: Dict[str, Any] = {
            "epoch": 0,
            "total_epochs": 0,
            "loss": 0.0,
            "map50": 0.0,
            "status": "idle"
        }

    def add_listener(self, queue: asyncio.Queue):
        self.listeners.add(queue)

    def remove_listener(self, queue: asyncio.Queue):
        self.listeners.discard(queue)

    def _broadcast_log(self, log_line: str):
        self.logs.append(log_line)
        if len(self.logs) > 1000:
            self.logs.pop(0)

        # Print to terminal
        print(f"[YOLOTrain] {log_line}")

        # Broadcast to all active websocket queues safely
        for q in list(self.listeners):
            try:
                q.put_nowait(log_line)
            except Exception:
                pass

    def start_training(self, yaml_path: str = "dataset/traffic.yaml", model_path: str = "yolo11n.pt", epochs: int = 50, batch: int = 16, imgsz: int = 640, workers: int = 2):
        if self.is_running:
            return False, "Training is already in progress."

        if not os.path.exists(yaml_path):
            return False, f"YAML configuration file '{yaml_path}' not found."

        def train_worker():
            self.is_running = True
            self.logs.clear()
            self.latest_metrics = {
                "epoch": 0,
                "total_epochs": epochs,
                "loss": 0.0,
                "map50": 0.0,
                "status": "training"
            }
            self._broadcast_log(f"Initializing YOLO training with model '{model_path}' on dataset '{yaml_path}'...")
            self._broadcast_log(f"Parameters: Epochs={epochs}, Batch={batch}, ImgSz={imgsz}, Workers={workers}")

            try:
                from ultralytics import YOLO
                model = YOLO(model_path)

                def on_train_epoch_end(trainer):
                    try:
                        ep = trainer.epoch + 1
                        tot = trainer.epochs
                        metrics = getattr(trainer, "metrics", {})
                        loss = getattr(trainer, "loss", 0.0)

                        self.latest_metrics["epoch"] = ep
                        self.latest_metrics["total_epochs"] = tot
                        self.latest_metrics["loss"] = float(loss) if isinstance(loss, (int, float)) else 0.0

                        log_msg = f"Epoch {ep}/{tot} completed."
                        self._broadcast_log(log_msg)
                    except Exception as ex:
                        pass

                # Add callback if available
                model.add_callback("on_train_epoch_end", on_train_epoch_end)

                results = model.train(
                    data=os.path.abspath(yaml_path),
                    epochs=epochs,
                    batch=batch,
                    imgsz=imgsz,
                    workers=workers,
                    verbose=True
                )

                self.latest_metrics["status"] = "completed"
                self._broadcast_log(f"YOLO training completed successfully! Results saved to '{results.save_dir}'.")
            except Exception as e:
                self.latest_metrics["status"] = "error"
                self._broadcast_log(f"Training Error: {e}")
            finally:
                self.is_running = False

        thread = threading.Thread(target=train_worker, daemon=True)
        thread.start()
        return True, "YOLO training started successfully."

train_service = TrainService()
