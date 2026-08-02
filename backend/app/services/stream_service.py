import os
import cv2
import glob
import time
import asyncio
from typing import List, Dict, Any, Optional, Callable

# Set FFmpeg timeout option for OpenCV to prevent network hanging
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000000"

class StreamCaptureService:
    def __init__(self):
        self.is_running = False
        self.should_cancel = False
        self.current_progress = 0
        self.total_frames = 0
        self.last_saved_frame = ""
        self.logs: List[str] = []

    def log(self, message: str):
        print(f"[StreamCapture] {message}")
        self.logs.append(message)
        if len(self.logs) > 500:
            self.logs.pop(0)

    def cancel(self):
        if self.is_running:
            self.should_cancel = True
            self.log("Cancel requested.")

    def run_capture(self, sources: List[str], num_frames: int, frame_skip: int, output_dir: str = "dataset_raw"):
        self.is_running = True
        self.should_cancel = False
        self.current_progress = 0
        self.total_frames = num_frames
        self.last_saved_frame = ""
        self.logs.clear()

        os.makedirs(output_dir, exist_ok=True)
        
        # Connect to streams
        caps = []
        for src in sources:
            try:
                src_val = int(src) if src.isdigit() else src
                self.log(f"Connecting to stream: {src_val}...")
                cap = cv2.VideoCapture(src_val)
                if cap.isOpened():
                    caps.append({"src": src_val, "cap": cap, "saved": 0, "read_count": 0, "fail_count": 0})
                else:
                    self.log(f"Failed to open stream: {src_val}")
            except Exception as e:
                self.log(f"Error opening {src}: {e}")

        if not caps:
            self.log("Failed to open any video streams.")
            self.is_running = False
            return False, "Failed to open any video streams."

        # Find highest existing frame index
        start_idx = 0
        existing_frames = glob.glob(os.path.join(output_dir, "frame_*.jpg"))
        if existing_frames:
            indices = []
            for f in existing_frames:
                try:
                    basename = os.path.basename(f)
                    idx = int(basename.replace("frame_", "").replace(".jpg", ""))
                    indices.append(idx)
                except ValueError:
                    pass
            if indices:
                start_idx = max(indices) + 1

        total_saved = 0
        self.log(f"Starting fair round-robin capture for {len(caps)} streams from frame_{start_idx:04d}.jpg...")

        try:
            # Round-robin loop to ensure balanced frame capture across ALL active streams
            active_index = 0
            
            while total_saved < num_frames and caps:
                if self.should_cancel:
                    self.log("Capture cancelled by user.")
                    break

                # Get current stream item
                item = caps[active_index % len(caps)]
                cap = item["cap"]
                src_name = item["src"]

                ret, frame = cap.read()
                if ret:
                    item["fail_count"] = 0
                    item["read_count"] += 1

                    # Apply frame skip interval
                    if item["read_count"] % max(1, frame_skip) == 0:
                        current_frame_id = start_idx + total_saved
                        filename = f"{output_dir}/frame_{current_frame_id:04d}.jpg"
                        cv2.imwrite(filename, frame)
                        total_saved += 1
                        item["saved"] += 1
                        self.current_progress = total_saved
                        self.last_saved_frame = filename
                        self.log(f"Saved frame {current_frame_id:04d} from {src_name} ({total_saved}/{num_frames})")
                else:
                    item["fail_count"] += 1
                    # If stream fails repeatedly, attempt auto-reconnect instead of instantly dropping
                    if item["fail_count"] == 3:
                        self.log(f"Stream {src_name} unresponsive, attempting auto-reconnect...")
                        try:
                            cap.release()
                            new_cap = cv2.VideoCapture(src_name)
                            if new_cap.isOpened():
                                item["cap"] = new_cap
                                item["fail_count"] = 0
                                self.log(f"Successfully reconnected to {src_name}")
                            else:
                                self.log(f"Failed to reconnect to {src_name}. Removing stream.")
                                caps.remove(item)
                        except Exception as ex:
                            self.log(f"Reconnect error for {src_name}: {ex}")
                            caps.remove(item)

                # Move to next stream in round-robin sequence
                active_index += 1
                time.sleep(0.01)

        finally:
            for item in caps:
                try:
                    item["cap"].release()
                except Exception:
                    pass

        self.is_running = False
        msg = f"Done! Saved {total_saved} frames to '{output_dir}' across {len(sources)} streams."
        self.log(msg)
        return True, msg

stream_capture_service = StreamCaptureService()
