# -*- coding: utf-8 -*-
"""
Bounding Box Annotation Tool
Created on Jul 8, 2025
"""

import os
import cv2
import tkinter as tk
from tkinter import filedialog
import json

class AnnotationApp:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))])
        self.index = 0
        self.annotations = {}

        self.window_name = "Bounding Box Annotator"
        self.drawing = False
        self.dragging = False
        self.drag_type = None  # 'move', 'tl', 'tr', 'bl', 'br'
        self.ix, self.iy = -1, -1
        self.offset_x, self.offset_y = 0, 0
        self.current_rect = None
        self.scale = 1.0

        self.load_annotations()
        self.process_images()

    def load_annotations(self):
        self.annotation_file = os.path.join(self.image_folder, "annotations.json")
        if os.path.exists(self.annotation_file):
            with open(self.annotation_file, "r") as f:
                self.annotations = json.load(f)

    def save_annotations(self):
        with open(self.annotation_file, "w") as f:
            json.dump(self.annotations, f, indent=4)

    def point_in_rect(self, x, y, rect):
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def near_corner(self, x, y, rect, threshold=10):
        x1, y1, x2, y2 = rect
        corners = {
            'tl': (x1, y1),
            'tr': (x2, y1),
            'bl': (x1, y2),
            'br': (x2, y2)
        }
        for name, (cx, cy) in corners.items():
            if abs(x - cx) < threshold and abs(y - cy) < threshold:
                return name
        return None

    def mouse_callback(self, event, x, y, flags, param):
        orig_x = int(x / self.scale)
        orig_y = int(y / self.scale)
    
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_rect:
                corner = self.near_corner(orig_x, orig_y, self.current_rect)
                if corner:
                    self.dragging = True
                    self.drag_type = corner
                elif self.point_in_rect(orig_x, orig_y, self.current_rect):
                    self.dragging = True
                    self.drag_type = 'move'
                    self.drag_start = (orig_x, orig_y)
                    x1, y1, x2, y2 = self.current_rect
                    self.initial_rect = (x1, y1, x2, y2)  # create a copy!
                else:
                    self.drawing = True
                    self.ix, self.iy = orig_x, orig_y
                    self.current_rect = None
            else:
                self.drawing = True
                self.ix, self.iy = orig_x, orig_y
    
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_rect = (self.ix, self.iy, orig_x, orig_y)
            elif self.dragging:
                if self.drag_type == 'move':
                    dx = orig_x - self.drag_start[0]
                    dy = orig_y - self.drag_start[1]
                    x1, y1, x2, y2 = self.initial_rect
                    self.current_rect = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
                elif self.drag_type == 'tl':
                    x2, y2 = self.current_rect[2], self.current_rect[3]
                    self.current_rect = (orig_x, orig_y, x2, y2)
                elif self.drag_type == 'tr':
                    x1, y2 = self.current_rect[0], self.current_rect[3]
                    self.current_rect = (x1, orig_y, orig_x, y2)
                elif self.drag_type == 'bl':
                    x2, y1 = self.current_rect[2], self.current_rect[1]
                    self.current_rect = (orig_x, y1, x2, orig_y)
                elif self.drag_type == 'br':
                    x1, y1 = self.current_rect[0], self.current_rect[1]
                    self.current_rect = (x1, y1, orig_x, orig_y)
    
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.dragging = False
            self.drag_type = None


    def save_current_annotation(self):
        if self.current_rect:
            x1, y1, x2, y2 = self.current_rect
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            cx = x + w // 2
            cy = y + h // 2

            image_name = self.image_files[self.index]
            self.annotations[image_name] = {
                "top_left": [x, y],
                "width": w,
                "height": h,
                "center": [cx, cy]
            }

            print(f"\nAnnotated {image_name}")
            print(f"Top-left: ({x}, {y})")
            print(f"Width: {w}")
            print(f"Height: {h}")
            print(f"Center: ({cx}, {cy})")

    def process_images(self):
        while self.index < len(self.image_files):
            img_path = os.path.join(self.image_folder, self.image_files[self.index])
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️ Could not load image: {img_path}")
                self.index += 1
                continue

            clone = image.copy()
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)

            # Carry over previous annotation
            if self.index > 0:
                prev_ann = self.annotations.get(self.image_files[self.index - 1])
                if prev_ann:
                    x, y = prev_ann["top_left"]
                    w = prev_ann["width"]
                    h = prev_ann["height"]
                    self.current_rect = (x, y, x + w, y + h)
                else:
                    self.current_rect = None
            else:
                self.current_rect = None

            while True:
                display = clone.copy()
                if self.current_rect:
                    x1, y1, x2, y2 = self.current_rect
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    for (cx, cy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                        cv2.circle(display, (cx, cy), 5, (255, 255, 255), -1)

                max_width, max_height = 1280, 720
                self.scale = min(max_width / display.shape[1], max_height / display.shape[0], 1.0)
                resized_display = cv2.resize(display, None, fx=self.scale, fy=self.scale)

                cv2.imshow(self.window_name, resized_display)
                key = cv2.waitKey(1) & 0xFF

                if key == 13:  # Enter key
                    self.save_current_annotation()
                    self.index += 1
                    break
                elif key == 27:  # Esc key
                    print("❌ Exiting...")
                    self.save_annotations()
                    cv2.destroyAllWindows()
                    return

        self.save_annotations()
        print("🎉 Annotation completed for all images!")
        cv2.destroyAllWindows()


def start_app():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder with Images")
    if folder_path:
        AnnotationApp(folder_path)
    else:
        print("No folder selected. Exiting.")

if __name__ == "__main__":
    start_app()
