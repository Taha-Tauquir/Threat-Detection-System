import cv2

video_path = "/Users/mzeeshan/Documents/PythonProjects/NewRppg/train/3020/P24-Rec1-2009.09.01.11.59.10_C1 trigger _C_Section_30_trimmed_original.avi"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

print(f"FPS: {fps}")
print(f"Total frames: {frame_count}")
print(f"Duration (sec): {duration:.2f}")
cap.release()