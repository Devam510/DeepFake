import sys
import traceback
from video_detector import detect_video

print('Starting detect_video...', file=sys.stdout)
sys.stdout.flush()

try:
    res = detect_video('datasets/raw_real_videos/VID_20241114201922.mp4')
    print('SUCCESS:', res, file=sys.stdout)
except Exception as e:
    print(f'CRASH: {e}', file=sys.stdout)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
