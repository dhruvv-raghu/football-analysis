import cv2

from utils.video_utils import read_video, save_video
from tracker.tracker import Tracker



def main():
    
    tracker= Tracker("models/best.pt")
    
    video_frames= read_video("input_vids/input.mp4")
    
    tracks= tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/track.pkl")

    for track_id, player in tracks['players'][0].items():
        bbox= player['bbox']
        frame= video_frames[0]

        cropped= frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        cv2.imwrite("output_vids/cropped_player.jpg", cropped)
        break

    output_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_frames, "output_vids/output.mp4")
    
if __name__ == "__main__":
    main()