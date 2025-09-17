import cv2

from utils.video_utils import read_video, save_video
from tracker.tracker import Tracker
from team_assigner.assigner import TeamAssigner



def main():
    
    tracker= Tracker("models/best.pt")
    
    video_frames= read_video("input_vids/input.mp4")
    
    tracks= tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/track.pkl")

    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]



    output_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_frames, "output_vids/output.mp4")
    
if __name__ == "__main__":
    main()