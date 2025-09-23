import cv2
import numpy as np
from player_ball_assigner.assigner import PlayerBallAssigner
from utils.video_utils import read_video, save_video
from tracker.tracker import Tracker
from team_assigner.assigner import TeamAssigner
from camera_movement.estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance_estimator.estimator import SpeedAndDistanceEstimator

def main():
    # Load video
    video_frames = read_video("input_vids/input.mp4")

    # Initialize tracker
    tracker = Tracker("models/best.pt")

    # Use a separate stub file for tracks
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/tracks.pkl"
    )

    tracker.add_position_to_tracks(tracks)

    # Initialize camera movement estimator
    camera_movement = CameraMovementEstimator(video_frames[0])

    # Use a different stub file for camera movement
    camera_movement_per_frame = camera_movement.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/camera_movement.pkl"
    )
    camera_movement.adjust_track_positions(tracks, camera_movement_per_frame)

    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    # Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign team colors
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                player_id
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign ball possession
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # fall back to last known possession if exists
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(-1)  # no team at first frame

    team_ball_control = np.array(team_ball_control)

    # Draw annotations
    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Draw camera movement
    output_frames = camera_movement.draw_camera_movement(output_frames, camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_frames, tracks)

    # Save output video
    save_video(output_frames, "output_vids/output.mp4")


if __name__ == "__main__":
    main()
