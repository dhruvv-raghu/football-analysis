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

    # Get object tracks from the video
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/tracks.pkl"
    )

    # Add pixel positions to tracks
    tracker.add_position_to_tracks(tracks)

    # Initialize and apply camera movement estimation
    camera_movement = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/camera_movement.pkl"
    )
    camera_movement.adjust_track_positions(tracks, camera_movement_per_frame)

    # Apply view transformation for a bird's-eye view perspective
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate ball positions for smoother tracking
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # Initialize and apply speed, distance, and player load calculations
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign team colors to players
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
            # Fall back to last known possession if it exists
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(-1)
    team_ball_control = np.array(team_ball_control)

    # --- DRAWING AND SAVING ---

    # Draw all annotations (ellipses, possession triangles, team control) onto the video frames
    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Draw camera movement arrows
    output_frames = camera_movement.draw_camera_movement(output_frames, camera_movement_per_frame)

    # Draw player metrics (speed, distance, acceleration, load) onto the video frames
    output_frames = speed_and_distance_estimator.draw_player_metrics(output_frames, tracks)

    # Save the final annotated video
    save_video(output_frames, "output_vids/output.mp4")

    # Generate and save formation plot images for each frame
    print("Generating formation and track plots...")
    speed_and_distance_estimator.plot_player_formations_and_tracks(
        tracks,
        video_frames[0].shape,
        "output_vids/formations"
    )
    print("Done generating plots.")


if __name__ == "__main__":
    main()