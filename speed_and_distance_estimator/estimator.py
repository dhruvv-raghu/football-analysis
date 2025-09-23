# speed_and_distance_estimator/estimator.py

import cv2
import numpy as np
from utils.bbox_utils import get_foot_position, measure_distance
import matplotlib.pyplot as plt
import os


class SpeedAndDistanceEstimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24
        self.previous_velocities = {}

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        total_player_load = {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue
            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue

                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    if time_elapsed == 0:
                        continue

                    displacement_vector = np.array(end_position) - np.array(start_position)
                    velocity_vector_ms = displacement_vector / time_elapsed

                    speed_meters_per_second = np.linalg.norm(velocity_vector_ms)
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    acceleration_vector = np.array([0.0, 0.0])
                    object_key = (object, track_id)
                    if object_key in self.previous_velocities:
                        previous_velocity = self.previous_velocities[object_key]
                        acceleration_vector = (velocity_vector_ms - previous_velocity) / time_elapsed
                    self.previous_velocities[object_key] = velocity_vector_ms

                    acceleration_magnitude = np.linalg.norm(acceleration_vector)

                    if object not in total_player_load:
                        total_player_load[object] = {}
                    if track_id not in total_player_load[object]:
                        total_player_load[object][track_id] = 0
                    total_player_load[object][track_id] += acceleration_magnitude

                    distance_covered = measure_distance(start_position, end_position)
                    if object not in total_distance:
                        total_distance[object] = {}
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
                        tracks[object][frame_num_batch][track_id]['acceleration'] = acceleration_magnitude
                        tracks[object][frame_num_batch][track_id]['player_load'] = total_player_load[object][track_id]

    def draw_player_metrics(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    if "player_load" in track_info:
                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40
                        position = tuple(map(int, position))

                        speed = track_info.get('speed', 0)
                        distance = track_info.get('distance', 0)
                        acceleration = track_info.get('acceleration', 0)
                        player_load = track_info.get('player_load', 0)

                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{acceleration:.2f} m/s^2", (position[0], position[1] + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"Load: {player_load:.2f}", (position[0], position[1] + 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_frames.append(frame)
        return output_frames

    # NEW: Replaces the heatmap function to plot player formations and tracks over time.
    def plot_player_formations_and_tracks(self, tracks, frame_shape, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        # Number of previous frames to use for drawing player trails
        trail_length = 20

        for frame_num, player_track in enumerate(tracks['players']):
            fig, ax = plt.subplots(figsize=(16, 9))

            # Set up the plot to look like a soccer pitch
            ax.set_facecolor('#228B22')  # ForestGreen
            ax.set_xlim(0, frame_shape[1])
            ax.set_ylim(frame_shape[0], 0)  # Invert Y-axis to match image coordinates
            ax.axis('off')

            # Plot Players and their trails
            for track_id, track_info in player_track.items():
                if 'position_transformed' in track_info and track_info['position_transformed'] is not None:
                    pos = track_info['position_transformed']

                    # Convert team color from BGR (OpenCV) to RGB (Matplotlib)
                    team_color_bgr = track_info.get('team_color', (255, 255, 255))
                    team_color_rgb = (team_color_bgr[2] / 255., team_color_bgr[1] / 255., team_color_bgr[0] / 255.)

                    # Draw player trail
                    history_positions = []
                    for i in range(max(0, frame_num - trail_length), frame_num):
                        if track_id in tracks['players'][i] and 'position_transformed' in tracks['players'][i][
                            track_id]:
                            history_positions.append(tracks['players'][i][track_id]['position_transformed'])

                    if len(history_positions) > 1:
                        history_x, history_y = zip(*history_positions)
                        ax.plot(history_x, history_y, color=team_color_rgb, linewidth=2, alpha=0.6)

                    # Draw current player position
                    ax.scatter(pos[0], pos[1], color=team_color_rgb, s=150, edgecolors='black', zorder=5)
                    ax.text(pos[0], pos[1] - 15, str(track_id), color='white', fontsize=10, ha='center', va='center',
                            weight='bold')

            # Plot Ball
            ball_pos = tracks['ball'][frame_num].get(1, {}).get('position_transformed')
            if ball_pos:
                ax.scatter(ball_pos[0], ball_pos[1], color='yellow', s=80, edgecolors='black', zorder=5)

            ax.set_title(f'Frame: {frame_num}', color='white', fontsize=16)

            output_path = os.path.join(output_dir, f"formation_{frame_num:05d}.png")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)