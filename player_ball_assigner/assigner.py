from utils.bbox_utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball(self, player, ball_bbox):
        MINIMUM_DISTANCE = 99999
        assigned_player = -1

        ball_bbox = get_center_of_bbox(ball_bbox)

        for player_id, player in player.items():
            player_bbox = player['bbox']

            distance_left= measure_distance((player_bbox[0], player_bbox[-1]), ball_bbox)
            distance_right= measure_distance((player_bbox[2], player_bbox[-1]), ball_bbox)

            distance = min(distance_left, distance_right)

            if distance <= self.max_player_ball_distance:
                if distance < MINIMUM_DISTANCE:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player

