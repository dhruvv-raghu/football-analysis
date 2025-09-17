    `for track_id, player in tracks['players'][0].items():
        bbox= player['bbox']
        frame= video_frames[0]

        cropped= frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        cv2.imwrite("output_vids/cropped_player.jpg", cropped)
        break`

# This is the code I used to generate a sample image for testing the clustering module.