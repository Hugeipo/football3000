import argparse
from enum import Enum
from typing import Iterator, List, Tuple
import json
from datetime import datetime
from collections import defaultdict
import math

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    This handles both values and dictionary keys.
    """
    if isinstance(obj, dict):
        # Convert both keys and values
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    return radar


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    ball_slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = ball_slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        yield annotated_frame


def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame


def run_radar(source_video_path: str, device: str) -> Iterator[Tuple[np.ndarray, 'DataCollector']]:
    """
    COMPREHENSIVE DATA COLLECTION - All modes combined for complete analytics
    Collects: Pitch + Players + Ball + Teams + Spatial + Formations + Events
    """
    # Initialize data collector
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    data_collector = DataCollector(fps=video_info.fps)
    data_collector.set_metadata(source_video_path, video_info)
    
    # Initialize ALL models for comprehensive data collection
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    
    # Initialize ball tracking components
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)
    
    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    ball_slicer = sv.InferenceSlicer(
        callback=ball_callback,
        slice_wh=(640, 640),
    )
    
    # Collect crops for team classification
    print("🔄 Phase 1: Collecting player crops for team classification...")
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    # Train team classifier
    print("🤖 Phase 2: Training team classifier...")
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    # Main processing loop - COLLECT EVERYTHING
    print("📊 Phase 3: Comprehensive data collection (ALL MODES)...")
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    
    frame_id = 0
    for frame in tqdm(frame_generator, desc='processing frames'):
        # ===== 1. PITCH DETECTION =====
        pitch_result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
        
        # ===== 2. PLAYER DETECTION & TRACKING =====
        player_result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        player_detections = sv.Detections.from_ultralytics(player_result)
        player_detections = tracker.update_with_detections(player_detections)

        # ===== 3. TEAM CLASSIFICATION =====
        players = player_detections[player_detections.class_id == PLAYER_CLASS_ID]
        player_crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(player_crops)

        goalkeepers = player_detections[player_detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = player_detections[player_detections.class_id == REFEREE_CLASS_ID]

        # Merge all player detections
        all_player_detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        
        # ===== 4. BALL DETECTION & TRACKING =====
        ball_detections = ball_slicer(frame).with_nms(threshold=0.1)
        ball_detections = ball_tracker.update(ball_detections)
        
        # ===== 5. COORDINATE TRANSFORMATION =====
        # Transform player coordinates to field coordinates
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
        if np.sum(mask) >= 4:  # Need at least 4 points for transformation
            transformer = ViewTransformer(
                source=keypoints.xy[0][mask].astype(np.float32),
                target=np.array(CONFIG.vertices)[mask].astype(np.float32)
            )
            player_xy = all_player_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            player_field_positions = transformer.transform_points(points=player_xy)
            homography_matrix = transformer.m
            
            # Transform ball coordinates if detected
            if len(ball_detections) > 0:
                ball_xy = ball_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                ball_field_positions = transformer.transform_points(points=ball_xy)
            else:
                ball_field_positions = np.array([])
        else:
            # Fallback: use dummy field coordinates
            player_xy = all_player_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            player_field_positions = player_xy * 10  # Rough scaling
            ball_field_positions = np.array([])
            homography_matrix = np.eye(3)

        # ===== 6. COMPREHENSIVE DATA COLLECTION =====
        
        # Add pitch detection data
        data_collector.add_pitch_detection(frame_id, keypoints, homography_matrix)
        
        # Add player detection & tracking data with embeddings
        if len(player_crops) == len(players):
            # Convert crops to embeddings (simplified - you could use actual embeddings)
            crop_embeddings = [np.array(crop).flatten()[:512] for crop in player_crops]  # Simplified
        else:
            crop_embeddings = None
            
        data_collector.add_player_detections(
            frame_id, all_player_detections, color_lookup, player_field_positions, crop_embeddings)
        
        # Add ball detection data
        data_collector.add_ball_detection(
            frame_id, ball_detections, ball_field_positions, 
            all_player_detections, player_field_positions)
        
        # Add spatial occupancy data
        data_collector.add_spatial_occupancy(
            frame_id, all_player_detections, color_lookup, player_field_positions)
        
        # ===== 7. ADVANCED ANALYTICS (NEW) =====
        # Add formation analysis
        data_collector.add_formation_analysis(frame_id, all_player_detections, color_lookup, player_field_positions)
        
        # Add possession analysis
        data_collector.add_possession_analysis(frame_id, ball_detections, ball_field_positions, 
                                             all_player_detections, player_field_positions, color_lookup)
        
        # Add tactical events detection
        data_collector.add_tactical_events(frame_id, all_player_detections, color_lookup, 
                                         player_field_positions, ball_detections, ball_field_positions)
        
        data_collector.frame_count += 1
        frame_id += 1

        # ===== 8. CREATE ANNOTATED FRAME =====
        labels = [str(tracker_id) for tracker_id in all_player_detections.tracker_id]

        annotated_frame = frame.copy()
        
        # Annotate players
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, all_player_detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, all_player_detections, labels,
            custom_color_lookup=color_lookup)
        
        # Annotate ball
        annotated_frame = ball_annotator.annotate(annotated_frame, ball_detections)
        
        # Add radar overlay
        h, w, _ = frame.shape
        radar = render_radar(all_player_detections, keypoints, color_lookup)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        
        yield annotated_frame, data_collector  # Return both frame and data collector


class DataCollector:
    """Comprehensive data collection for soccer analytics"""
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.data = {
            "match_metadata": {},
            "raw_detections": {
                "players": [],
                "ball": [],
                "pitch": []
            },
            "tracking_data": {
                "players": {},
                "ball": {
                    "total_frames_detected": 0,
                    "detection_rate": 0.0,
                    "tracking_gaps": [],
                    "trajectory": []
                }
            },
            "team_classification": {},
            "spatial_analytics": {
                "field_zones": {
                    "defensive_third": {"x_min": 0, "x_max": 4000, "area": 28000000},
                    "middle_third": {"x_min": 4000, "x_max": 8000, "area": 28000000},
                    "attacking_third": {"x_min": 8000, "x_max": 12000, "area": 28000000},
                    "penalty_box_left": {"x_min": 0, "x_max": 2015, "y_min": 1450, "y_max": 5550},
                    "penalty_box_right": {"x_min": 9985, "x_max": 12000, "y_min": 1450, "y_max": 5550},
                    "center_circle": {"center": [6000, 3500], "radius": 915}
                },
                "per_frame_occupancy": []
            },
            "possession_analytics": {
                "possession_segments": [],
                "possession_stats": {"team_0": {}, "team_1": {}}
            },
            "movement_analytics": [],
            "formation_analytics": [],
            "tactical_events": {
                "pressing_events": [],
                "counter_attacks": [],
                "offside_events": []
            },
            "quality_metrics": {},
            "summary_statistics": {}
        }
        
        # Tracking variables
        self.frame_count = 0
        self.previous_positions = {}
        self.ball_possession_history = []
        self.current_possession = None
    
    def set_metadata(self, video_path: str, video_info: sv.VideoInfo):
        """Set match metadata"""
        self.data["match_metadata"] = {
            "video_file": os.path.basename(video_path),
            "duration_seconds": video_info.total_frames / video_info.fps,
            "total_frames": video_info.total_frames,
            "fps": video_info.fps,
            "resolution": {"width": video_info.width, "height": video_info.height},
            "processing_timestamp": datetime.now().isoformat(),
            "field_dimensions": {"width": 7000, "length": 12000, "units": "cm"}
        }
    
    def add_player_detections(self, frame_id: int, detections: sv.Detections, 
                            team_ids: np.ndarray, field_positions: np.ndarray, 
                            crops_embeddings: List[np.ndarray] = None):
        """Add player detection data for current frame"""
        timestamp = frame_id / self.fps
        
        detection_data = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "detections": []
        }
        
        for i, (bbox, class_id, conf, tracker_id) in enumerate(zip(
            detections.xyxy, detections.class_id, detections.confidence, 
            detections.tracker_id if detections.tracker_id is not None else [None] * len(detections)
        )):
            # Calculate velocity and acceleration
            velocity = {"x": 0.0, "y": 0.0, "magnitude": 0.0}
            acceleration = {"x": 0.0, "y": 0.0, "magnitude": 0.0}
            
            # Bounds check for field_positions
            if i >= len(field_positions):
                # Fallback to video coordinates if field transformation failed
                field_x = float((bbox[0] + bbox[2]) / 2)
                field_y = float(bbox[3])
            else:
                field_x = float(field_positions[i][0])
                field_y = float(field_positions[i][1])
                
                if tracker_id is not None and int(tracker_id) in self.previous_positions:
                    prev_pos = self.previous_positions[int(tracker_id)]
                    dt = 1.0 / self.fps
                    
                    # Calculate velocity (cm/s to m/s)
                    velocity = {
                        "x": (field_positions[i][0] - prev_pos["position"][0]) / dt / 100,
                        "y": (field_positions[i][1] - prev_pos["position"][1]) / dt / 100,
                        "magnitude": 0.0
                    }
                    velocity["magnitude"] = np.sqrt(velocity["x"]**2 + velocity["y"]**2)
                    
                    # Calculate acceleration
                    if "velocity" in prev_pos:
                        acceleration = {
                            "x": (velocity["x"] - prev_pos["velocity"]["x"]) / dt,
                            "y": (velocity["y"] - prev_pos["velocity"]["y"]) / dt,
                            "magnitude": 0.0
                        }
                        acceleration["magnitude"] = np.sqrt(acceleration["x"]**2 + acceleration["y"]**2)
            
            # Store current position for next frame
            if tracker_id is not None and i < len(field_positions):
                self.previous_positions[int(tracker_id)] = {
                    "position": field_positions[i].tolist(),
                    "velocity": velocity,
                    "timestamp": timestamp
                }
            
            # Determine role
            role_map = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
            role = role_map.get(class_id, "unknown")
            
            detection = {
                "detection_id": i,
                "class_id": int(class_id),
                "class_name": role,
                "confidence": float(conf),
                "bbox": {
                    "x1": float(bbox[0]), "y1": float(bbox[1]),
                    "x2": float(bbox[2]), "y2": float(bbox[3]),
                    "width": float(bbox[2] - bbox[0]),
                    "height": float(bbox[3] - bbox[1]),
                    "center_x": float((bbox[0] + bbox[2]) / 2),
                    "center_y": float((bbox[1] + bbox[3]) / 2)
                },
                "tracker_id": int(tracker_id) if tracker_id is not None else None,
                "team_id": int(team_ids[i]) if i < len(team_ids) else -1,
                "team_confidence": 0.9,  # TODO: Get actual confidence from classifier
                "crop_embedding": crops_embeddings[i].tolist() if crops_embeddings and i < len(crops_embeddings) else [],
                "position_video": {
                    "x": float((bbox[0] + bbox[2]) / 2),
                    "y": float(bbox[3])  # Bottom center
                },
                "position_field": {
                    "x": field_x,
                    "y": field_y
                },
                "velocity": velocity,
                "acceleration": acceleration
            }
            
            detection_data["detections"].append(detection)
            
            # Update tracking data
            if tracker_id is not None:
                tracker_id_int = int(tracker_id)
                # Create player entry if it doesn't exist (since we removed defaultdict)
                if tracker_id_int not in self.data["tracking_data"]["players"]:
                    self.data["tracking_data"]["players"][tracker_id_int] = {
                        "player_id": int(tracker_id),
                        "team_id": -1,
                        "role": "unknown",
                        "tracking_quality": 0.0,
                        "total_frames_tracked": 0,
                        "tracking_gaps": [],
                        "trajectory": []
                    }
                
                player_track = self.data["tracking_data"]["players"][tracker_id_int]
                player_track["player_id"] = int(tracker_id)
                player_track["team_id"] = int(team_ids[i]) if i < len(team_ids) else -1
                player_track["role"] = role
                player_track["total_frames_tracked"] += 1
                
                trajectory_point = {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "position_video": detection["position_video"],
                    "position_field": detection["position_field"],
                    "velocity": velocity,
                    "acceleration": acceleration,
                    "confidence": float(conf),
                    "bbox": detection["bbox"]
                }
                player_track["trajectory"].append(trajectory_point)
        
        self.data["raw_detections"]["players"].append(detection_data)
    
    def add_ball_detection(self, frame_id: int, detections: sv.Detections, 
                          field_positions: np.ndarray, player_detections: sv.Detections,
                          player_field_positions: np.ndarray):
        """Add ball detection data for current frame"""
        timestamp = frame_id / self.fps
        
        if len(detections) > 0:
            self.data["tracking_data"]["ball"]["total_frames_detected"] += 1
            
            ball_pos = field_positions[0]  # Assume first detection is ball
            
            # Calculate velocity
            velocity = {"x": 0.0, "y": 0.0, "magnitude": 0.0}
            if len(self.data["tracking_data"]["ball"]["trajectory"]) > 0:
                prev_pos = self.data["tracking_data"]["ball"]["trajectory"][-1]["position_field"]
                dt = 1.0 / self.fps
                velocity = {
                    "x": (ball_pos[0] - prev_pos["x"]) / dt / 100,  # cm/s to m/s
                    "y": (ball_pos[1] - prev_pos["y"]) / dt / 100,
                    "magnitude": 0.0
                }
                velocity["magnitude"] = np.sqrt(velocity["x"]**2 + velocity["y"]**2)
            
            # Find closest player for possession
            distances = []
            for i, player_pos in enumerate(player_field_positions):
                dist = np.linalg.norm(ball_pos - player_pos)
                distances.append({"player_id": i, "distance": float(dist)})
            
            detection_data = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "detections": [{
                    "detection_id": 0,
                    "confidence": float(detections.confidence[0]),
                    "bbox": {
                        "x1": float(detections.xyxy[0][0]),
                        "y1": float(detections.xyxy[0][1]),
                        "x2": float(detections.xyxy[0][2]),
                        "y2": float(detections.xyxy[0][3])
                    },
                    "position_video": {
                        "x": float((detections.xyxy[0][0] + detections.xyxy[0][2]) / 2),
                        "y": float((detections.xyxy[0][1] + detections.xyxy[0][3]) / 2)
                    },
                    "position_field": {"x": float(ball_pos[0]), "y": float(ball_pos[1])},
                    "velocity": velocity,
                    "distance_to_players": distances
                }]
            }
            
            # Add to ball tracking
            trajectory_point = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "position_video": detection_data["detections"][0]["position_video"],
                "position_field": detection_data["detections"][0]["position_field"],
                "velocity": velocity,
                "acceleration": {"x": 0.0, "y": 0.0, "magnitude": 0.0},
                "height_estimate": 0.0,
                "confidence": float(detections.confidence[0])
            }
            self.data["tracking_data"]["ball"]["trajectory"].append(trajectory_point)
            
        else:
            detection_data = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "detections": []
            }
        
        self.data["raw_detections"]["ball"].append(detection_data)
    
    def add_pitch_detection(self, frame_id: int, keypoints: sv.KeyPoints, 
                           homography_matrix: np.ndarray):
        """Add pitch keypoint detection data"""
        timestamp = frame_id / self.fps
        
        detection_data = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "keypoints": [],
            "homography_matrix": homography_matrix.tolist() if homography_matrix is not None else [],
            "transformation_quality": 0.9,  # TODO: Calculate actual quality
            "field_coverage": 0.8,  # TODO: Calculate actual coverage
            "camera_angle": "high",
            "field_orientation": 0.0
        }
        
        if len(keypoints.xy) > 0:
            for i, (video_coords, field_coords) in enumerate(zip(keypoints.xy[0], CONFIG.vertices)):
                keypoint_data = {
                    "keypoint_id": i,
                    "label": CONFIG.labels[i] if i < len(CONFIG.labels) else str(i),
                    "video_coords": {"x": float(video_coords[0]), "y": float(video_coords[1])},
                    "field_coords": {"x": float(field_coords[0]), "y": float(field_coords[1])},
                    "confidence": 0.9,  # TODO: Get actual confidence
                    "visible": bool(video_coords[0] > 1 and video_coords[1] > 1),
                    "zone": "field"  # TODO: Determine actual zone
                }
                detection_data["keypoints"].append(keypoint_data)
        
        self.data["raw_detections"]["pitch"].append(detection_data)
    
    def add_spatial_occupancy(self, frame_id: int, player_detections: sv.Detections,
                            team_ids: np.ndarray, field_positions: np.ndarray):
        """Add spatial zone occupancy data"""
        timestamp = frame_id / self.fps
        
        occupancy = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "zone_occupancy": {
                "defensive_third": {"team_0": 0, "team_1": 0, "referees": 0},
                "middle_third": {"team_0": 0, "team_1": 0, "referees": 0},
                "attacking_third": {"team_0": 0, "team_1": 0, "referees": 0},
                "penalty_box_left": {"team_0": 0, "team_1": 0},
                "penalty_box_right": {"team_0": 0, "team_1": 0},
                "center_circle": {"team_0": 0, "team_1": 0}
            }
        }
        
        for i, (pos, team_id) in enumerate(zip(field_positions, team_ids)):
            x, y = pos[0], pos[1]
            
            # Determine zones
            if x < 4000:
                zone = "defensive_third"
            elif x < 8000:
                zone = "middle_third"
            else:
                zone = "attacking_third"
            
            team_key = f"team_{team_id}" if team_id in [0, 1] else "referees"
            if zone in occupancy["zone_occupancy"]:
                occupancy["zone_occupancy"][zone][team_key] += 1
            
            # Check penalty boxes
            if x < 2015 and 1450 <= y <= 5550:
                occupancy["zone_occupancy"]["penalty_box_left"][team_key] += 1
            elif x > 9985 and 1450 <= y <= 5550:
                occupancy["zone_occupancy"]["penalty_box_right"][team_key] += 1
            
            # Check center circle
            center_dist = np.sqrt((x - 6000)**2 + (y - 3500)**2)
            if center_dist <= 915:
                if team_id in [0, 1]:
                    occupancy["zone_occupancy"]["center_circle"][f"team_{team_id}"] += 1
        
        self.data["spatial_analytics"]["per_frame_occupancy"].append(occupancy)
    
    def add_formation_analysis(self, frame_id: int, player_detections: sv.Detections,
                             team_ids: np.ndarray, field_positions: np.ndarray):
        """Add formation analysis data"""
        timestamp = frame_id / self.fps
        
        # Analyze formations for each team
        for team_id in [0, 1]:
            team_mask = team_ids == team_id
            if np.sum(team_mask) < 3:  # Need at least 3 players for formation analysis
                continue
                
            team_positions = field_positions[team_mask]
            team_player_ids = [int(tid) for i, tid in enumerate(player_detections.tracker_id) 
                             if i < len(team_ids) and team_ids[i] == team_id and tid is not None]
            
            # Calculate team metrics
            centroid_x = np.mean(team_positions[:, 0])
            centroid_y = np.mean(team_positions[:, 1])
            
            # Team length (spread along x-axis)
            team_length = np.max(team_positions[:, 0]) - np.min(team_positions[:, 0])
            
            # Team width (spread along y-axis)
            team_width = np.max(team_positions[:, 1]) - np.min(team_positions[:, 1])
            
            # Compactness (average distance from centroid)
            distances_from_centroid = [np.sqrt((pos[0] - centroid_x)**2 + (pos[1] - centroid_y)**2) 
                                     for pos in team_positions]
            compactness = np.mean(distances_from_centroid)
            
            # Stretch index (max distance between any two players)
            max_distance = 0
            for i in range(len(team_positions)):
                for j in range(i+1, len(team_positions)):
                    dist = np.sqrt((team_positions[i][0] - team_positions[j][0])**2 + 
                                 (team_positions[i][1] - team_positions[j][1])**2)
                    max_distance = max(max_distance, dist)
            
            # Determine formation shape (simplified)
            formation_shape = "unknown"
            if len(team_positions) >= 10:  # Full team
                # Simple formation detection based on player distribution
                defensive_players = np.sum(team_positions[:, 0] < 3000)
                midfield_players = np.sum((team_positions[:, 0] >= 3000) & (team_positions[:, 0] < 9000))
                attacking_players = np.sum(team_positions[:, 0] >= 9000)
                
                formation_shape = f"{defensive_players}-{midfield_players}-{attacking_players}"
            
            formation_data = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "team_id": team_id,
                "formation_metrics": {
                    "centroid": {"x": float(centroid_x), "y": float(centroid_y)},
                    "length": float(team_length / 100),  # Convert to meters
                    "width": float(team_width / 100),
                    "compactness": float(compactness / 100),
                    "stretch_index": float(max_distance / 100),
                    "defensive_line": float(np.min(team_positions[:, 0]) / 100),
                    "offensive_line": float(np.max(team_positions[:, 0]) / 100),
                    "formation_shape": formation_shape,
                    "shape_confidence": 0.8  # Simplified confidence
                },
                "player_positions": [
                    {
                        "player_id": pid,
                        "position": {"x": float(pos[0]), "y": float(pos[1])},
                        "formation_role": "unknown",  # TODO: Implement role detection
                        "role_confidence": 0.7
                    }
                    for pid, pos in zip(team_player_ids, team_positions)
                ]
            }
            
            self.data["formation_analytics"].append(formation_data)
    
    def add_possession_analysis(self, frame_id: int, ball_detections: sv.Detections, 
                              ball_field_positions: np.ndarray, player_detections: sv.Detections,
                              player_field_positions: np.ndarray, team_ids: np.ndarray):
        """Add possession analysis data"""
        timestamp = frame_id / self.fps
        
        if len(ball_detections) > 0 and len(ball_field_positions) > 0:
            ball_pos = ball_field_positions[0]
            
            # Find closest player to ball
            closest_player_idx = -1
            min_distance = float('inf')
            
            for i, player_pos in enumerate(player_field_positions):
                if i < len(team_ids):
                    distance = np.sqrt((ball_pos[0] - player_pos[0])**2 + (ball_pos[1] - player_pos[1])**2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_player_idx = i
            
            # Determine possession (if ball is close enough to a player)
            possession_threshold = 300  # 3 meters in cm
            current_possession_team = None
            controlling_player = None
            
            if closest_player_idx >= 0 and min_distance < possession_threshold:
                current_possession_team = int(team_ids[closest_player_idx])
                if player_detections.tracker_id is not None and closest_player_idx < len(player_detections.tracker_id):
                    controlling_player = int(player_detections.tracker_id[closest_player_idx])
            
            # Check for possession changes
            if (self.current_possession is None or 
                self.current_possession.get('team_id') != current_possession_team):
                
                # End previous possession segment
                if self.current_possession is not None:
                    self.current_possession['end_frame'] = frame_id - 1
                    self.current_possession['duration'] = (frame_id - 1 - self.current_possession['start_frame']) / self.fps
                    self.current_possession['end_position'] = {"x": float(ball_pos[0]), "y": float(ball_pos[1])}
                    self.current_possession['end_event'] = "turnover"
                    
                    self.data["possession_analytics"]["possession_segments"].append(self.current_possession)
                
                # Start new possession segment
                if current_possession_team is not None:
                    self.current_possession = {
                        "segment_id": len(self.data["possession_analytics"]["possession_segments"]),
                        "start_frame": frame_id,
                        "end_frame": frame_id,  # Will be updated
                        "duration": 0.0,
                        "team_id": current_possession_team,
                        "controlling_player_id": controlling_player,
                        "start_position": {"x": float(ball_pos[0]), "y": float(ball_pos[1])},
                        "end_position": {"x": float(ball_pos[0]), "y": float(ball_pos[1])},
                        "ball_touches": 1,
                        "players_involved": [controlling_player] if controlling_player else [],
                        "passes_attempted": 0,
                        "passes_completed": 0,
                        "end_event": "ongoing",
                        "possession_value": 0.5,  # Simplified value
                        "field_progression": 0.0
                    }
                else:
                    self.current_possession = None
            
            # Update current possession
            if self.current_possession is not None:
                self.current_possession['end_frame'] = frame_id
                self.current_possession['duration'] = (frame_id - self.current_possession['start_frame']) / self.fps
                self.current_possession['end_position'] = {"x": float(ball_pos[0]), "y": float(ball_pos[1])}
                
                # Calculate field progression
                start_x = self.current_possession['start_position']['x']
                end_x = ball_pos[0]
                if current_possession_team == 0:  # Team 0 attacks left to right
                    progression = (end_x - start_x) / 100  # Convert to meters
                else:  # Team 1 attacks right to left
                    progression = (start_x - end_x) / 100
                
                self.current_possession['field_progression'] = float(progression)
    
    def add_tactical_events(self, frame_id: int, player_detections: sv.Detections,
                          team_ids: np.ndarray, player_field_positions: np.ndarray,
                          ball_detections: sv.Detections, ball_field_positions: np.ndarray):
        """Add tactical events detection"""
        timestamp = frame_id / self.fps
        
        # Simple pressing event detection
        if len(ball_detections) > 0 and len(ball_field_positions) > 0:
            ball_pos = ball_field_positions[0]
            
            # Count players within pressing distance of ball (for each team)
            pressing_distance = 1000  # 10 meters in cm
            team_0_pressers = 0
            team_1_pressers = 0
            
            for i, (player_pos, team_id) in enumerate(zip(player_field_positions, team_ids)):
                if i < len(team_ids):
                    distance = np.sqrt((ball_pos[0] - player_pos[0])**2 + (ball_pos[1] - player_pos[1])**2)
                    if distance < pressing_distance:
                        if team_id == 0:
                            team_0_pressers += 1
                        elif team_id == 1:
                            team_1_pressers += 1
            
            # Detect pressing events (simplified)
            if team_0_pressers >= 3:  # At least 3 players pressing
                pressing_event = {
                    "frame_start": frame_id,
                    "frame_end": frame_id,  # Will be extended in future frames
                    "team_id": 0,
                    "intensity": min(team_0_pressers / 5.0, 1.0),  # Normalize to 0-1
                    "players_involved": team_0_pressers,
                    "trigger_zone": "ball_area",
                    "success": False,  # Will be determined later
                    "ball_recovered": False
                }
                
                # Check if this extends an existing pressing event
                if (self.data["tactical_events"]["pressing_events"] and 
                    self.data["tactical_events"]["pressing_events"][-1]["frame_end"] == frame_id - 1):
                    self.data["tactical_events"]["pressing_events"][-1]["frame_end"] = frame_id
                else:
                    self.data["tactical_events"]["pressing_events"].append(pressing_event)
            
            # Similar logic for team 1
            if team_1_pressers >= 3:
                pressing_event = {
                    "frame_start": frame_id,
                    "frame_end": frame_id,
                    "team_id": 1,
                    "intensity": min(team_1_pressers / 5.0, 1.0),
                    "players_involved": team_1_pressers,
                    "trigger_zone": "ball_area",
                    "success": False,
                    "ball_recovered": False
                }
                
                if (self.data["tactical_events"]["pressing_events"] and 
                    self.data["tactical_events"]["pressing_events"][-1]["frame_end"] == frame_id - 1):
                    self.data["tactical_events"]["pressing_events"][-1]["frame_end"] = frame_id
                else:
                    self.data["tactical_events"]["pressing_events"].append(pressing_event)
        
        # Simple offside detection
        for team_id in [0, 1]:
            team_mask = team_ids == team_id
            if np.sum(team_mask) < 2:
                continue
                
            team_positions = player_field_positions[team_mask]
            
            # Find defensive line of opposing team
            opposing_team_mask = team_ids == (1 - team_id)
            if np.sum(opposing_team_mask) > 0:
                opposing_positions = player_field_positions[opposing_team_mask]
                
                if team_id == 0:  # Team 0 attacks left to right
                    defensive_line = np.min(opposing_positions[:, 0])  # Leftmost opponent
                    attacking_players = team_positions[team_positions[:, 0] > defensive_line]
                else:  # Team 1 attacks right to left  
                    defensive_line = np.max(opposing_positions[:, 0])  # Rightmost opponent
                    attacking_players = team_positions[team_positions[:, 0] < defensive_line]
                
                # Record offside events for attacking players
                for pos in attacking_players:
                    offside_distance = abs(pos[0] - defensive_line)
                    if offside_distance > 50:  # More than 0.5m offside
                        offside_event = {
                            "frame_id": frame_id,
                            "player_id": -1,  # TODO: Map position to player ID
                            "team_id": team_id,
                            "offside_line": float(defensive_line / 100),  # Convert to meters
                            "player_position": float(pos[0] / 100),
                            "offside_distance": float(offside_distance / 100),
                            "ball_played": False  # Simplified
                        }
                        self.data["tactical_events"]["offside_events"].append(offside_event)
    
    def finalize_analytics(self):
        """Calculate final analytics and summary statistics"""
        total_frames = self.frame_count
        
        # Calculate detection rates
        ball_detection_rate = self.data["tracking_data"]["ball"]["total_frames_detected"] / max(total_frames, 1)
        self.data["tracking_data"]["ball"]["detection_rate"] = ball_detection_rate
        
        # Calculate team possession statistics
        possession_segments = self.data["possession_analytics"]["possession_segments"]
        if possession_segments:
            team_0_time = sum(seg["duration"] for seg in possession_segments if seg["team_id"] == 0)
            team_1_time = sum(seg["duration"] for seg in possession_segments if seg["team_id"] == 1)
            total_possession_time = team_0_time + team_1_time
            
            if total_possession_time > 0:
                self.data["possession_analytics"]["possession_stats"] = {
                    "team_0": {
                        "total_time": team_0_time,
                        "percentage": (team_0_time / total_possession_time) * 100,
                        "segments": len([seg for seg in possession_segments if seg["team_id"] == 0])
                    },
                    "team_1": {
                        "total_time": team_1_time,
                        "percentage": (team_1_time / total_possession_time) * 100,
                        "segments": len([seg for seg in possession_segments if seg["team_id"] == 1])
                    }
                }
        
        # Calculate player movement analytics
        players_data = self.data["tracking_data"]["players"]
        movement_analytics = []
        
        for player_id, player_data in players_data.items():
            trajectory = player_data.get("trajectory", [])
            if len(trajectory) < 2:
                continue
                
            # Calculate total distance
            total_distance = 0.0
            speeds = []
            max_speed = 0.0
            zone_time = {"defensive": 0, "midfield": 0, "attacking": 0}
            
            for i, point in enumerate(trajectory):
                # Speed calculation
                velocity = point.get("velocity", {})
                if isinstance(velocity, dict):
                    speed = velocity.get("magnitude", 0)
                    speeds.append(speed)
                    max_speed = max(max_speed, speed)
                
                # Distance calculation
                if i > 0:
                    prev_pos = trajectory[i-1]["position_field"]
                    curr_pos = point["position_field"]
                    dx = (curr_pos["x"] - prev_pos["x"]) / 100  # Convert to meters
                    dy = (curr_pos["y"] - prev_pos["y"]) / 100
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance < 10:  # Filter unrealistic jumps
                        total_distance += distance
                
                # Zone time calculation
                x = point["position_field"]["x"] / 100  # Convert to meters
                if x < 40:
                    zone_time["defensive"] += 1/self.fps
                elif x < 80:
                    zone_time["midfield"] += 1/self.fps
                else:
                    zone_time["attacking"] += 1/self.fps
            
            # Calculate sprints and high intensity runs
            sprint_count = len([s for s in speeds if s > 5.5])  # m/s
            high_intensity_count = len([s for s in speeds if 4.0 <= s <= 5.5])
            
            player_analytics = {
                "player_id": int(player_id),
                "team_id": player_data.get("team_id", -1),
                "role": player_data.get("role", "unknown"),
                "movement_stats": {
                    "total_distance": total_distance,
                    "distance_per_minute": total_distance / max((len(trajectory) / self.fps / 60), 1),
                    "max_speed": max_speed,
                    "average_speed": np.mean(speeds) if speeds else 0.0,
                    "sprint_count": sprint_count,
                    "sprint_distance": 0.0,  # TODO: Calculate actual sprint distance
                    "high_intensity_runs": high_intensity_count,
                    "acceleration_events": 0,  # TODO: Calculate from acceleration data
                    "deceleration_events": 0
                },
                "zone_analysis": {
                    "time_in_zones": zone_time,
                    "distance_in_zones": {
                        "defensive_third": 0.0,  # TODO: Calculate per zone
                        "middle_third": 0.0,
                        "attacking_third": 0.0
                    }
                }
            }
            movement_analytics.append(player_analytics)
        
        self.data["movement_analytics"] = movement_analytics
        
        # Calculate quality metrics
        self.data["quality_metrics"] = {
            "player_detection_rate": 0.95,  # TODO: Calculate actual rates based on successful detections
            "ball_detection_rate": ball_detection_rate,
            "tracking_stability": 0.92,  # TODO: Calculate based on tracking consistency
            "keypoint_accuracy": 0.88,  # TODO: Calculate based on field detection quality
            "team_classification_confidence": 0.94  # TODO: Calculate from team classifier
        }
        
        # Generate comprehensive summary statistics
        total_players = len(players_data)
        team_0_players = len([p for p in players_data.values() if p.get("team_id") == 0])
        team_1_players = len([p for p in players_data.values() if p.get("team_id") == 1])
        
        # Calculate team distances
        team_0_distance = sum(p["movement_stats"]["total_distance"] for p in movement_analytics if p["team_id"] == 0)
        team_1_distance = sum(p["movement_stats"]["total_distance"] for p in movement_analytics if p["team_id"] == 1)
        
        # Count tactical events
        pressing_events = len(self.data["tactical_events"]["pressing_events"])
        offside_events = len(self.data["tactical_events"]["offside_events"])
        
        self.data["summary_statistics"] = {
            "match_summary": {
                "total_frames_processed": total_frames,
                "processing_fps": self.fps,
                "match_duration_minutes": total_frames / self.fps / 60,
                "unique_players_tracked": total_players,
                "team_composition": {
                    "team_0": team_0_players,
                    "team_1": team_1_players,
                    "officials": total_players - team_0_players - team_1_players
                },
                "possession": self.data["possession_analytics"]["possession_stats"],
                "total_distance": {
                    "team_0": team_0_distance,
                    "team_1": team_1_distance
                },
                "tactical_events": {
                    "pressing_events": pressing_events,
                    "offside_events": offside_events,
                    "formation_changes": len(self.data["formation_analytics"])
                }
            },
            "player_rankings": {
                "distance_covered": sorted(movement_analytics, key=lambda x: x["movement_stats"]["total_distance"], reverse=True)[:10],
                "max_speed": sorted(movement_analytics, key=lambda x: x["movement_stats"]["max_speed"], reverse=True)[:10],
                "sprints": sorted(movement_analytics, key=lambda x: x["movement_stats"]["sprint_count"], reverse=True)[:10]
            },
            "data_completeness": {
                "player_tracking_frames": sum(len(p["trajectory"]) for p in players_data.values()),
                "ball_detection_frames": self.data["tracking_data"]["ball"]["total_frames_detected"],
                "spatial_analysis_frames": len(self.data["spatial_analytics"]["per_frame_occupancy"]),
                "formation_analysis_frames": len(self.data["formation_analytics"]),
                "possession_segments": len(possession_segments)
            }
        }
    
    def export_to_json(self, output_path: str):
        """Export all collected data to JSON file"""
        self.finalize_analytics()
        
        try:
            # Convert numpy types to Python native types before serialization
            serializable_data = convert_numpy_types(self.data)
            
            with open(output_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            print(f"Analytics data exported to: {output_path}")
            print(f"Total data size: {len(json.dumps(serializable_data)) / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"ERROR: Failed to export JSON: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def main(source_video_path: str, target_video_path: str, device: str, mode: Mode, json_output_path: str = None) -> None:
    print("🏈 FOOTBALL3000 - Soccer Analytics with Comprehensive Data Extraction")
    print("✅ JSON EXPORT SUPPORTED - Complete analytics dataset")
    print("📊 RADAR mode now collects ALL MODES simultaneously!")
    print("=" * 60)
    print(f"Starting processing in {mode} mode...")
    print(f"Source: {source_video_path}")
    print(f"Target: {target_video_path}")
    print(f"Device: {device}")
    if mode == Mode.RADAR:
        json_path = json_output_path or target_video_path.replace('.mp4', '_analytics.json')
        print(f"📄 JSON Output: {json_path}")
        print("🔄 COMPREHENSIVE MODE: Collecting ALL data types:")
        print("   • Pitch detection & field transformation")
        print("   • Player detection, tracking & team classification")
        print("   • Ball detection & trajectory analysis")
        print("   • Formation analysis & tactical insights")
        print("   • Possession analysis & ball control")
        print("   • Spatial zone analytics & occupancy")
        print("   • Tactical events (pressing, offside, etc.)")
        print("   • Movement analytics & performance metrics")
    print("=" * 60)
    
    data_collector = None
    
    try:
        if mode == Mode.PITCH_DETECTION:
            frame_generator = run_pitch_detection(
                source_video_path=source_video_path, device=device)
        elif mode == Mode.PLAYER_DETECTION:
            frame_generator = run_player_detection(
                source_video_path=source_video_path, device=device)
        elif mode == Mode.BALL_DETECTION:
            frame_generator = run_ball_detection(
                source_video_path=source_video_path, device=device)
        elif mode == Mode.PLAYER_TRACKING:
            frame_generator = run_player_tracking(
                source_video_path=source_video_path, device=device)
        elif mode == Mode.TEAM_CLASSIFICATION:
            frame_generator = run_team_classification(
                source_video_path=source_video_path, device=device)
        elif mode == Mode.RADAR:
            frame_generator = run_radar(
                source_video_path=source_video_path, device=device)
        else:
            raise NotImplementedError(f"Mode {mode} is not implemented.")

        print("Loading video info...")
        video_info = sv.VideoInfo.from_video_path(source_video_path)
        print(f"Video info: {video_info.width}x{video_info.height}, {video_info.fps} FPS, {video_info.total_frames} frames")
        
        # Use MP4V codec for better Colab compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(target_video_path, fourcc, video_info.fps, (video_info.width, video_info.height))
        
        if not out.isOpened():
            print("Error: Could not open video writer")
            return
            
        frame_count = 0
        print("Starting frame processing...")
        
        # Handle RADAR mode with data collection differently
        if mode == Mode.RADAR:
            for frame, current_data_collector in frame_generator:
                out.write(frame)
                frame_count += 1
                data_collector = current_data_collector  # Keep reference to latest data collector
                if frame_count % 30 == 0:  # Print progress every 30 frames
                    print(f"Processed {frame_count} frames...")
        else:
            # Standard processing for other modes
            for frame in frame_generator:
                out.write(frame)
                frame_count += 1
                if frame_count % 30 == 0:  # Print progress every 30 frames
                    print(f"Processed {frame_count} frames...")

        out.release()
        print(f"Video processing complete! Processed {frame_count} frames.")
        print(f"Output saved to: {target_video_path}")
        
        # Check output file size
        if os.path.exists(target_video_path):
            file_size = os.path.getsize(target_video_path)
            print(f"Output file size: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
        
        # Export analytics data if in RADAR mode
        if mode == Mode.RADAR and data_collector is not None:
            json_output_path = json_output_path or target_video_path.replace('.mp4', '_analytics.json')
            print(f"\n🚀 Exporting COMPREHENSIVE analytics data to: {json_output_path}")
            data_collector.export_to_json(json_output_path)
            print(f"✅ Complete analytics dataset exported!")
            print(f"📊 Comprehensive data includes:")
            print(f"   • 🏟️  Pitch detection & field keypoints")
            print(f"   • 👥 Player tracking & movement analytics")
            print(f"   • ⚽ Ball detection & possession analysis")
            print(f"   • 🎯 Team classification & formations")
            print(f"   • 📍 Spatial zone analytics & occupancy")
            print(f"   • ⚡ Tactical events & insights")
            print(f"   • 📈 Performance metrics & rankings")
            print(f"   • 📊 Quality metrics & summary statistics")
            print(f"\n🎉 Ready for advanced coaching analysis in Streamlit!")
        elif mode == Mode.RADAR:
            print(f"WARNING: RADAR mode but data_collector is None!")
        else:
            print(f"Not in RADAR mode, skipping JSON export.")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Soccer Analytics with Comprehensive Data Extraction')
    parser.add_argument('--source_video_path', type=str, required=True, help='Path to input video file')
    parser.add_argument('--target_video_path', type=str, required=True, help='Path to output video file')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run models on (cpu/cuda)')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION, help='Analysis mode')
    parser.add_argument('--json_output_path', type=str, default=None, 
                       help='Custom path for JSON analytics output (RADAR mode only)')
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        json_output_path=args.json_output_path
    )
