# COMPREHENSIVE Soccer Analytics Data Extraction Specification
## Complete Data Extraction from Roboflow Sports System

## Overview
This specification defines **ALL POSSIBLE DATA EXTRACTION** from the roboflow/sports soccer analysis system, covering every frame, every detection, every calculation, and every insight that can be extracted.

## 1. RAW DETECTION DATA (Per Frame)

### 1.1 Player Detections (YOLO Output)
```python
PlayerDetection = {
    "frame_id": int,
    "timestamp": float,
    "detections": [
        {
            "detection_id": int,
            "class_id": int,              # 2=PLAYER, 1=GOALKEEPER, 3=REFEREE
            "class_name": str,            # "player", "goalkeeper", "referee"
            "confidence": float,          # YOLO confidence score
            "bbox": {
                "x1": float, "y1": float, "x2": float, "y2": float,
                "width": float, "height": float,
                "center_x": float, "center_y": float
            },
            "tracker_id": int,            # ByteTrack ID (if tracking enabled)
            "team_id": int,               # 0, 1, or 2 (referee) from TeamClassifier
            "team_confidence": float,     # Confidence of team classification
            "crop_embedding": [float],    # SiglipVision feature vector (512-dim)
            "position_video": {"x": float, "y": float},      # Bottom-center anchor
            "position_field": {"x": float, "y": float},      # Real-world coordinates (cm)
            "velocity": {"x": float, "y": float, "magnitude": float},  # Speed calculation
            "acceleration": {"x": float, "y": float, "magnitude": float}
        }
    ]
}
```

### 1.2 Ball Detections (Inference Slicer + Tracking)
```python
BallDetection = {
    "frame_id": int,
    "timestamp": float,
    "detections": [
        {
            "detection_id": int,
            "confidence": float,
            "bbox": {"x1": float, "y1": float, "x2": float, "y2": float},
            "position_video": {"x": float, "y": float},
            "position_field": {"x": float, "y": float},
            "slice_origin": {"x": int, "y": int},    # Which image slice detected it
            "tracking_buffer": [                     # Ball tracker history
                {"timestamp": float, "x": float, "y": float}
            ],
            "velocity": {"x": float, "y": float, "magnitude": float},
            "trajectory_prediction": {"x": float, "y": float},
            "possession_player_id": int,             # Closest player
            "possession_team_id": int,
            "distance_to_players": [
                {"player_id": int, "distance": float}
            ]
        }
    ]
}
```

### 1.3 Pitch Keypoint Detections (32 Points)
```python
PitchDetection = {
    "frame_id": int,
    "timestamp": float,
    "keypoints": [
        {
            "keypoint_id": int,           # 0-31 (32 total keypoints)
            "label": str,                 # "01", "02", ..., "32"
            "video_coords": {"x": float, "y": float},
            "field_coords": {"x": float, "y": float},  # Real-world position
            "confidence": float,
            "visible": bool,              # Is keypoint visible in frame
            "zone": str                   # "penalty_box", "center_circle", etc.
        }
    ],
    "homography_matrix": [[float]],       # 3x3 transformation matrix
    "transformation_quality": float,     # How good is the transformation
    "field_coverage": float,             # % of field visible
    "camera_angle": str,                 # "high", "low", "side"
    "field_orientation": float           # Rotation angle
}
```

## 2. TRACKING DATA (Temporal Consistency)

### 2.1 Player Tracking History
```python
PlayerTrackingData = {
    "player_id": int,
    "team_id": int,
    "role": str,                         # "player", "goalkeeper", "referee"
    "tracking_quality": float,           # ID consistency score
    "total_frames_tracked": int,
    "tracking_gaps": [                   # Frames where ID was lost
        {"start_frame": int, "end_frame": int, "duration": float}
    ],
    "trajectory": [
        {
            "frame_id": int,
            "timestamp": float,
            "position_video": {"x": float, "y": float},
            "position_field": {"x": float, "y": float},
            "velocity": {"x": float, "y": float, "magnitude": float},
            "acceleration": {"x": float, "y": float, "magnitude": float},
            "confidence": float,
            "bbox": {"x1": float, "y1": float, "x2": float, "y2": float}
        }
    ]
}
```

### 2.2 Ball Tracking History
```python
BallTrackingData = {
    "total_frames_detected": int,
    "detection_rate": float,             # % of frames with ball detected
    "tracking_gaps": [
        {"start_frame": int, "end_frame": int, "reason": str}
    ],
    "trajectory": [
        {
            "frame_id": int,
            "timestamp": float,
            "position_video": {"x": float, "y": float},
            "position_field": {"x": float, "y": float},
            "velocity": {"x": float, "y": float, "magnitude": float},
            "acceleration": {"x": float, "y": float, "magnitude": float},
            "height_estimate": float,     # Estimated ball height
            "confidence": float
        }
    ]
}
```

## 3. TEAM CLASSIFICATION DATA

### 3.1 Team Clustering Results
```python
TeamClassificationData = {
    "clustering_method": "SiglipVision + UMAP + KMeans",
    "feature_dimensions": 512,
    "umap_dimensions": 3,
    "cluster_centers": [
        {"team_id": 0, "center": [float], "color_signature": str},
        {"team_id": 1, "center": [float], "color_signature": str}
    ],
    "classification_confidence": float,
    "embedding_quality": float,
    "per_player_classification": [
        {
            "player_id": int,
            "team_assignments": [         # History of team assignments
                {
                    "frame_id": int,
                    "team_id": int,
                    "confidence": float,
                    "embedding": [float]  # Feature vector for this frame
                }
            ],
            "final_team_id": int,
            "assignment_stability": float
        }
    ]
}
```

## 4. SPATIAL ANALYTICS

### 4.1 Field Zone Analytics
```python
SpatialAnalytics = {
    "field_zones": {
        "defensive_third": {"x_min": 0, "x_max": 4000, "area": 28000000},  # cm²
        "middle_third": {"x_min": 4000, "x_max": 8000, "area": 28000000},
        "attacking_third": {"x_min": 8000, "x_max": 12000, "area": 28000000},
        "penalty_box_left": {"x_min": 0, "x_max": 2015, "y_min": 1450, "y_max": 5550},
        "penalty_box_right": {"x_min": 9985, "x_max": 12000, "y_min": 1450, "y_max": 5550},
        "goal_box_left": {"x_min": 0, "x_max": 550, "y_min": 2584, "y_max": 4416},
        "goal_box_right": {"x_min": 11450, "x_max": 12000, "y_min": 2584, "y_max": 4416},
        "center_circle": {"center": [6000, 3500], "radius": 915}
    },
    "per_frame_occupancy": [
        {
            "frame_id": int,
            "timestamp": float,
            "zone_occupancy": {
                "defensive_third": {"team_0": int, "team_1": int, "referees": int},
                "middle_third": {"team_0": int, "team_1": int, "referees": int},
                "attacking_third": {"team_0": int, "team_1": int, "referees": int},
                "penalty_box_left": {"team_0": int, "team_1": int},
                "penalty_box_right": {"team_0": int, "team_1": int},
                "center_circle": {"team_0": int, "team_1": int}
            }
        }
    ]
}
```

## 5. POSSESSION ANALYTICS

### 5.1 Ball Possession Detection
```python
PossessionAnalytics = {
    "possession_segments": [
        {
            "segment_id": int,
            "start_frame": int,
            "end_frame": int,
            "duration": float,            # Seconds
            "team_id": int,
            "controlling_player_id": int,
            "start_position": {"x": float, "y": float},
            "end_position": {"x": float, "y": float},
            "ball_touches": int,
            "players_involved": [int],
            "passes_attempted": int,
            "passes_completed": int,
            "end_event": str,             # "turnover", "shot", "foul", "out_of_bounds"
            "possession_value": float,    # Expected value of possession
            "field_progression": float    # Net yards gained
        }
    ],
    "possession_stats": {
        "team_0": {"total_time": float, "percentage": float, "segments": int},
        "team_1": {"total_time": float, "percentage": float, "segments": int}
    }
}
```

## 6. PLAYER MOVEMENT ANALYTICS

### 6.1 Individual Player Metrics
```python
PlayerMovementAnalytics = {
    "player_id": int,
    "team_id": int,
    "role": str,
    "movement_stats": {
        "total_distance": float,          # Meters
        "distance_per_minute": float,
        "max_speed": float,               # m/s
        "average_speed": float,
        "sprint_count": int,              # Runs > 5.5 m/s
        "sprint_distance": float,
        "high_intensity_runs": int,       # 4-5.5 m/s
        "acceleration_events": int,       # > 2 m/s²
        "deceleration_events": int        # < -2 m/s²
    },
    "zone_analysis": {
        "time_in_zones": {
            "defensive_third": float,
            "middle_third": float,
            "attacking_third": float,
            "own_penalty_box": float,
            "opponent_penalty_box": float
        },
        "distance_in_zones": {
            "defensive_third": float,
            "middle_third": float,
            "attacking_third": float
        }
    },
    "heat_map": {
        "grid_size": {"width": 120, "height": 80},  # 1.5m x 1.75m cells
        "values": [[float]]               # 80x120 grid of presence intensity
    },
    "position_timeline": [
        {
            "timestamp": float,
            "position": {"x": float, "y": float},
            "speed": float,
            "acceleration": float,
            "zone": str
        }
    ]
}
```

## 7. TEAM FORMATION ANALYTICS

### 7.1 Formation Analysis
```python
FormationAnalytics = {
    "per_frame_formations": [
        {
            "frame_id": int,
            "timestamp": float,
            "team_id": int,
            "formation_metrics": {
                "centroid": {"x": float, "y": float},
                "length": float,          # Team length in meters
                "width": float,           # Team width in meters
                "compactness": float,     # Average distance from centroid
                "stretch_index": float,   # Max distance between players
                "defensive_line": float,  # Y-coordinate of defensive line
                "offensive_line": float,  # Y-coordinate of most advanced players
                "formation_shape": str,   # Detected formation (e.g., "4-4-2")
                "shape_confidence": float
            },
            "player_positions": [
                {
                    "player_id": int,
                    "position": {"x": float, "y": float},
                    "formation_role": str,    # "CB", "LB", "CM", "LW", etc.
                    "role_confidence": float
                }
            ]
        }
    ],
    "formation_transitions": [
        {
            "start_frame": int,
            "end_frame": int,
            "from_formation": str,
            "to_formation": str,
            "transition_speed": float,    # Frames to complete transition
            "trigger_event": str          # "attack", "defense", "set_piece"
        }
    ]
}
```

## 8. TACTICAL EVENTS

### 8.1 Automatically Detected Events
```python
TacticalEvents = {
    "pressing_events": [
        {
            "frame_start": int,
            "frame_end": int,
            "team_id": int,
            "intensity": float,           # 0-1 scale
            "players_involved": [int],
            "trigger_zone": str,
            "success": bool,
            "ball_recovered": bool
        }
    ],
    "counter_attacks": [
        {
            "frame_start": int,
            "frame_end": int,
            "attacking_team": int,
            "players_involved": [int],
            "speed": float,               # Transition speed
            "field_progression": float,   # Meters advanced
            "outcome": str               # "shot", "turnover", "foul"
        }
    ],
    "offside_events": [
        {
            "frame_id": int,
            "player_id": int,
            "team_id": int,
            "offside_line": float,       # Y-coordinate
            "player_position": float,
            "offside_distance": float,   # How far offside
            "ball_played": bool
        }
    ]
}
```

## 9. COMPLETE JSON OUTPUT STRUCTURE

### 9.1 Master JSON Schema
```json
{
    "match_metadata": {
        "video_file": "string",
        "duration_seconds": 5400,
        "total_frames": 162000,
        "fps": 30.0,
        "resolution": {"width": 1920, "height": 1080},
        "processing_timestamp": "2024-01-01T12:00:00Z",
        "field_dimensions": {"width": 7000, "length": 12000, "units": "cm"}
    },
    "raw_detections": {
        "players": [...],             // PlayerDetection[] - every frame
        "ball": [...],                // BallDetection[] - every frame
        "pitch": [...]                // PitchDetection[] - every frame
    },
    "tracking_data": {
        "players": [...],             // PlayerTrackingData[] - per player
        "ball": {...}                 // BallTrackingData - single object
    },
    "team_classification": {...},    // TeamClassificationData
    "spatial_analytics": {...},      // SpatialAnalytics
    "possession_analytics": {...},   // PossessionAnalytics
    "movement_analytics": [...],     // PlayerMovementAnalytics[] - per player
    "formation_analytics": [...],    // FormationAnalytics[] - per team
    "tactical_events": {...},        // TacticalEvents
    "quality_metrics": {
        "player_detection_rate": 0.95,
        "ball_detection_rate": 0.78,
        "tracking_stability": 0.92,
        "keypoint_accuracy": 0.88,
        "team_classification_confidence": 0.94
    },
    "summary_statistics": {
        "match_summary": {
            "possession": {"team_0": 52.3, "team_1": 47.7},
            "total_passes": {"team_0": 432, "team_1": 389},
            "distance_covered": {"team_0": 98420, "team_1": 101230},
            "sprints": {"team_0": 156, "team_1": 142}
        },
        "player_rankings": {
            "distance_covered": [...],
            "max_speed": [...],
            "time_on_ball": [...]
        }
    }
}
```

## 10. IMPLEMENTATION EXTRACTION POINTS

### 10.1 Data Collection Points in Code
```python
# In run_radar() function - collect EVERYTHING:
data_collector = {
    "frame_data": [],
    "player_tracks": {},
    "ball_tracks": [],
    "formations": [],
    "events": []
}

# Per frame collection:
for frame in frame_generator:
    # Collect pitch keypoints
    pitch_result = pitch_detection_model(frame)
    keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
    
    # Collect player detections
    player_result = player_detection_model(frame)
    detections = sv.Detections.from_ultralytics(player_result)
    detections = tracker.update_with_detections(detections)
    
    # Collect team classifications
    crops = get_crops(frame, players)
    team_ids = team_classifier.predict(crops)
    
    # Transform coordinates
    transformer = ViewTransformer(...)
    field_positions = transformer.transform_points(...)
    
    # SAVE EVERYTHING TO data_collector
```

### 10.2 Final Export Function
```python
def export_complete_analytics(data_collector, output_path):
    """Export EVERYTHING to comprehensive JSON"""
    complete_data = {
        "match_metadata": extract_metadata(),
        "raw_detections": extract_raw_detections(data_collector),
        "tracking_data": calculate_tracking_analytics(data_collector),
        "spatial_analytics": calculate_spatial_analytics(data_collector),
        "possession_analytics": calculate_possession(data_collector),
        "movement_analytics": calculate_movement_stats(data_collector),
        "formation_analytics": analyze_formations(data_collector),
        "tactical_events": detect_tactical_events(data_collector),
        "quality_metrics": calculate_quality_metrics(data_collector),
        "summary_statistics": generate_summary(data_collector)
    }
    
    with open(output_path, 'w') as f:
        json.dump(complete_data, f, indent=2)
```

This specification covers **EVERY SINGLE DATA POINT** that can be extracted from the roboflow/sports system for your Streamlit dashboard! 