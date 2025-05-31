import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import math
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="⚽ Football Analytics Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional minimal CSS
st.markdown("""
<style>
    /* Clean minimal design */
    .main { padding: 1rem 2rem; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #f8fafc;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
        background-color: transparent;
        border-radius: 0.375rem;
        color: #64748b;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #0f172a;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Header styling */
    .analytics-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 2rem;
        border-radius: 0.75rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .analytics-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .analytics-subtitle {
        font-size: 1.1rem;
        opacity: 0.8;
        margin-bottom: 0;
    }
    
    /* Metric cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-delta {
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }
    .metric-positive { color: #059669; }
    .metric-negative { color: #dc2626; }
    
    /* Timeline controls */
    .timeline-controls {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    /* Insights panel */
    .insights-panel {
        background: linear-gradient(135deg, #fef3c7 0%, #fbbf24 100%);
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #f59e0b;
    }
    .insights-title {
        font-weight: 600;
        color: #92400e;
        margin-bottom: 1rem;
    }
    .insight-item {
        color: #78350f;
        margin-bottom: 0.5rem;
        padding-left: 1rem;
        position: relative;
    }
    .insight-item:before {
        content: "→";
        position: absolute;
        left: 0;
        color: #f59e0b;
        font-weight: bold;
    }
    
    /* Data quality indicators */
    .quality-excellent { color: #059669; font-weight: 600; }
    .quality-good { color: #0284c7; font-weight: 600; }
    .quality-fair { color: #ea580c; font-weight: 600; }
    .quality-poor { color: #dc2626; font-weight: 600; }
    
    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_match_data(uploaded_file) -> Optional[Dict]:
    """Load and validate comprehensive match data with enhanced validation"""
    try:
        content = uploaded_file.read()
        data = json.loads(content)
        
        # Enhanced data validation and cleaning
        required_sections = {
            'match_metadata': {},
            'raw_detections': {'players': [], 'ball': [], 'pitch': []},
            'tracking_data': {'players': {}, 'ball': {}},
            'spatial_analytics': {'field_zones': {}, 'per_frame_occupancy': []},
            'possession_analytics': {'possession_segments': [], 'possession_stats': {}},
            'movement_analytics': [],
            'formation_analytics': [],
            'tactical_events': {'pressing_events': [], 'offside_events': []},
            'quality_metrics': {},
            'summary_statistics': {}
        }
        
        # Fill missing sections with defaults
        missing_sections = []
        for section, default in required_sections.items():
            if section not in data:
                data[section] = default
                missing_sections.append(section.replace('_', ' ').title())
        
        # Clean and validate movement analytics
        speed_filtered_count = 0
        if 'movement_analytics' in data:
            cleaned_movement = []
            for player in data['movement_analytics']:
                if 'movement_stats' in player:
                    stats = player['movement_stats']
                    # Filter unrealistic speeds (human max speed ~12 m/s)
                    if stats.get('max_speed', 0) > 15:
                        speed_filtered_count += 1
                        stats['max_speed'] = min(stats.get('max_speed', 0), 12.0)
                    if stats.get('average_speed', 0) > 8:
                        stats['average_speed'] = min(stats.get('average_speed', 0), 8.0)
                    
                    # Ensure non-negative values
                    for key in ['total_distance', 'max_speed', 'average_speed']:
                        if key in stats and stats[key] < 0:
                            stats[key] = 0
                
                cleaned_movement.append(player)
            data['movement_analytics'] = cleaned_movement
        
        # Show summary messages only if there are issues
        if missing_sections and len(missing_sections) <= 3:
            st.info(f"ℹ️ Using default values for: {', '.join(missing_sections)}")
        elif len(missing_sections) > 3:
            st.info(f"ℹ️ Using default values for {len(missing_sections)} data sections")
            
        if speed_filtered_count > 0:
            st.info(f"ℹ️ Speed validation: {speed_filtered_count} players had unrealistic speeds (>15 m/s) automatically corrected")
        
        return data
        
    except json.JSONDecodeError as e:
        st.error(f"❌ Invalid JSON format: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

def safe_get(data: Dict, path: str, default=None):
    """Safely get nested dictionary value"""
    try:
        keys = path.split('.')
        value = data
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError, AttributeError):
        return default

def format_time(seconds: float) -> str:
    """Format seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def get_quality_class(value: float, thresholds: List[float]) -> str:
    """Get CSS class based on quality thresholds"""
    if value >= thresholds[0]:
        return "quality-excellent"
    elif value >= thresholds[1]:
        return "quality-good"
    else:
        return "quality-poor"

def create_main_header(data: Dict):
    """Create main header with match information"""
    metadata = safe_get(data, 'match_metadata', {})
    
    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">⚽ Football Analytics Pro</h1>
        <h2 style="margin: 0.5rem 0; font-size: 1.2rem; opacity: 0.9;">
            Match Analysis Dashboard
        </h2>
        <p style="margin: 0; opacity: 0.8;">
            📁 {metadata.get('video_file', 'Unknown match')} | 
            ⏱️ {format_time(metadata.get('duration_seconds', 0))} | 
            🎯 {metadata.get('total_frames', 0):,} frames
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_data_overview(data: Dict):
    """Create key performance metrics overview"""
    summary = safe_get(data, 'summary_statistics.match_summary', {})
    quality = safe_get(data, 'quality_metrics', {})
    possession = safe_get(data, 'summary_statistics.match_summary.possession', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        players_tracked = summary.get('unique_players_tracked', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{players_tracked}</div>
            <div class="metric-label">Players Tracked</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        ball_rate = quality.get('ball_detection_rate', 0) * 100
        quality_class = get_quality_class(ball_rate, [85, 70, 50])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {quality_class}">{ball_rate:.0f}%</div>
            <div class="metric-label">Ball Detection</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        team_0_poss = possession.get('team_0', {}).get('percentage', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{team_0_poss:.0f}%</div>
            <div class="metric-label">Team A Possession</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        pressing_events = len(safe_get(data, 'tactical_events.pressing_events', []))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{pressing_events}</div>
            <div class="metric-label">Pressing Events</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        total_distance = safe_get(data, 'summary_statistics.match_summary.total_distance', {})
        team_0_dist = total_distance.get('team_0', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{team_0_dist/1000:.1f}km</div>
            <div class="metric-label">Team A Distance</div>
        </div>
        """, unsafe_allow_html=True)

def create_timeline_controls(data: Dict):
    """Enhanced timeline controls with match events"""
    metadata = safe_get(data, 'match_metadata', {})
    max_frames = metadata.get('total_frames', 1000)
    fps = metadata.get('fps', 30.0)
    
    st.markdown('<div class="timeline-controls">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        frame_number = st.slider(
            "Select Match Moment",
            min_value=0,
            max_value=max_frames-1,
            value=st.session_state.get('selected_frame', 0),
            format="%d",
            help="Scrub through match timeline like video analysis software"
        )
        
        # Display comprehensive time info
        time_seconds = frame_number / fps
        progress_pct = (frame_number / max_frames) * 100
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.info(f"⏱️ **{format_time(time_seconds)}** / {format_time(max_frames/fps)}")
        with col_b:
            st.info(f"🎬 Frame **{frame_number:,}** / {max_frames:,}")
        with col_c:
            st.info(f"📊 Progress **{progress_pct:.1f}%**")
    
    with col2:
        st.markdown("**Quick Jump**")
        jump_options = {
            "🥅 Start": 0,
            "🥾 1st Half": max_frames // 4,
            "⏸️ Half Time": max_frames // 2,
            "🥾 2nd Half": 3 * max_frames // 4,
            "🏁 End": max_frames - 1
        }
        
        for label, target_frame in jump_options.items():
            if st.button(label, key=f"jump_{target_frame}", use_container_width=True):
                st.session_state['selected_frame'] = target_frame
                st.rerun()
    
    # Event markers on timeline
    events_data = []
    pressing_events = safe_get(data, 'tactical_events.pressing_events', [])
    offside_events = safe_get(data, 'tactical_events.offside_events', [])
    
    for event in pressing_events:
        events_data.append({
            'frame': event.get('frame_start', 0),
            'type': 'Pressing',
            'team': f"Team {'A' if event.get('team_id') == 0 else 'B'}",
            'color': '#ef4444' if event.get('team_id') == 0 else '#3b82f6'
        })
    
    for event in offside_events:
        events_data.append({
            'frame': event.get('frame_id', 0),
            'type': 'Offside',
            'team': f"Team {'A' if event.get('team_id') == 0 else 'B'}",
            'color': '#f59e0b'
        })
    
    if events_data:
        st.markdown("**🎯 Match Events Timeline**")
        events_df = pd.DataFrame(events_data)
        
        fig = go.Figure()
        
        # Add timeline bar
        fig.add_trace(go.Scatter(
            x=[0, max_frames],
            y=[0, 0],
            mode='lines',
            line=dict(color='#e5e7eb', width=10),
            showlegend=False
        ))
        
        # Add current position marker
        fig.add_trace(go.Scatter(
            x=[frame_number],
            y=[0],
            mode='markers',
            marker=dict(size=15, color='#dc2626', symbol='diamond'),
            name='Current Position',
            showlegend=False
        ))
        
        # Add event markers
        for event_type in events_df['type'].unique():
            type_events = events_df[events_df['type'] == event_type]
            fig.add_trace(go.Scatter(
                x=type_events['frame'],
                y=[0.1 if event_type == 'Pressing' else 0.2] * len(type_events),
                mode='markers',
                marker=dict(size=8, color=type_events['color'].iloc[0]),
                name=event_type,
                text=type_events['team'],
                hovertemplate=f"<b>{event_type}</b><br>Frame: %{{x}}<br>Team: %{{text}}<extra></extra>"
            ))
        
        fig.update_layout(
            height=150,
            xaxis=dict(title="Frame Number", range=[0, max_frames]),
            yaxis=dict(visible=False, range=[-0.1, 0.3]),
            showlegend=True,
            margin=dict(l=0, r=0, t=20, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    return frame_number

def create_fifa_tactical_view(data: Dict, frame_number: int, raw_players: list, raw_ball: list, movement_analytics: list):
    """FIFA-style tactical field view with proper coordinate mapping - PROFESSIONAL VERSION"""
    
    # Get frame data with proper bounds checking
    if not raw_players or frame_number >= len(raw_players):
        st.error(f"❌ **NO PLAYER DATA** - Frame {frame_number} not available (total frames: {len(raw_players) if raw_players else 0})")
        return
    
    frame_data = raw_players[frame_number]
    detections = frame_data.get('detections', [])
    
    # Get ball data with bounds checking
    ball_detection = None
    if raw_ball and frame_number < len(raw_ball):
        ball_frame = raw_ball[frame_number]
        ball_detections = ball_frame.get('detections', [])
        if ball_detections:
            ball_detection = ball_detections[0]
    
    # Create player analytics lookup
    player_analytics = {}
    for analytics in movement_analytics:
        player_id = analytics.get('player_id')
        if player_id:
            player_analytics[player_id] = analytics.get('movement_stats', {})
    
    # Professional field setup - FIFA style
    fig = go.Figure()
    
    # Field dimensions EXACTLY matching main.py CONFIG:
    # Field is 12000cm x 7000cm (from SoccerPitchConfiguration)
    # For display, we'll scale to a nice aspect ratio
    DISPLAY_LENGTH = 120  # Display length 
    DISPLAY_WIDTH = 70    # Display width (maintaining 12000:7000 ratio)
    
    # Field coordinate system from main.py:
    FIELD_LENGTH_CM = 12000  # cm
    FIELD_WIDTH_CM = 7000    # cm
    
    # Professional dark theme background
    fig.add_shape(
        type="rect", x0=-5, y0=-5, x1=DISPLAY_LENGTH+5, y1=DISPLAY_WIDTH+5,
        fillcolor="#0a0f1c", line=dict(color="#0a0f1c", width=0), opacity=1,
        layer="below"
    )
    
    # Main field
    fig.add_shape(
        type="rect", x0=0, y0=0, x1=DISPLAY_LENGTH, y1=DISPLAY_WIDTH,
        fillcolor="#1a4d3a", line=dict(color="#ffffff", width=3), opacity=1,
        layer="below"
    )
    
    # Professional field markings (scaled to display coordinates)
    field_markings = [
        # Center line
        {"type": "line", "x0": 60, "y0": 0, "x1": 60, "y1": DISPLAY_WIDTH, "line": {"color": "#ffffff", "width": 3}, "layer": "below"},
        
        # Center circle (radius = 915cm = 9.15m)
        {"type": "circle", "x0": 60-9.15, "y0": 35-9.15, "x1": 60+9.15, "y1": 35+9.15, 
         "line": {"color": "#ffffff", "width": 3}, "fillcolor": "rgba(0,0,0,0)", "layer": "below"},
        
        # Center spot
        {"type": "circle", "x0": 59, "y0": 34, "x1": 61, "y1": 36, 
         "fillcolor": "#ffffff", "line": {"color": "#ffffff", "width": 2}, "layer": "below"},
        
        # Left penalty area (2015cm x 4100cm)
        {"type": "rect", "x0": 0, "y0": 14.5, "x1": 20.15, "y1": 55.5, 
         "line": {"color": "#ffffff", "width": 3}, "fillcolor": "rgba(0,0,0,0)", "layer": "below"},
        
        # Right penalty area
        {"type": "rect", "x0": 99.85, "y0": 14.5, "x1": 120, "y1": 55.5, 
         "line": {"color": "#ffffff", "width": 3}, "fillcolor": "rgba(0,0,0,0)", "layer": "below"},
        
        # Left goal area (550cm x 1832cm)
        {"type": "rect", "x0": 0, "y0": 26.16, "x1": 5.5, "y1": 43.84, 
         "line": {"color": "#ffffff", "width": 3}, "fillcolor": "rgba(0,0,0,0)", "layer": "below"},
        
        # Right goal area
        {"type": "rect", "x0": 114.5, "y0": 26.16, "x1": 120, "y1": 43.84, 
         "line": {"color": "#ffffff", "width": 3}, "fillcolor": "rgba(0,0,0,0)", "layer": "below"},
        
        # Left goal
        {"type": "rect", "x0": -2, "y0": 31.34, "x1": 0, "y1": 38.66, 
         "fillcolor": "#ffffff", "line": {"color": "#ffffff", "width": 3}, "layer": "below"},
        
        # Right goal
        {"type": "rect", "x0": 120, "y0": 31.34, "x1": 122, "y1": 38.66, 
         "fillcolor": "#ffffff", "line": {"color": "#ffffff", "width": 3}, "layer": "below"},
        
        # Penalty spots (1100cm from goal line)
        {"type": "circle", "x0": 10.5, "y0": 34, "x1": 11.5, "y1": 36, 
         "fillcolor": "#ffffff", "line": {"color": "#ffffff", "width": 2}, "layer": "below"},
        {"type": "circle", "x0": 108.5, "y0": 34, "x1": 109.5, "y1": 36, 
         "fillcolor": "#ffffff", "line": {"color": "#ffffff", "width": 2}, "layer": "below"},
    ]
    
    for marking in field_markings:
        fig.add_shape(**marking)
    
    # Professional team colors
    team_styles = {
        0: {"color": "#dc2626", "name": "HOME", "edge": "#991b1b", "text": "🏠"},
        1: {"color": "#1e40af", "name": "AWAY", "edge": "#1e3a8a", "text": "✈️"},
        2: {"color": "#eab308", "name": "REF", "edge": "#a16207", "text": "👔"},
        3: {"color": "#eab308", "name": "REF", "edge": "#a16207", "text": "👔"}
    }
    
    # Process players with CORRECT coordinate conversion (based on main.py)
    players_added = 0
    for detection in detections:
        field_pos = detection.get('position_field', {})
        
        # Get raw field coordinates (stored in cm as per main.py)
        raw_x = field_pos.get('x', 0)
        raw_y = field_pos.get('y', 0)
        
        if raw_x <= 0 or raw_y <= 0:
            continue  # Skip invalid positions
            
        # Convert from main.py coordinate system (cm) to display coordinates:
        # main.py field: 0-12000cm x 0-7000cm
        # display field: 0-120 x 0-70
        x = (raw_x / FIELD_LENGTH_CM) * DISPLAY_LENGTH  # Scale from 0-12000cm to 0-120
        y = (raw_y / FIELD_WIDTH_CM) * DISPLAY_WIDTH    # Scale from 0-7000cm to 0-70
        
        # CRITICAL: Flip Y coordinate to match video orientation (not X!)
        y = DISPLAY_WIDTH - y
        
        # Ensure coordinates are within field bounds
        x = max(0, min(x, DISPLAY_LENGTH))
        y = max(0, min(y, DISPLAY_WIDTH))
        
        team_id = detection.get('team_id', -1)
        player_id = detection.get('tracker_id', '?')
        
        # Get velocity from detection
        velocity_data = detection.get('velocity', {})
        speed = velocity_data.get('magnitude', 0)
        
        # Get player stats from analytics
        analytics = player_analytics.get(player_id, {})
        distance = analytics.get('total_distance', 0)
        max_speed = analytics.get('max_speed', 0)
        sprints = analytics.get('sprint_count', 0)
        
        team_data = team_styles.get(team_id, {"color": "#64748b", "name": "UNK", "edge": "#475569", "text": "❓"})
        
        # CONSISTENT professional player styling
        player_size = 28  # Fixed size for all players
        border_width = 3
        opacity = 1.0
        
        # Speed indicator through border thickness only
        if speed > 7:  # Sprint
            border_width = 5
        elif speed > 4:  # Run
            border_width = 4
        else:  # Walk/Jog
            border_width = 3
        
        # Professional hover information
        hover_data = f"""
<b style='font-size:16px; color:{team_data['color']}'>{team_data['text']} {team_data['name']} #{player_id}</b><br>
<br><b>📍 POSITION:</b> ({x:.1f}, {y:.1f})
<br><b>🗺️ FIELD POS:</b> ({raw_x:.0f}cm, {raw_y:.0f}cm)
<br><b>🏃 SPEED:</b> {speed:.1f} m/s
<br><b>📏 DISTANCE:</b> {distance:.0f}m
<br><b>⚡ TOP SPEED:</b> {max_speed:.1f} m/s
<br><b>🚀 SPRINTS:</b> {sprints}
<br><b>💪 STATUS:</b> {'🔥 SPRINTING' if speed > 7 else '💨 RUNNING' if speed > 4 else '🚶 JOGGING'}
<extra></extra>"""
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(
                size=player_size,
                color=team_data['color'],
                line=dict(width=border_width, color=team_data['edge']),
                opacity=opacity,
                symbol='circle'
            ),
            text=[str(player_id)],
            textfont=dict(color='white', size=12, family="Arial Black"),
            textposition="middle center",
            name=team_data['name'],
            showlegend=team_id in [0, 1] and players_added < 2,  # Show legend once per team
            hovertemplate=hover_data,
            hoverlabel=dict(
                bgcolor="rgba(15, 23, 42, 0.95)",
                bordercolor=team_data['color'],
                font=dict(color="white", size=12)
            )
        ))
        players_added += 1
    
    # Professional ball rendering
    if ball_detection:
        ball_pos = ball_detection.get('position_field', {})
        raw_ball_x = ball_pos.get('x', 0)
        raw_ball_y = ball_pos.get('y', 0)
        
        if raw_ball_x > 0 and raw_ball_y > 0:
            # Convert ball coordinates using same transformation as players
            ball_x = (raw_ball_x / FIELD_LENGTH_CM) * DISPLAY_LENGTH
            ball_y = (raw_ball_y / FIELD_WIDTH_CM) * DISPLAY_WIDTH
            
            # CRITICAL: Flip Y coordinate to match video orientation (same as players)
            ball_y = DISPLAY_WIDTH - ball_y
            
            # Ensure ball is within field bounds
            ball_x = max(0, min(ball_x, DISPLAY_LENGTH))
            ball_y = max(0, min(ball_y, DISPLAY_WIDTH))
            
            ball_speed = ball_detection.get('velocity', {}).get('magnitude', 0)
            
            fig.add_trace(go.Scatter(
                x=[ball_x], y=[ball_y],
                mode='markers',
                marker=dict(
                    size=18, 
                    color='#ffffff', 
                    symbol='circle',
                    line=dict(width=3, color='#000000')
                ),
                name='⚽ BALL',
                showlegend=True,
                hovertemplate=f"<b style='font-size:16px'>⚽ BALL</b><br><b>SPEED:</b> {ball_speed:.1f} m/s<br><b>POSITION:</b> ({ball_x:.1f}, {ball_y:.1f})<br><b>FIELD POS:</b> ({raw_ball_x:.0f}cm, {raw_ball_y:.0f}cm)<extra></extra>"
            ))
    
    # Professional layout
    fig.update_layout(
        title=dict(
            text=f"<b style='color:#ffffff; font-size:24px'>⚽ TACTICAL VIEW</b><br><span style='color:#94a3b8; font-size:14px'>Live Match Analysis - {format_time(frame_number/30)}</span>",
            x=0.5, 
            font=dict(family="Arial", color="#ffffff")
        ),
        xaxis=dict(
            range=[-2, DISPLAY_LENGTH+2], 
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            fixedrange=True,
            showline=False
        ),
        yaxis=dict(
            range=[-2, DISPLAY_WIDTH+2], 
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            fixedrange=True,
            scaleanchor="x", 
            scaleratio=1,
            showline=False
        ),
        plot_bgcolor='#0a0f1c', 
        paper_bgcolor='#0a0f1c', 
        height=600,
        showlegend=True,
        legend=dict(
            x=1.02, 
            y=1, 
            bgcolor="rgba(15, 23, 42, 0.95)", 
            bordercolor="#374151", 
            font=dict(color="white", size=12)
        ),
        margin=dict(l=10, r=150, t=80, b=10)
    )
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.markdown("### 🎮 **LIVE INTEL**")
        
        # Show coordinate system info
        st.markdown("#### **🔍 COORD SYSTEM**")
        st.write(f"Field: {FIELD_LENGTH_CM}cm × {FIELD_WIDTH_CM}cm")
        st.write(f"Display: {DISPLAY_LENGTH} × {DISPLAY_WIDTH}")
        st.write(f"Players found: {players_added}")
        st.write(f"Ball detected: {'Yes' if ball_detection else 'No'}")
        
        # Live team stats
        team_counts = {0: 0, 1: 0}
        team_speeds = {0: [], 1: []}
        
        for detection in detections:
            team_id = detection.get('team_id', -1)
            if team_id in [0, 1]:
                team_counts[team_id] += 1
                velocity_data = detection.get('velocity', {})
                speed = velocity_data.get('magnitude', 0)
                team_speeds[team_id].append(speed)
        
        st.markdown("#### **🏠 HOME TEAM**")
        st.metric("Players", team_counts[0])
        if team_speeds[0]:
            avg_speed = sum(team_speeds[0]) / len(team_speeds[0])
            st.metric("Avg Speed", f"{avg_speed:.1f} m/s")
        
        st.markdown("#### **✈️ AWAY TEAM**")
        st.metric("Players", team_counts[1])
        if team_speeds[1]:
            avg_speed = sum(team_speeds[1]) / len(team_speeds[1])
            st.metric("Avg Speed", f"{avg_speed:.1f} m/s")
        
        # Tactical advantage
        if team_counts[0] > team_counts[1]:
            st.success(f"🏠 **NUMERICAL ADVANTAGE**\nHome +{team_counts[0] - team_counts[1]}")
        elif team_counts[1] > team_counts[0]:
            st.error(f"✈️ **NUMERICAL ADVANTAGE**\nAway +{team_counts[1] - team_counts[0]}")
        else:
            st.info("⚖️ **BALANCED**\nEqual numbers")
        
        # Quick stats
        st.markdown("#### **📊 QUICK STATS**")
        total_players = sum(team_counts.values())
        all_speeds = team_speeds[0] + team_speeds[1]
        if all_speeds:
            max_speed = max(all_speeds)
            avg_speed = sum(all_speeds) / len(all_speeds)
            st.metric("Players", total_players)
            st.metric("Max Speed", f"{max_speed:.1f} m/s")
            st.metric("Avg Speed", f"{avg_speed:.1f} m/s")

def create_fifa_radar_view(data: Dict, frame_number: int, raw_players: list, movement_analytics: list, spatial_data: list):
    """FIFA Manager RADAR view with heat maps and movement patterns"""
    
    st.markdown("### 📡 **RADAR MODE** - Heat Maps & Movement Analysis")
    
    if not spatial_data or frame_number >= len(spatial_data):
        st.warning("⚠️ No spatial data for RADAR view")
        return
    
    # Get current occupancy data
    current_occupancy = spatial_data[frame_number].get('zone_occupancy', {})
    
    # Create RADAR visualization
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Zone control heat map
        zones = ['defensive_third', 'middle_third', 'attacking_third']
        zone_names = ['Defense', 'Midfield', 'Attack']
        
        heat_data = []
        for zone, name in zip(zones, zone_names):
            if zone in current_occupancy:
                team_a = current_occupancy[zone].get('team_0', 0)
                team_b = current_occupancy[zone].get('team_1', 0)
                control = team_a - team_b  # Positive = Team A control
                
                heat_data.append({
                    'Zone': name,
                    'Control': control,
                    'Intensity': abs(control),
                    'Team A': team_a,
                    'Team B': team_b
                })
        
        if heat_data:
            df = pd.DataFrame(heat_data)
            
            fig = px.bar(df, x='Zone', y=['Team A', 'Team B'], 
                        title="📡 ZONE CONTROL RADAR",
                        color_discrete_map={'Team A': '#2563eb', 'Team B': '#dc2626'})
            fig.update_layout(height=400, plot_bgcolor='#0f172a', paper_bgcolor='#0f172a')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### **🎯 ZONE ANALYSIS**")
        for row in heat_data:
            zone = row['Zone']
            control = row['Control']
            if control > 0:
                st.success(f"🔵 **{zone}**\nHome +{control}")
            elif control < 0:
                st.error(f"🔴 **{zone}**\nAway +{abs(control)}")
            else:
                st.info(f"⚖️ **{zone}**\nBalanced")

def create_fifa_formation_view(data: Dict, frame_number: int, formation_data: list, raw_players: list):
    """FIFA Manager formation analysis"""
    
    st.markdown("### 🎯 **FORMATION ANALYSIS** - Tactical Shape")
    
    if not formation_data:
        st.warning("⚠️ No formation data available")
        return
    
    # Find current formation data
    current_formations = [f for f in formation_data if f.get('frame_id', 0) <= frame_number]
    if not current_formations:
        st.warning("⚠️ No formation data for this time period")
        return
    
    col1, col2 = st.columns(2)
    
    # Analyze formations for both teams
    for team_id, col in [(0, col1), (1, col2)]:
        team_formations = [f for f in current_formations if f.get('team_id') == team_id]
        
        if team_formations:
            latest = team_formations[-1]
            metrics = latest.get('formation_metrics', {})
            
            team_name = "🔵 HOME TEAM" if team_id == 0 else "🔴 AWAY TEAM"
            
            with col:
                st.markdown(f"#### **{team_name}**")
                
                formation = metrics.get('formation_shape', 'Unknown')
                length = metrics.get('length', 0)
                width = metrics.get('width', 0)
                compactness = metrics.get('compactness', 0)
                
                st.metric("Formation", formation)
                st.metric("Length", f"{length:.1f}m")
                st.metric("Width", f"{width:.1f}m")
                st.metric("Compactness", f"{compactness:.1f}m")
                
                # Tactical assessment
                if compactness < 15:
                    st.success("🔥 **COMPACT** - Good defensive shape")
                elif compactness < 25:
                    st.info("⚡ **BALANCED** - Flexible formation")
                else:
                    st.warning("🔄 **SPREAD** - Vulnerable to counters")

def create_fifa_analytics_dashboard(data: Dict, frame_number: int, movement_analytics: list):
    """FIFA Manager analytics dashboard"""
    
    st.markdown("### 📊 **MANAGER DASHBOARD** - Key Performance Indicators")
    
    if not movement_analytics:
        st.warning("⚠️ No analytics data available")
        return
    
    # Calculate team performance metrics
    team_stats = {0: {'players': [], 'total_distance': 0, 'avg_speed': 0, 'sprints': 0},
                  1: {'players': [], 'total_distance': 0, 'avg_speed': 0, 'sprints': 0}}
    
    for player in movement_analytics:
        team_id = player.get('team_id', -1)
        if team_id in [0, 1]:
            stats = player.get('movement_stats', {})
            team_stats[team_id]['players'].append(player)
            team_stats[team_id]['total_distance'] += stats.get('total_distance', 0)
            team_stats[team_id]['avg_speed'] += stats.get('average_speed', 0)
            team_stats[team_id]['sprints'] += stats.get('sprint_count', 0)
    
    # Normalize averages
    for team_id in [0, 1]:
        count = len(team_stats[team_id]['players'])
        if count > 0:
            team_stats[team_id]['avg_speed'] /= count
    
    # Display team comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### **🔵 HOME TEAM KPIs**")
        st.metric("Total Distance", f"{team_stats[0]['total_distance']/1000:.1f} km")
        st.metric("Avg Speed", f"{team_stats[0]['avg_speed']:.1f} m/s")
        st.metric("Sprint Events", team_stats[0]['sprints'])
    
    with col2:
        st.markdown("#### **🔴 AWAY TEAM KPIs**")
        st.metric("Total Distance", f"{team_stats[1]['total_distance']/1000:.1f} km")
        st.metric("Avg Speed", f"{team_stats[1]['avg_speed']:.1f} m/s")
        st.metric("Sprint Events", team_stats[1]['sprints'])
    
    with col3:
        st.markdown("#### **⚡ PERFORMANCE EDGE**")
        
        # Calculate advantages
        dist_diff = team_stats[0]['total_distance'] - team_stats[1]['total_distance']
        speed_diff = team_stats[0]['avg_speed'] - team_stats[1]['avg_speed']
        sprint_diff = team_stats[0]['sprints'] - team_stats[1]['sprints']
        
        if dist_diff > 500:
            st.success("🔵 **WORK RATE**\nHome team ahead")
        elif dist_diff < -500:
            st.error("🔴 **WORK RATE**\nAway team ahead")
        else:
            st.info("⚖️ **EVEN**\nMatched work rate")
            
        if speed_diff > 0.3:
            st.success("🔵 **INTENSITY**\nHome faster")
        elif speed_diff < -0.3:
            st.error("🔴 **INTENSITY**\nAway faster")
        else:
            st.info("⚖️ **MATCHED**\nEqual intensity")
    
    # Top performers
    st.markdown("#### **🏆 TOP PERFORMERS THIS FRAME**")
    
    all_players = team_stats[0]['players'] + team_stats[1]['players']
    if all_players:
        # Sort by total distance
        top_distance = sorted(all_players, key=lambda x: x.get('movement_stats', {}).get('total_distance', 0), reverse=True)[:3]
        
        cols = st.columns(3)
        for i, player in enumerate(top_distance):
            with cols[i]:
                team_name = "🔵 HOME" if player.get('team_id') == 0 else "🔴 AWAY"
                distance = player.get('movement_stats', {}).get('total_distance', 0)
                st.metric(f"{team_name} #{player.get('player_id', '?')}", f"{distance:.0f}m")

def create_comprehensive_performance_analysis(data: Dict):
    """Create FIFA Manager-style performance analysis with real-time insights"""
    st.markdown('<div class="data-section">', unsafe_allow_html=True)
    st.subheader("📊 FIFA Manager Performance Center")
    
    movement_data = safe_get(data, 'movement_analytics', [])
    
    if not movement_data:
        st.warning("⚠️ No player performance data available")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # FIFA Manager style tabs
    perf_tab1, perf_tab2, perf_tab3, perf_tab4, perf_tab5 = st.tabs([
        "⚽ **PLAYER RATINGS**", 
        "🔥 **HEAT MAPS**", 
        "📈 **MATCH TRENDS**", 
        "🏆 **LEADERBOARDS**",
        "⚡ **QUICK INSIGHTS**"
    ])
    
    with perf_tab1:
        create_fifa_player_ratings(movement_data)
    
    with perf_tab2:
        create_fifa_heat_maps(data, movement_data)
    
    with perf_tab3:
        create_fifa_match_trends(movement_data)
    
    with perf_tab4:
        create_fifa_leaderboards(movement_data)
    
    with perf_tab5:
        create_fifa_quick_insights(movement_data)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_fifa_player_ratings(movement_data: list):
    """FIFA-style player ratings and performance cards"""
    
    st.markdown("### ⚽ **PLAYER PERFORMANCE RATINGS**")
    st.markdown("*Real-time performance analysis with FIFA-style ratings*")
    
    # Create performance ratings
    performance_cards = []
    for player in movement_data:
        if player.get('team_id', -1) in [0, 1]:
            stats = player.get('movement_stats', {})
            
            # Calculate FIFA-style rating (0-100)
            distance_score = min((stats.get('total_distance', 0) / 12000) * 30, 30)  # Max 30 points
            speed_score = min((stats.get('max_speed', 0) / 12) * 25, 25)  # Max 25 points  
            work_rate_score = min((stats.get('sprint_count', 0) / 20) * 25, 25)  # Max 25 points
            consistency_score = min((stats.get('average_speed', 0) / 6) * 20, 20)  # Max 20 points
            
            overall_rating = int(distance_score + speed_score + work_rate_score + consistency_score)
            
            # Determine rating class
            if overall_rating >= 85:
                rating_class = "🔥 WORLD CLASS"
                rating_color = "#16a34a"
            elif overall_rating >= 75:
                rating_class = "⭐ EXCELLENT"
                rating_color = "#2563eb"
            elif overall_rating >= 65:
                rating_class = "💪 GOOD"
                rating_color = "#ea580c"
            elif overall_rating >= 50:
                rating_class = "📊 AVERAGE"
                rating_color = "#eab308"
            else:
                rating_class = "⚠️ POOR"
                rating_color = "#dc2626"
            
            performance_cards.append({
                'Player': f"#{player['player_id']}",
                'Team': "🔵 HOME" if player['team_id'] == 0 else "🔴 AWAY",
                'Rating': overall_rating,
                'Class': rating_class,
                'Color': rating_color,
                'Distance': f"{stats.get('total_distance', 0):.0f}m",
                'Speed': f"{stats.get('max_speed', 0):.1f} m/s",
                'Sprints': stats.get('sprint_count', 0),
                'Work Rate': f"{stats.get('average_speed', 0):.1f} m/s"
            })
    
    # Sort by rating
    performance_cards.sort(key=lambda x: x['Rating'], reverse=True)
    
    # Display top performers in cards
    st.markdown("#### 🏆 **TOP PERFORMERS**")
    
    if len(performance_cards) >= 6:
        top_cols = st.columns(3)
        for i, card in enumerate(performance_cards[:3]):
            with top_cols[i]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {card['Color']}20, {card['Color']}40); 
                            border: 2px solid {card['Color']}; border-radius: 10px; padding: 15px; text-align: center;">
                    <h3 style="color: {card['Color']}; margin: 0;">{card['Rating']}</h3>
                    <p style="margin: 5px 0; font-weight: bold;">{card['Team']} {card['Player']}</p>
                    <p style="margin: 0; font-size: 12px;">{card['Class']}</p>
                    <hr style="margin: 10px 0; border-color: {card['Color']};">
                    <p style="margin: 2px; font-size: 11px;">🏃 {card['Distance']} | ⚡ {card['Speed']}</p>
                    <p style="margin: 2px; font-size: 11px;">🚀 {card['Sprints']} sprints | 📊 {card['Work Rate']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Full ratings table
    st.markdown("#### 📋 **COMPLETE SQUAD RATINGS**")
    ratings_df = pd.DataFrame([
        {
            'Player': card['Player'],
            'Team': card['Team'], 
            'Rating': card['Rating'],
            'Performance': card['Class'],
            'Distance': card['Distance'],
            'Max Speed': card['Speed'],
            'Sprints': card['Sprints']
        } for card in performance_cards
    ])
    
    st.dataframe(
        ratings_df,
        use_container_width=True,
        column_config={
            'Rating': st.column_config.ProgressColumn(
                'Overall Rating',
                help='FIFA-style performance rating (0-100)',
                min_value=0,
                max_value=100,
                format='%d'
            )
        }
    )

def create_fifa_heat_maps(data: Dict, movement_data: list):
    """FIFA Manager heat maps and movement analysis with field visualization"""
    
    st.markdown("### 🔥 **PLAYER HEAT MAPS & MOVEMENT PATTERNS**")
    
    # Get raw player data for all frames
    raw_players = safe_get(data, 'raw_detections.players', [])
    if not raw_players:
        st.warning("⚠️ No player position data available for heat maps")
        return
    
    # Field dimensions - same as tactical view
    DISPLAY_LENGTH = 120
    DISPLAY_WIDTH = 70
    FIELD_LENGTH_CM = 12000
    FIELD_WIDTH_CM = 7000
    
    # Heat map mode selection
    st.markdown("#### **🎯 HEAT MAP MODE SELECTION**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        heat_mode = st.radio(
            "Select Heat Map Type:",
            ["🔥 Individual Player", "👥 Team Heat Maps", "⚡ Movement Trails"],
            key="heat_mode"
        )
    
    with col2:
        if heat_mode == "🔥 Individual Player":
            # Player selection for individual analysis
            player_options = [f"#{player['player_id']} ({'HOME' if player['team_id'] == 0 else 'AWAY'})" 
                             for player in movement_data if player.get('team_id', -1) in [0, 1]]
            
            if player_options:
                selected_player = st.selectbox("🎯 **Select Player:**", player_options)
                player_id = selected_player.split('#')[1].split(' ')[0]
            else:
                st.warning("No valid players found")
                return
        else:
            player_id = None
    
    with col3:
        frame_range = st.slider(
            "Frame Range for Analysis:",
            min_value=0,
            max_value=len(raw_players)-1,
            value=(0, min(len(raw_players)-1, 1000)),
            help="Select frame range to analyze (smaller ranges load faster)"
        )
    
    # Process player positions for heat map
    def process_player_positions():
        """Extract and convert player positions from all frames"""
        positions_data = {}
        
        start_frame, end_frame = frame_range
        for frame_idx in range(start_frame, min(end_frame + 1, len(raw_players))):
            frame_data = raw_players[frame_idx]
            detections = frame_data.get('detections', [])
            
            for detection in detections:
                field_pos = detection.get('position_field', {})
                raw_x = field_pos.get('x', 0)
                raw_y = field_pos.get('y', 0)
                
                if raw_x <= 0 or raw_y <= 0:
                    continue
                
                # Convert coordinates - same as tactical view
                x = (raw_x / FIELD_LENGTH_CM) * DISPLAY_LENGTH
                y = (raw_y / FIELD_WIDTH_CM) * DISPLAY_WIDTH
                y = DISPLAY_WIDTH - y  # Flip Y coordinate
                
                # Ensure coordinates are within bounds
                x = max(0, min(x, DISPLAY_LENGTH))
                y = max(0, min(y, DISPLAY_WIDTH))
                
                player_id_detect = detection.get('tracker_id', '?')
                team_id = detection.get('team_id', -1)
                
                if team_id in [0, 1]:  # Only valid teams
                    if player_id_detect not in positions_data:
                        positions_data[player_id_detect] = {
                            'x': [], 'y': [], 'frames': [], 'team_id': team_id
                        }
                    
                    positions_data[player_id_detect]['x'].append(x)
                    positions_data[player_id_detect]['y'].append(y)
                    positions_data[player_id_detect]['frames'].append(frame_idx)
        
        return positions_data
    
    # Create field base with markings - same as tactical view
    def create_field_base():
        """Create the same field layout as tactical view"""
        fig = go.Figure()
        
        # Background
        fig.add_shape(
            type="rect", x0=-5, y0=-5, x1=DISPLAY_LENGTH+5, y1=DISPLAY_WIDTH+5,
            fillcolor="#0a0f1c", line=dict(color="#0a0f1c", width=0), opacity=1,
            layer="below"
        )
        
        # Main field
        fig.add_shape(
            type="rect", x0=0, y0=0, x1=DISPLAY_LENGTH, y1=DISPLAY_WIDTH,
            fillcolor="#1a4d3a", line=dict(color="#ffffff", width=2), opacity=1,
            layer="below"
        )
        
        # Field markings
        field_markings = [
            # Center line
            {"type": "line", "x0": 60, "y0": 0, "x1": 60, "y1": DISPLAY_WIDTH, 
             "line": {"color": "#ffffff", "width": 2}, "layer": "below"},
            
            # Center circle
            {"type": "circle", "x0": 60-9.15, "y0": 35-9.15, "x1": 60+9.15, "y1": 35+9.15, 
             "line": {"color": "#ffffff", "width": 2}, "fillcolor": "rgba(0,0,0,0)", "layer": "below"},
            
            # Left penalty area
            {"type": "rect", "x0": 0, "y0": 14.5, "x1": 20.15, "y1": 55.5, 
             "line": {"color": "#ffffff", "width": 2}, "fillcolor": "rgba(0,0,0,0)", "layer": "below"},
            
            # Right penalty area
            {"type": "rect", "x0": 99.85, "y0": 14.5, "x1": 120, "y1": 55.5, 
             "line": {"color": "#ffffff", "width": 2}, "fillcolor": "rgba(0,0,0,0)", "layer": "below"},
            
            # Left goal area
            {"type": "rect", "x0": 0, "y0": 26.16, "x1": 5.5, "y1": 43.84, 
             "line": {"color": "#ffffff", "width": 2}, "fillcolor": "rgba(0,0,0,0)", "layer": "below"},
            
            # Right goal area
            {"type": "rect", "x0": 114.5, "y0": 26.16, "x1": 120, "y1": 43.84, 
             "line": {"color": "#ffffff", "width": 2}, "fillcolor": "rgba(0,0,0,0)", "layer": "below"},
            
            # Goals
            {"type": "rect", "x0": -2, "y0": 31.34, "x1": 0, "y1": 38.66, 
             "fillcolor": "#ffffff", "line": {"color": "#ffffff", "width": 2}, "layer": "below"},
            {"type": "rect", "x0": 120, "y0": 31.34, "x1": 122, "y1": 38.66, 
             "fillcolor": "#ffffff", "line": {"color": "#ffffff", "width": 2}, "layer": "below"},
        ]
        
        for marking in field_markings:
            fig.add_shape(**marking)
        
        return fig
    
    with st.spinner("🔄 Processing player positions..."):
        positions_data = process_player_positions()
    
    if not positions_data:
        st.error("❌ No valid position data found in selected frame range")
        return
    
    # Create visualizations based on mode
    if heat_mode == "🔥 Individual Player":
        if player_id in positions_data:
            player_data = positions_data[player_id]
            team_color = "#dc2626" if player_data['team_id'] == 0 else "#1e40af"
            team_name = "HOME" if player_data['team_id'] == 0 else "AWAY"
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Create individual player heat map
                fig = create_field_base()
                
                # Add heat map using histogram2d
                if len(player_data['x']) > 1:
                    # Create density heatmap
                    fig.add_trace(go.Histogram2d(
                        x=player_data['x'],
                        y=player_data['y'],
                        nbinsx=30,
                        nbinsy=20,
                        colorscale=[
                            [0, 'rgba(255,255,255,0)'],
                            [0.3, f'rgba({team_color[1:3]},{team_color[3:5]},{team_color[5:7]},0.3)'],
                            [0.6, f'rgba({team_color[1:3]},{team_color[3:5]},{team_color[5:7]},0.6)'],
                            [1.0, f'rgba({team_color[1:3]},{team_color[3:5]},{team_color[5:7]},0.9)']
                        ],
                        showscale=True,
                        colorbar=dict(title="Activity<br>Density", titleside="right"),
                        hovertemplate="<b>Position Density</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Count: %{z}<extra></extra>"
                    ))
                    
                    # Add movement trail
                    fig.add_trace(go.Scatter(
                        x=player_data['x'][::5],  # Sample every 5th point for performance
                        y=player_data['y'][::5],
                        mode='lines+markers',
                        line=dict(color=team_color, width=3, dash='dot'),
                        marker=dict(size=4, color=team_color, opacity=0.7),
                        name=f'Movement Trail',
                        showlegend=True,
                        hovertemplate="<b>Movement Point</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Frame: %{text}<extra></extra>",
                        text=player_data['frames'][::5]
                    ))
                
                fig.update_layout(
                    title=f"🔥 Heat Map: {team_name} Player #{player_id}",
                    xaxis=dict(range=[-2, DISPLAY_LENGTH+2], showgrid=False, zeroline=False, 
                              showticklabels=False, fixedrange=True),
                    yaxis=dict(range=[-2, DISPLAY_WIDTH+2], showgrid=False, zeroline=False, 
                              showticklabels=False, fixedrange=True, scaleanchor="x", scaleratio=1),
                    plot_bgcolor='#0a0f1c',
                    paper_bgcolor='#0a0f1c',
                    height=600,
                    font=dict(color='white'),
                    showlegend=True,
                    legend=dict(x=1.02, y=1, bgcolor="rgba(15, 23, 42, 0.95)", 
                               bordercolor="#374151", font=dict(color="white"))
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            with col2:
                st.markdown("#### **📊 MOVEMENT STATS**")
                
                total_positions = len(player_data['x'])
                frames_analyzed = len(set(player_data['frames']))
                
                st.metric("Total Positions", f"{total_positions:,}")
                st.metric("Frames Analyzed", f"{frames_analyzed:,}")
                
                if len(player_data['x']) > 1:
                    # Calculate movement statistics
                    distances = []
                    for i in range(1, len(player_data['x'])):
                        dx = player_data['x'][i] - player_data['x'][i-1]
                        dy = player_data['y'][i] - player_data['y'][i-1]
                        dist = (dx**2 + dy**2)**0.5
                        distances.append(dist)
                    
                    if distances:
                        total_distance = sum(distances)
                        avg_distance_per_frame = total_distance / len(distances)
                        
                        st.metric("Total Movement", f"{total_distance:.1f} units")
                        st.metric("Avg/Frame", f"{avg_distance_per_frame:.2f} units")
                
                # Most active areas
                st.markdown("#### **🎯 ACTIVITY ZONES**")
                
                # Calculate zone activity
                left_third = len([x for x in player_data['x'] if x < 40])
                middle_third = len([x for x in player_data['x'] if 40 <= x < 80])
                right_third = len([x for x in player_data['x'] if x >= 80])
                
                total_zone = left_third + middle_third + right_third
                
                if total_zone > 0:
                    st.metric("Left Third", f"{left_third} ({left_third/total_zone*100:.1f}%)")
                    st.metric("Middle Third", f"{middle_third} ({middle_third/total_zone*100:.1f}%)")
                    st.metric("Right Third", f"{right_third} ({right_third/total_zone*100:.1f}%)")
        else:
            st.error(f"❌ No position data found for Player #{player_id}")
    
    elif heat_mode == "👥 Team Heat Maps":
        # Create team-based heat maps
        st.markdown("#### **👥 TEAM HEAT MAP COMPARISON**")
        
        team_data = {0: {'x': [], 'y': [], 'name': 'HOME', 'color': '#dc2626'},
                     1: {'x': [], 'y': [], 'name': 'AWAY', 'color': '#1e40af'}}
        
        # Aggregate team positions
        for player_id_key, player_positions in positions_data.items():
            team_id = player_positions['team_id']
            if team_id in team_data:
                team_data[team_id]['x'].extend(player_positions['x'])
                team_data[team_id]['y'].extend(player_positions['y'])
        
        # Create team heat map
        fig = create_field_base()
        
        for team_id, data in team_data.items():
            if len(data['x']) > 10:  # Minimum positions for meaningful heatmap
                # Convert hex color to RGB for alpha blending
                hex_color = data['color']
                r = int(hex_color[1:3], 16)
                g = int(hex_color[3:5], 16)
                b = int(hex_color[5:7], 16)
                
                fig.add_trace(go.Histogram2d(
                    x=data['x'],
                    y=data['y'],
                    nbinsx=25,
                    nbinsy=15,
                    colorscale=[
                        [0, f'rgba({r},{g},{b},0)'],
                        [0.4, f'rgba({r},{g},{b},0.3)'],
                        [0.7, f'rgba({r},{g},{b},0.6)'],
                        [1.0, f'rgba({r},{g},{b},0.8)']
                    ],
                    showscale=False,
                    name=f"{data['name']} Heat Map",
                    hovertemplate=f"<b>{data['name']} Team Activity</b><br>X: %{{x:.1f}}<br>Y: %{{y:.1f}}<br>Density: %{{z}}<extra></extra>"
                ))
        
        fig.update_layout(
            title="👥 Team Heat Map Comparison",
            xaxis=dict(range=[-2, DISPLAY_LENGTH+2], showgrid=False, zeroline=False, 
                      showticklabels=False, fixedrange=True),
            yaxis=dict(range=[-2, DISPLAY_WIDTH+2], showgrid=False, zeroline=False, 
                      showticklabels=False, fixedrange=True, scaleanchor="x", scaleratio=1),
            plot_bgcolor='#0a0f1c',
            paper_bgcolor='#0a0f1c',
            height=600,
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Team comparison stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### **🔴 HOME TEAM COVERAGE**")
            home_positions = len(team_data[0]['x'])
            st.metric("Total Positions", f"{home_positions:,}")
            if home_positions > 0:
                avg_x = sum(team_data[0]['x']) / len(team_data[0]['x'])
                avg_y = sum(team_data[0]['y']) / len(team_data[0]['y'])
                st.metric("Avg Position", f"({avg_x:.1f}, {avg_y:.1f})")
        
        with col2:
            st.markdown("#### **🔵 AWAY TEAM COVERAGE**")
            away_positions = len(team_data[1]['x'])
            st.metric("Total Positions", f"{away_positions:,}")
            if away_positions > 0:
                avg_x = sum(team_data[1]['x']) / len(team_data[1]['x'])
                avg_y = sum(team_data[1]['y']) / len(team_data[1]['y'])
                st.metric("Avg Position", f"({avg_x:.1f}, {avg_y:.1f})")
    
    elif heat_mode == "⚡ Movement Trails":
        # Show movement trails for all players
        st.markdown("#### **⚡ PLAYER MOVEMENT TRAILS**")
        
        # Enhanced selection options
        col1, col2 = st.columns([2, 2])
        
        with col1:
            trail_mode = st.radio(
                "📍 **Trail Display Mode:**",
                ["🌟 All Players", "👥 Team Filter", "🎯 Individual Player"],
                horizontal=False,
                key="trail_mode"
            )
        
        with col2:
            if trail_mode == "👥 Team Filter":
                team_filter = st.selectbox(
                    "🔍 **Select Team:**",
                    ["Both Teams", "HOME Only", "AWAY Only"],
                    key="team_filter_trails"
                )
            elif trail_mode == "🎯 Individual Player":
                # Create player options list
                player_options = []
                for player_id_key, player_positions in positions_data.items():
                    team_id = player_positions['team_id']
                    if team_id in [0, 1]:  # Only valid teams
                        team_name = "HOME" if team_id == 0 else "AWAY"
                        player_options.append({
                            'id': player_id_key,
                            'label': f"#{player_id_key} ({team_name})",
                            'team_id': team_id
                        })
                
                if player_options:
                    # Sort by team and then by player ID
                    player_options.sort(key=lambda x: (x['team_id'], int(x['id']) if x['id'].isdigit() else x['id']))
                    
                    selected_player_label = st.selectbox(
                        "🎯 **Select Player:**",
                        [p['label'] for p in player_options],
                        key="individual_player_trails"
                    )
                    
                    # Find selected player ID
                    selected_player_id = None
                    for p in player_options:
                        if p['label'] == selected_player_label:
                            selected_player_id = p['id']
                            break
                else:
                    st.warning("⚠️ No players available for individual selection")
                    return
        
        # Trail visualization settings
        st.markdown("#### **🎨 VISUALIZATION SETTINGS**")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            trail_opacity = st.slider("Trail Opacity", 0.1, 1.0, 0.7, 0.1, key="trail_opacity")
        with col_b:
            trail_width = st.slider("Trail Width", 1, 5, 2, 1, key="trail_width")
        with col_c:
            sample_rate = st.slider("Detail Level", 1, 10, 3, 1, 
                                   help="Higher = more detailed trail (slower)", key="trail_sample")
        
        # Create the field visualization
        fig = create_field_base()
        
        # Add helpful info about legend interaction
        if trail_mode in ["🌟 All Players", "👥 Team Filter"]:
            st.info("💡 **Interactive Legend**: Click on any player name in the legend to show/hide their trail. Double-click to isolate a single player.")
        
        colors = ["#dc2626", "#1e40af"]  # HOME, AWAY
        trail_count = 0
        total_points = 0
        
        # Apply filtering based on mode
        if trail_mode == "🌟 All Players":
            # Show all players
            filtered_players = positions_data.items()
            
        elif trail_mode == "👥 Team Filter":
            # Filter by team
            filtered_players = []
            for player_id_key, player_positions in positions_data.items():
                team_id = player_positions['team_id']
                
                if team_filter == "HOME Only" and team_id == 0:
                    filtered_players.append((player_id_key, player_positions))
                elif team_filter == "AWAY Only" and team_id == 1:
                    filtered_players.append((player_id_key, player_positions))
                elif team_filter == "Both Teams" and team_id in [0, 1]:
                    filtered_players.append((player_id_key, player_positions))
                    
        elif trail_mode == "🎯 Individual Player":
            # Show only selected player
            if selected_player_id and selected_player_id in positions_data:
                filtered_players = [(selected_player_id, positions_data[selected_player_id])]
            else:
                filtered_players = []
        
        # Create trails for filtered players
        for player_id_key, player_positions in filtered_players:
            team_id = player_positions['team_id']
            
            if len(player_positions['x']) > 5:  # Minimum trail length
                # Sample points for performance (adjustable detail level)
                sample_x = player_positions['x'][::sample_rate]
                sample_y = player_positions['y'][::sample_rate]
                sample_frames = player_positions['frames'][::sample_rate]
                
                team_color = colors[team_id]
                team_name = "HOME" if team_id == 0 else "AWAY"
                
                # Enhanced styling for individual player mode
                if trail_mode == "🎯 Individual Player":
                    # Special styling for individual player
                    line_width = trail_width + 2
                    opacity = min(1.0, trail_opacity + 0.2)
                    
                    # Add movement direction arrows (sample every 10th point)
                    arrow_x = sample_x[::10]
                    arrow_y = sample_y[::10]
                    
                    # Add the main trail
                    fig.add_trace(go.Scatter(
                        x=sample_x,
                        y=sample_y,
                        mode='lines+markers',
                        line=dict(color=team_color, width=line_width),
                        marker=dict(size=4, color=team_color, opacity=0.6),
                        opacity=opacity,
                        name=f'🎯 {team_name} #{player_id_key} Trail',
                        showlegend=True,
                        hovertemplate=f"<b>🎯 SELECTED: {team_name} Player #{player_id_key}</b><br>Position: (%{{x:.1f}}, %{{y:.1f}})<br>Frame: %{{text}}<br><b>Trail Point %{{pointNumber}} of {len(sample_x)}</b><extra></extra>",
                        text=sample_frames
                    ))
                    
                    # Add start and end markers
                    if len(sample_x) >= 2:
                        # Start marker
                        fig.add_trace(go.Scatter(
                            x=[sample_x[0]],
                            y=[sample_y[0]],
                            mode='markers+text',
                            marker=dict(size=15, color='#22c55e', symbol='circle', 
                                       line=dict(width=3, color='white')),
                            text=['START'],
                            textposition="top center",
                            textfont=dict(color='white', size=10, family="Arial Black"),
                            name=f'🟢 Start Position',
                            showlegend=True,
                            hovertemplate=f"<b>🟢 START POSITION</b><br>{team_name} #{player_id_key}<br>Frame: {sample_frames[0]}<extra></extra>"
                        ))
                        
                        # End marker
                        fig.add_trace(go.Scatter(
                            x=[sample_x[-1]],
                            y=[sample_y[-1]],
                            mode='markers+text',
                            marker=dict(size=15, color='#ef4444', symbol='circle',
                                       line=dict(width=3, color='white')),
                            text=['END'],
                            textposition="top center",
                            textfont=dict(color='white', size=10, family="Arial Black"),
                            name=f'🔴 End Position',
                            showlegend=True,
                            hovertemplate=f"<b>🔴 END POSITION</b><br>{team_name} #{player_id_key}<br>Frame: {sample_frames[-1]}<extra></extra>"
                        ))
                else:
                    # Standard styling for multiple players
                    fig.add_trace(go.Scatter(
                        x=sample_x,
                        y=sample_y,
                        mode='lines',
                        line=dict(color=team_color, width=trail_width),
                        opacity=trail_opacity,
                        name=f'{team_name} #{player_id_key}',
                        showlegend=True,  # Show all players in legend for individual control
                        hovertemplate=f"<b>{team_name} Player #{player_id_key}</b><br>X: %{{x:.1f}}<br>Y: %{{y:.1f}}<br>Frame: %{{text}}<extra></extra>",
                        text=sample_frames
                    ))
                
                trail_count += 1
                total_points += len(sample_x)
        
        # Update chart title based on mode
        if trail_mode == "🎯 Individual Player":
            chart_title = f"🎯 Individual Trail: {selected_player_label if 'selected_player_label' in locals() else 'Player'}"
        elif trail_mode == "👥 Team Filter":
            chart_title = f"👥 Team Trails: {team_filter if 'team_filter' in locals() else 'All Teams'}"
        else:
            chart_title = "🌟 All Player Movement Trails"
        
        fig.update_layout(
            title=chart_title,
            xaxis=dict(range=[-2, DISPLAY_LENGTH+2], showgrid=False, zeroline=False, 
                      showticklabels=False, fixedrange=True),
            yaxis=dict(range=[-2, DISPLAY_WIDTH+2], showgrid=False, zeroline=False, 
                      showticklabels=False, fixedrange=True, scaleanchor="x", scaleratio=1),
            plot_bgcolor='#0a0f1c',
            paper_bgcolor='#0a0f1c',
            height=600,
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                x=1.02, 
                y=1, 
                bgcolor="rgba(15, 23, 42, 0.95)", 
                bordercolor="#374151", 
                font=dict(color="white", size=11),
                itemsizing="constant",
                itemwidth=30,
                tracegroupgap=2,
                title=dict(text="<b>Player Trails</b><br><i>Click to show/hide</i>", font=dict(size=12)),
                orientation="v"
            ),
            margin=dict(l=10, r=200, t=80, b=10)  # More space for legend
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Enhanced movement statistics
        st.markdown("#### **📊 TRAIL ANALYSIS**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Players Shown", trail_count)
        
        with col2:
            st.metric("Total Trail Points", f"{total_points:,}")
        
        with col3:
            frames_range = frame_range[1] - frame_range[0] + 1
            st.metric("Frames Analyzed", f"{frames_range:,}")
        
        with col4:
            avg_points = total_points / max(trail_count, 1)
            st.metric("Avg Points/Player", f"{avg_points:.0f}")
        
        # Individual player detailed stats (only in individual mode)
        if trail_mode == "🎯 Individual Player" and 'selected_player_id' in locals() and selected_player_id:
            st.markdown("#### **🎯 INDIVIDUAL PLAYER ANALYSIS**")
            
            player_data = positions_data[selected_player_id]
            
            col_i, col_ii, col_iii = st.columns(3)
            
            with col_i:
                # Calculate movement distance
                distances = []
                for i in range(1, len(player_data['x'])):
                    dx = player_data['x'][i] - player_data['x'][i-1]
                    dy = player_data['y'][i] - player_data['y'][i-1]
                    dist = (dx**2 + dy**2)**0.5
                    distances.append(dist)
                
                total_movement = sum(distances) if distances else 0
                st.metric("Total Movement", f"{total_movement:.1f} units")
                
                # Movement zones
                left_positions = len([x for x in player_data['x'] if x < 40])
                center_positions = len([x for x in player_data['x'] if 40 <= x < 80])
                right_positions = len([x for x in player_data['x'] if x >= 80])
                total_positions = len(player_data['x'])
                
                st.metric("Left Zone", f"{left_positions/total_positions*100:.1f}%")
            
            with col_ii:
                # Time-based metrics
                total_frames = len(player_data['frames'])
                if total_frames > 1:
                    frame_span = player_data['frames'][-1] - player_data['frames'][0]
                    avg_movement_per_frame = total_movement / frame_span if frame_span > 0 else 0
                    st.metric("Movement/Frame", f"{avg_movement_per_frame:.2f} units")
                
                st.metric("Center Zone", f"{center_positions/total_positions*100:.1f}%")
            
            with col_iii:
                # Coverage metrics
                x_range = max(player_data['x']) - min(player_data['x']) if player_data['x'] else 0
                y_range = max(player_data['y']) - min(player_data['y']) if player_data['y'] else 0
                
                st.metric("X Coverage", f"{x_range:.1f} units")
                st.metric("Right Zone", f"{right_positions/total_positions*100:.1f}%")
            
            # Movement intensity over time
            if distances and len(distances) > 10:
                st.markdown("#### **📈 MOVEMENT INTENSITY OVER TIME**")
                
                # Create intensity chart (rolling average of movement)
                window_size = min(10, len(distances) // 10)
                if window_size > 1:
                    rolling_avg = []
                    for i in range(len(distances)):
                        start_idx = max(0, i - window_size + 1)
                        end_idx = i + 1
                        avg_dist = sum(distances[start_idx:end_idx]) / len(distances[start_idx:end_idx])
                        rolling_avg.append(avg_dist)
                    
                    intensity_fig = go.Figure()
                    intensity_fig.add_trace(go.Scatter(
                        x=list(range(len(rolling_avg))),
                        y=rolling_avg,
                        mode='lines',
                        line=dict(color=colors[player_data['team_id']], width=2),
                        name='Movement Intensity',
                        fill='tonexty'
                    ))
                    
                    intensity_fig.update_layout(
                        title="Movement Intensity Timeline",
                        xaxis_title="Time Progression",
                        yaxis_title="Movement Intensity",
                        height=250,
                        showlegend=False,
                        plot_bgcolor='#0a0f1c',
                        paper_bgcolor='#0a0f1c',
                        font=dict(color='white')
                    )
                    
                    st.plotly_chart(intensity_fig, use_container_width=True)
    
    # Analysis insights
    st.markdown("#### **🎯 HEAT MAP INSIGHTS**")
    
    # Calculate field coverage insights
    all_x = []
    all_y = []
    for player_positions in positions_data.values():
        all_x.extend(player_positions['x'])
        all_y.extend(player_positions['y'])
    
    if all_x and all_y:
        # Field utilization
        field_coverage_x = (max(all_x) - min(all_x)) / DISPLAY_LENGTH * 100
        field_coverage_y = (max(all_y) - min(all_y)) / DISPLAY_WIDTH * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            if field_coverage_x > 80:
                st.success(f"🔥 **EXCELLENT FIELD WIDTH USAGE** - {field_coverage_x:.1f}% coverage")
            elif field_coverage_x > 60:
                st.info(f"📊 **GOOD FIELD WIDTH USAGE** - {field_coverage_x:.1f}% coverage")
            else:
                st.warning(f"⚠️ **LIMITED FIELD WIDTH USAGE** - {field_coverage_x:.1f}% coverage")
        
        with col2:
            if field_coverage_y > 80:
                st.success(f"🔥 **EXCELLENT FIELD LENGTH USAGE** - {field_coverage_y:.1f}% coverage")
            elif field_coverage_y > 60:
                st.info(f"📊 **GOOD FIELD LENGTH USAGE** - {field_coverage_y:.1f}% coverage")
            else:
                st.warning(f"⚠️ **LIMITED FIELD LENGTH USAGE** - {field_coverage_y:.1f}% coverage")

        if st.button("📋 Export Structure Map", use_container_width=True):
            structure_map = {}
            for section_key, section_name in sections:
                section_data = safe_get(data, section_key, None)
                if section_data is not None:
                    if isinstance(section_data, dict):
                        structure_map[section_key] = list(section_data.keys())
                    elif isinstance(section_data, list) and section_data:
                        structure_map[section_key] = f"List with {len(section_data)} items"
                    else:
                        structure_map[section_key] = str(type(section_data).__name__)
            
            st.download_button(
                "Download Structure Map",
                data=json.dumps(structure_map, indent=2),
                file_name=f"structure_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("🔧 Export Debug Info", use_container_width=True):
            debug_info = {
                'file_size_mb': len(str(data)) / (1024 * 1024),
                'total_keys': len(data.keys()) if isinstance(data, dict) else 0,
                'data_type': str(type(data).__name__),
                'timestamp': datetime.now().isoformat(),
                'sections_analysis': availability_data
            }
            
            st.download_button(
                "Download Debug Info",
                data=json.dumps(debug_info, indent=2),
                file_name=f"debug_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_enhanced_field_visualization(data: Dict, frame_number: int):
    """Create FIFA Manager-style tactical interface with RADAR view"""
    st.markdown('<div class="data-section">', unsafe_allow_html=True)
    
    # FIFA-style header with manager controls
    metadata = safe_get(data, 'match_metadata', {})
    fps = metadata.get('fps', 30.0)
    max_frames = metadata.get('total_frames', 1000)
    match_time = format_time(frame_number / fps)
    
    # Manager Dashboard Header
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    with col1:
        st.markdown("### ⚽ **MANAGER VIEW** - Tactical Analysis")
    with col2:
        st.markdown(f"**⏱️ {match_time}**")
    with col3:
        progress = (frame_number / max_frames) * 100
        st.markdown(f"**📊 {progress:.1f}%**")
    with col4:
        half = "1st Half" if frame_number < max_frames/2 else "2nd Half"
        st.markdown(f"**🕐 {half}**")
    with col5:
        tempo = "HIGH" if frame_number % 3 == 0 else "MED"  # Simplified tempo
        st.markdown(f"**🔥 {tempo} TEMPO**")
    
    # Get comprehensive match data
    raw_players = safe_get(data, 'raw_detections.players', [])
    raw_ball = safe_get(data, 'raw_detections.ball', [])
    movement_analytics = safe_get(data, 'movement_analytics', [])
    formation_data = safe_get(data, 'formation_analytics', [])
    spatial_data = safe_get(data, 'spatial_analytics.per_frame_occupancy', [])
    
    if not raw_players or frame_number >= len(raw_players):
        st.error("❌ **NO MATCH DATA** - Unable to generate manager view")
        st.info("📊 **Data Structure Available:**")
        st.json({
            "raw_players_frames": len(raw_players) if raw_players else 0,
            "raw_ball_frames": len(raw_ball) if raw_ball else 0,
            "movement_analytics_count": len(movement_analytics) if movement_analytics else 0,
            "current_frame": frame_number
        })
        return
    
    # Manager View Tabs
    view_tab1, view_tab2, view_tab3, view_tab4 = st.tabs([
        "🏟️ **TACTICAL VIEW**", 
        "📡 **RADAR MODE**", 
        "🎯 **FORMATION**", 
        "📊 **ANALYTICS**"
    ])
    
    with view_tab1:
        create_fifa_tactical_view(data, frame_number, raw_players, raw_ball, movement_analytics)
    
    with view_tab2:
        create_fifa_radar_view(data, frame_number, raw_players, movement_analytics, spatial_data)
    
    with view_tab3:
        create_fifa_formation_view(data, frame_number, formation_data, raw_players)
    
    with view_tab4:
        create_fifa_analytics_dashboard(data, frame_number, movement_analytics)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function with comprehensive analysis"""
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">⚽ Football Analytics Pro</h1>
        <h2 style="margin: 0.5rem 0; font-size: 1.2rem; opacity: 0.9;">
            Professional Match Analysis Dashboard
        </h2>
        <p style="margin: 0; opacity: 0.8;">
            Comprehensive soccer analytics platform for coaches and analysts
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload Match Analytics JSON",
        type=['json'],
        help="Upload the comprehensive analytics file generated by RADAR mode (up to 300MB)",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Load and validate data
        with st.spinner("🔄 Loading and validating match data..."):
            data = load_match_data(uploaded_file)
        
        if data:
            # Store in session state
            st.session_state['match_data'] = data
            
            # Create main header with match info
            create_main_header(data)
            
            # Data overview section
            create_data_overview(data)
            
            # Timeline controls
            frame_number = create_timeline_controls(data)
            st.session_state['selected_frame'] = frame_number
            
            # Main analysis sections
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "🏟️ Live View", 
                "📊 Performance", 
                "⚽ Possession", 
                "🎯 Formation", 
                "🗺️ Spatial", 
                "⚡ Events", 
                "🔍 Data"
            ])
            
            with tab1:
                create_enhanced_field_visualization(data, frame_number)
            
            with tab2:
                create_comprehensive_performance_analysis(data)
            
            with tab3:
                create_enhanced_possession_analysis(data)
            
            with tab4:
                create_formation_analytics(data, frame_number)
            
            with tab5:
                create_spatial_analytics(data, frame_number)
            
            with tab6:
                create_tactical_events_analysis(data)
            
            with tab7:
                create_raw_data_inspector(data)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to Football Analytics Pro
        
        **The most comprehensive soccer match analysis platform**
        
        ### 🎯 Key Features:
        - **🏟️ Live Field Visualization**: Real-time player positions with speed-based markers
        - **📊 Performance Analytics**: Comprehensive player statistics with speed validation
        - **⚽ Possession Analysis**: Detailed ball control and team dynamics
        - **🎯 Formation Analytics**: Team shape analysis and tactical insights
        - **🗺️ Spatial Analytics**: Zone control and field occupancy analysis
        - **⚡ Tactical Events**: Pressing, offside, and game intelligence detection
        - **🔍 Raw Data Inspector**: Full data structure exploration and debugging
        
        ### 📁 Getting Started:
        1. **Generate Data**: Use `python main.py --mode RADAR` to create comprehensive analytics
        2. **Upload JSON**: Upload your analytics file (up to 300MB supported)
        3. **Analyze**: Navigate through tabs to explore every aspect of the match
        
        ### 🚀 New in This Version:
        - ✅ **Speed Validation**: Automatic filtering of unrealistic player speeds (>15 m/s)
        - ✅ **Enhanced Visualizations**: Professional field layout with ball tracking
        - ✅ **Comprehensive Data Usage**: Every available data section is now utilized
        - ✅ **Performance Optimization**: Better handling of large JSON files
        - ✅ **Advanced Analytics**: Formation analysis, spatial control, tactical events
        
        ### 🎥 Data Requirements:
        Your JSON file should be generated using RADAR mode for complete analysis:
        ```bash
        python main.py --source match.mp4 --target output.mp4 --device cuda --mode RADAR
        ```
        
        **Ready to analyze your match? Upload your analytics file above! ⚽**
        """)

def create_fifa_match_trends(movement_data: list):
    """Visualize aggregated match trends for both teams on a radar chart and summary table."""
    import pandas as pd  # local import to keep global namespace clean

    st.markdown("### 📈 **MATCH TRENDS & TEAM COMPARISON**")

    # Aggregate movement stats by team
    team_raw = {0: [], 1: []}
    for p in movement_data:
        t_id = p.get("team_id", -1)
        if t_id in team_raw:
            team_raw[t_id].append(p.get("movement_stats", {}))

    # Build per-team metrics
    team_metrics = {}
    for t_id, stats in team_raw.items():
        if not stats:
            continue
        team_metrics[t_id] = {
            "avg_distance": sum(s.get("total_distance", 0) for s in stats) / len(stats),
            "avg_speed": sum(s.get("average_speed", 0) for s in stats) / len(stats),
            "max_speed": max(s.get("max_speed", 0) for s in stats),
            "total_sprints": sum(s.get("sprint_count", 0) for s in stats),
        }

    if not team_metrics:
        st.info("No movement data available for trend visualisation.")
        return

    # Radar plot helper
    def radar_for_team(label: str, color: str, values: list):
        return go.Scatterpolar(r=values, theta=categories, fill="toself", name=label, line=dict(color=color))

    categories = ["Distance/Player", "Avg Speed", "Max Speed", "Sprint Count"]

    # Scale raw metrics for nicer visual balance
    def scale_metrics(m):
        return [
            m["avg_distance"] / 100,  # distance in hundreds of meters
            m["avg_speed"] * 10,
            m["max_speed"] * 5,
            m["total_sprints"],
        ]

    fig = go.Figure()
    if 0 in team_metrics:
        fig.add_trace(radar_for_team("HOME", "#dc2626", scale_metrics(team_metrics[0])))
    if 1 in team_metrics:
        fig.add_trace(radar_for_team("AWAY", "#1e40af", scale_metrics(team_metrics[1])))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
        showlegend=True,
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="white"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Tabular comparison
    if len(team_metrics) == 2:
        comp_df = pd.DataFrame(
            {
                "Metric": [c for c in categories],
                "HOME": [f"{team_metrics[0]['avg_distance']:.0f}m",
                          f"{team_metrics[0]['avg_speed']:.2f} m/s",
                          f"{team_metrics[0]['max_speed']:.2f} m/s",
                          f"{team_metrics[0]['total_sprints']}"],
                "AWAY": [f"{team_metrics[1]['avg_distance']:.0f}m",
                          f"{team_metrics[1]['avg_speed']:.2f} m/s",
                          f"{team_metrics[1]['max_speed']:.2f} m/s",
                          f"{team_metrics[1]['total_sprints']}"]
            }
        )
        st.dataframe(comp_df, use_container_width=True)

def create_fifa_leaderboards(movement_data: list):
    """FIFA-style leaderboards for various performance metrics"""
    
    st.markdown("### 🏆 **PERFORMANCE LEADERBOARDS**")
    
    if not movement_data:
        st.info("No player data available for leaderboards")
        return
    
    # Process player data
    players = []
    for player in movement_data:
        if player.get('team_id', -1) in [0, 1]:
            stats = player.get('movement_stats', {})
            players.append({
                'Player': f"#{player['player_id']}",
                'Team': "🔴 HOME" if player['team_id'] == 0 else "🔵 AWAY",
                'Distance': stats.get('total_distance', 0),
                'Max Speed': stats.get('max_speed', 0),
                'Avg Speed': stats.get('average_speed', 0),
                'Sprints': stats.get('sprint_count', 0)
            })
    
    if not players:
        st.info("No valid player data for leaderboards")
        return
    
    # Create leaderboards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### **📏 DISTANCE LEADERS**")
        distance_leaders = sorted(players, key=lambda x: x['Distance'], reverse=True)[:5]
        for i, player in enumerate(distance_leaders, 1):
            st.write(f"{i}. {player['Team']} {player['Player']} - {player['Distance']:.0f}m")
        
        st.markdown("#### **⚡ SPEED LEADERS**")
        speed_leaders = sorted(players, key=lambda x: x['Max Speed'], reverse=True)[:5]
        for i, player in enumerate(speed_leaders, 1):
            st.write(f"{i}. {player['Team']} {player['Player']} - {player['Max Speed']:.1f} m/s")
    
    with col2:
        st.markdown("#### **🚀 SPRINT LEADERS**")
        sprint_leaders = sorted(players, key=lambda x: x['Sprints'], reverse=True)[:5]
        for i, player in enumerate(sprint_leaders, 1):
            st.write(f"{i}. {player['Team']} {player['Player']} - {player['Sprints']} sprints")
        
        st.markdown("#### **💪 WORK RATE LEADERS**")
        work_leaders = sorted(players, key=lambda x: x['Avg Speed'], reverse=True)[:5]
        for i, player in enumerate(work_leaders, 1):
            st.write(f"{i}. {player['Team']} {player['Player']} - {player['Avg Speed']:.1f} m/s")

def create_fifa_quick_insights(movement_data: list):
    """FIFA-style quick insights and tactical observations"""
    
    st.markdown("### ⚡ **QUICK INSIGHTS & TACTICAL OBSERVATIONS**")
    
    if not movement_data:
        st.info("No movement data available for insights")
        return
    
    # Calculate team statistics
    team_stats = {0: [], 1: []}
    for player in movement_data:
        team_id = player.get('team_id', -1)
        if team_id in team_stats:
            team_stats[team_id].append(player.get('movement_stats', {}))
    
    insights = []
    
    # Distance comparison
    if team_stats[0] and team_stats[1]:
        team0_avg_dist = sum(s.get('total_distance', 0) for s in team_stats[0]) / len(team_stats[0])
        team1_avg_dist = sum(s.get('total_distance', 0) for s in team_stats[1]) / len(team_stats[1])
        
        if team0_avg_dist > team1_avg_dist * 1.1:
            insights.append("🔴 HOME team showing higher work rate - covering more ground per player")
        elif team1_avg_dist > team0_avg_dist * 1.1:
            insights.append("🔵 AWAY team showing higher work rate - covering more ground per player")
        else:
            insights.append("⚖️ Both teams showing similar work rate levels")
    
    # Speed analysis
    all_speeds = []
    for team_id in [0, 1]:
        for stats in team_stats[team_id]:
            all_speeds.append(stats.get('max_speed', 0))
    
    if all_speeds:
        avg_speed = sum(all_speeds) / len(all_speeds)
        if avg_speed > 8:
            insights.append("🚀 High-intensity match - players reaching excellent speeds")
        elif avg_speed > 6:
            insights.append("📈 Moderate-intensity match - good athletic performance")
        else:
            insights.append("📊 Conservative pace - tactical, possession-based approach")
    
    # Sprint activity
    total_sprints = sum(s.get('sprint_count', 0) for team_stats_list in team_stats.values() for s in team_stats_list)
    if total_sprints > 50:
        insights.append("⚡ High sprint activity - dynamic, attacking gameplay")
    elif total_sprints > 20:
        insights.append("🏃 Moderate sprint activity - balanced approach")
    else:
        insights.append("🚶 Low sprint activity - controlled, methodical gameplay")
    
    # Display insights
    st.markdown('<div class="insights-panel">', unsafe_allow_html=True)
    st.markdown('<div class="insights-title">🎯 Key Tactical Insights</div>', unsafe_allow_html=True)
    
    for insight in insights:
        st.markdown(f'<div class="insight-item">{insight}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### **📊 MATCH INTENSITY**")
        intensity_score = min(100, int((avg_speed / 10) * 100)) if all_speeds else 0
        st.progress(intensity_score / 100)
        st.write(f"**{intensity_score}/100** - Match Intensity Score")
    
    with col2:
        st.markdown("#### **🔥 TACTICAL TEMPO**")
        tempo_score = min(100, int((total_sprints / 100) * 100))
        st.progress(tempo_score / 100)
        st.write(f"**{tempo_score}/100** - Tactical Tempo Score")

def create_enhanced_possession_analysis(data: Dict):
    """Enhanced possession analysis with team dynamics"""
    
    st.markdown("### ⚽ **POSSESSION ANALYSIS**")
    
    possession_data = safe_get(data, 'possession_analytics', {})
    possession_segments = possession_data.get('possession_segments', [])
    possession_stats = possession_data.get('possession_stats', {})
    
    if not possession_segments:
        st.warning("⚠️ No possession data available")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        team_0_poss = possession_stats.get('team_0', {}).get('percentage', 0)
        st.metric("🔴 Home Possession", f"{team_0_poss:.1f}%")
    
    with col2:
        team_1_poss = possession_stats.get('team_1', {}).get('percentage', 0)
        st.metric("🔵 Away Possession", f"{team_1_poss:.1f}%")
    
    with col3:
        total_segments = len(possession_segments)
        st.metric("Total Segments", total_segments)
    
    # Possession timeline
    if possession_segments:
        st.markdown("#### **📈 POSSESSION TIMELINE**")
        
        timeline_data = []
        for segment in possession_segments:
            timeline_data.append({
                'Start': segment.get('start_frame', 0),
                'Duration': segment.get('duration_frames', 0),
                'Team': segment.get('team_id', 0),
                'Quality': segment.get('quality_score', 0.5)
            })
        
        if timeline_data:
            df = pd.DataFrame(timeline_data)
            
            # Create possession timeline visualization
            fig = go.Figure()
            
            colors = {0: '#dc2626', 1: '#1e40af'}
            
            for team_id in [0, 1]:
                team_data = df[df['Team'] == team_id]
                if not team_data.empty:
                    team_name = "HOME" if team_id == 0 else "AWAY"
                    fig.add_trace(go.Scatter(
                        x=team_data['Start'],
                        y=[team_id] * len(team_data),
                        mode='markers',
                        marker=dict(
                            size=team_data['Duration'] / 10,
                            color=colors[team_id],
                            opacity=0.7
                        ),
                        name=f"{team_name} Possession",
                        hovertemplate=f"<b>{team_name} Possession</b><br>Start: %{{x}}<br>Duration: %{{text}} frames<extra></extra>",
                        text=team_data['Duration']
                    ))
            
            fig.update_layout(
                title="Possession Timeline",
                xaxis_title="Frame Number",
                yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['HOME', 'AWAY']),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)

def create_formation_analytics(data: Dict, frame_number: int):
    """Formation and tactical shape analysis"""
    
    st.markdown("### 🎯 **FORMATION ANALYTICS**")
    
    formation_data = safe_get(data, 'formation_analytics', [])
    
    if not formation_data:
        st.warning("⚠️ No formation data available")
        return
    
    # Find formations around current frame
    current_formations = [f for f in formation_data if abs(f.get('frame_id', 0) - frame_number) < 100]
    
    if not current_formations:
        st.info("ℹ️ No formation data near current frame")
        return
    
    # Display formation information
    col1, col2 = st.columns(2)
    
    for team_id, col in [(0, col1), (1, col2)]:
        team_formations = [f for f in current_formations if f.get('team_id') == team_id]
        
        if team_formations:
            latest = team_formations[-1]
            metrics = latest.get('formation_metrics', {})
            
            team_name = "🔴 HOME" if team_id == 0 else "🔵 AWAY"
            
            with col:
                st.markdown(f"#### **{team_name} FORMATION**")
                
                formation = metrics.get('formation_shape', 'Unknown')
                length = metrics.get('length', 0)
                width = metrics.get('width', 0)
                compactness = metrics.get('compactness', 0)
                
                st.metric("Shape", formation)
                st.metric("Length", f"{length:.1f}m")
                st.metric("Width", f"{width:.1f}m")
                st.metric("Compactness", f"{compactness:.1f}m")

def create_spatial_analytics(data: Dict, frame_number: int):
    """Spatial analysis and zone control"""
    
    st.markdown("### 🗺️ **SPATIAL ANALYTICS**")
    
    spatial_data = safe_get(data, 'spatial_analytics', {})
    occupancy_data = spatial_data.get('per_frame_occupancy', [])
    
    if not occupancy_data or frame_number >= len(occupancy_data):
        st.warning("⚠️ No spatial data available for current frame")
        return
    
    current_occupancy = occupancy_data[frame_number]
    zone_occupancy = current_occupancy.get('zone_occupancy', {})
    
    # Zone control visualization
    st.markdown("#### **🎯 ZONE CONTROL**")
    
    zones = ['defensive_third', 'middle_third', 'attacking_third']
    zone_names = ['Defensive Third', 'Middle Third', 'Attacking Third']
    
    zone_data = []
    for zone, name in zip(zones, zone_names):
        if zone in zone_occupancy:
            team_0 = zone_occupancy[zone].get('team_0', 0)
            team_1 = zone_occupancy[zone].get('team_1', 0)
            
            zone_data.append({
                'Zone': name,
                'HOME': team_0,
                'AWAY': team_1,
                'Total': team_0 + team_1
            })
    
    if zone_data:
        df = pd.DataFrame(zone_data)
        
        fig = px.bar(df, x='Zone', y=['HOME', 'AWAY'], 
                    title="Zone Occupancy",
                    color_discrete_map={'HOME': '#dc2626', 'AWAY': '#1e40af'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Zone control summary
        col1, col2, col3 = st.columns(3)
        
        for i, row in df.iterrows():
            with [col1, col2, col3][i]:
                home_control = row['HOME'] / (row['Total'] + 0.001) * 100
                st.metric(
                    row['Zone'], 
                    f"{home_control:.0f}% HOME",
                    delta=f"{home_control - 50:.0f}%" if home_control != 50 else None
                )

def create_tactical_events_analysis(data: Dict):
    """Tactical events analysis including pressing and offside"""
    
    st.markdown("### ⚡ **TACTICAL EVENTS**")
    
    tactical_events = safe_get(data, 'tactical_events', {})
    pressing_events = tactical_events.get('pressing_events', [])
    offside_events = tactical_events.get('offside_events', [])
    
    # Event summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Pressing Events", len(pressing_events))
    
    with col2:
        st.metric("Offside Events", len(offside_events))
    
    with col3:
        total_events = len(pressing_events) + len(offside_events)
        st.metric("Total Events", total_events)
    
    # Event details
    if pressing_events:
        st.markdown("#### **🔥 PRESSING EVENTS**")
        
        pressing_data = []
        for event in pressing_events[:10]:  # Show first 10
            team_name = "HOME" if event.get('team_id') == 0 else "AWAY"
            pressing_data.append({
                'Frame': event.get('frame_start', 0),
                'Team': team_name,
                'Intensity': event.get('intensity', 0),
                'Duration': event.get('duration_frames', 0)
            })
        
        if pressing_data:
            df = pd.DataFrame(pressing_data)
            st.dataframe(df, use_container_width=True)
    
    if offside_events:
        st.markdown("#### **🚩 OFFSIDE EVENTS**")
        
        offside_data = []
        for event in offside_events[:10]:  # Show first 10
            team_name = "HOME" if event.get('team_id') == 0 else "AWAY"
            offside_data.append({
                'Frame': event.get('frame_id', 0),
                'Team': team_name,
                'Player': event.get('player_id', 'Unknown'),
                'Distance': event.get('offside_distance', 0)
            })
        
        if offside_data:
            df = pd.DataFrame(offside_data)
            st.dataframe(df, use_container_width=True)

def create_raw_data_inspector(data: Dict):
    """Raw data inspector for debugging and exploration"""
    
    st.markdown("### 🔍 **RAW DATA INSPECTOR**")
    
    # Data structure overview
    st.markdown("#### **📊 DATA STRUCTURE OVERVIEW**")
    
    sections = [
        ('match_metadata', 'Match Metadata'),
        ('raw_detections', 'Raw Detections'),
        ('tracking_data', 'Tracking Data'),
        ('spatial_analytics', 'Spatial Analytics'),
        ('possession_analytics', 'Possession Analytics'),
        ('movement_analytics', 'Movement Analytics'),
        ('formation_analytics', 'Formation Analytics'),
        ('tactical_events', 'Tactical Events'),
        ('quality_metrics', 'Quality Metrics'),
        ('summary_statistics', 'Summary Statistics')
    ]
    
    availability_data = []
    
    for section_key, section_name in sections:
        section_data = safe_get(data, section_key, None)
        is_available = section_data is not None
        
        if isinstance(section_data, dict):
            data_info = f"Dict with {len(section_data)} keys"
        elif isinstance(section_data, list):
            data_info = f"List with {len(section_data)} items"
        else:
            data_info = str(type(section_data).__name__) if section_data else "Missing"
        
        availability_data.append({
            'Section': section_name,
            'Available': '✅' if is_available else '❌',
            'Type': data_info,
            'Key': section_key
        })
    
    # Display availability table
    availability_df = pd.DataFrame(availability_data)
    st.dataframe(availability_df, use_container_width=True)
    
    # Section explorer
    st.markdown("#### **🔍 SECTION EXPLORER**")
    
    available_sections = [row for row in availability_data if row['Available'] == '✅']
    
    if available_sections:
        selected_section = st.selectbox(
            "Select section to explore:",
            [section['Section'] for section in available_sections]
        )
        
        # Find the key for selected section
        selected_key = None
        for section in available_sections:
            if section['Section'] == selected_section:
                selected_key = section['Key']
                break
        
        if selected_key:
            section_data = safe_get(data, selected_key, {})
            
            # Display section data
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 Show JSON Structure", use_container_width=True):
                    if isinstance(section_data, dict):
                        st.json(list(section_data.keys()))
                    else:
                        st.json(str(type(section_data)))
            
            with col2:
                if st.button("📄 Show Sample Data", use_container_width=True):
                    if isinstance(section_data, dict):
                        # Show first few key-value pairs
                        sample = dict(list(section_data.items())[:3])
                        st.json(sample)
                    elif isinstance(section_data, list) and section_data:
                        # Show first item
                        st.json(section_data[0])
                    else:
                        st.json(section_data)
            
            with col3:
                if st.button("💾 Export Section", use_container_width=True):
                    st.download_button(
                        "Download Section Data",
                        data=json.dumps(section_data, indent=2),
                        file_name=f"{selected_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key=f"download_{selected_key}"
                    )
    
    # Data quality metrics
    st.markdown("#### **📈 DATA QUALITY METRICS**")
    
    quality_metrics = safe_get(data, 'quality_metrics', {})
    
    if quality_metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ball_detection_rate = quality_metrics.get('ball_detection_rate', 0) * 100
            st.metric("Ball Detection Rate", f"{ball_detection_rate:.1f}%")
        
        with col2:
            player_tracking_quality = quality_metrics.get('player_tracking_quality', 0) * 100
            st.metric("Player Tracking Quality", f"{player_tracking_quality:.1f}%")
        
        with col3:
            overall_quality = quality_metrics.get('overall_quality_score', 0) * 100
            st.metric("Overall Quality Score", f"{overall_quality:.1f}%")
    else:
        st.info("ℹ️ No quality metrics available")

if __name__ == "__main__":
    main()