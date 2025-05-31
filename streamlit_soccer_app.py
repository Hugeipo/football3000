import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import math
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="⚽ Coach Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for coach-friendly styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e88e5, #43a047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .coach-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .player-card {
        background: white;
        padding: 1rem;
        border-radius: 0.8rem;
        border-left: 5px solid #1e88e5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .tactical-insight {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #ff6b35;
        margin: 1rem 0;
    }
    .performance-excellent { color: #4caf50; font-weight: bold; }
    .performance-good { color: #2196f3; font-weight: bold; }
    .performance-average { color: #ff9800; font-weight: bold; }
    .performance-poor { color: #f44336; font-weight: bold; }
    
    .stTab {
        background-color: #f8f9fa;
    }
    .team-comparison {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        background: linear-gradient(90deg, #74b9ff, #0984e3);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_validate_json(uploaded_file) -> Optional[Dict]:
    """Enhanced JSON loading with better error handling and fallbacks"""
    try:
        content = uploaded_file.read()
        data = json.loads(content)
        
        # Validate with fallbacks for missing data
        required_sections = {
            'match_metadata': {},
            'raw_detections': {'players': [], 'ball': [], 'pitch': []},
            'tracking_data': {'players': {}, 'ball': {}},
            'spatial_analytics': {'field_zones': {}, 'per_frame_occupancy': []},
            'quality_metrics': {}
        }
        
        # Fill in missing sections with defaults
        for section, default in required_sections.items():
            if section not in data:
                data[section] = default
                st.warning(f"⚠️ Missing {section} - using defaults")
        
        # Validate and repair data structure
        if not data['raw_detections']['players']:
            st.warning("⚠️ No player detection data found")
        
        if not data['tracking_data']['players']:
            st.warning("⚠️ No player tracking data found")
            
        st.success("✅ Match data loaded successfully!")
        return data
        
    except json.JSONDecodeError as e:
        st.error(f"❌ Invalid JSON format: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

def get_safe_data(data: Dict, path: List[str], default=None):
    """Safely navigate nested dictionaries with fallbacks"""
    try:
        current = data
        for key in path:
            current = current[key]
        return current
    except (KeyError, TypeError, IndexError):
        return default

def display_match_overview(data: Dict):
    """Enhanced match overview with coach-relevant metrics"""
    metadata = get_safe_data(data, ['match_metadata'], {})
    quality = get_safe_data(data, ['quality_metrics'], {})
    
    st.markdown('<div class="main-header">⚽ COACH ANALYTICS DASHBOARD</div>', unsafe_allow_html=True)
    
    # Key match information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        duration = metadata.get('duration_seconds', 0)
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        st.markdown(f"""
        <div class="coach-metric">
            <h3>📹 Match Duration</h3>
            <h2>{minutes}:{seconds:02d}</h2>
            <p>{metadata.get('total_frames', 0):,} frames</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        detection_rate = quality.get('player_detection_rate', 0) * 100
        status = "EXCELLENT" if detection_rate > 90 else "GOOD" if detection_rate > 80 else "FAIR"
        st.markdown(f"""
        <div class="coach-metric">
            <h3>👥 Player Tracking</h3>
            <h2>{detection_rate:.1f}%</h2>
            <p>Quality: {status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        ball_rate = quality.get('ball_detection_rate', 0) * 100
        ball_status = "EXCELLENT" if ball_rate > 85 else "GOOD" if ball_rate > 70 else "FAIR"
        st.markdown(f"""
        <div class="coach-metric">
            <h3>⚽ Ball Tracking</h3>
            <h2>{ball_rate:.1f}%</h2>
            <p>Quality: {ball_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        players_tracked = len(get_safe_data(data, ['tracking_data', 'players'], {}))
        st.markdown(f"""
        <div class="coach-metric">
            <h3>🎯 Players Found</h3>
            <h2>{players_tracked}</h2>
            <p>Unique IDs tracked</p>
        </div>
        """, unsafe_allow_html=True)

def create_enhanced_field_layout(highlight_zones=None):
    """Create an enhanced soccer field with coach-friendly annotations"""
    # Standard soccer field dimensions in meters
    field_length = 120
    field_width = 80
    
    fig = go.Figure()
    
    # Field background
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=field_length, y1=field_width,
        line=dict(color="white", width=3),
        fillcolor="darkgreen",
        opacity=0.8
    )
    
    # Field markings
    markings = [
        # Center line
        {"type": "line", "x0": field_length/2, "y0": 0, "x1": field_length/2, "y1": field_width},
        # Center circle
        {"type": "circle", "x0": field_length/2-9.15, "y0": field_width/2-9.15, 
         "x1": field_length/2+9.15, "y1": field_width/2+9.15},
        # Left penalty box
        {"type": "rect", "x0": 0, "y0": field_width/2-20.15, "x1": 16.5, "y1": field_width/2+20.15},
        # Right penalty box  
        {"type": "rect", "x0": field_length-16.5, "y0": field_width/2-20.15, 
         "x1": field_length, "y1": field_width/2+20.15},
        # Left goal box
        {"type": "rect", "x0": 0, "y0": field_width/2-9.16, "x1": 5.5, "y1": field_width/2+9.16},
        # Right goal box
        {"type": "rect", "x0": field_length-5.5, "y0": field_width/2-9.16, 
         "x1": field_length, "y1": field_width/2+9.16},
    ]
    
    for marking in markings:
        fig.add_shape(line=dict(color="white", width=2), **marking)
    
    # Zone highlights for tactical analysis
    if highlight_zones:
        zone_colors = {"attacking": "rgba(255,99,71,0.3)", "defensive": "rgba(70,130,180,0.3)", 
                      "midfield": "rgba(255,215,0,0.3)"}
        
        if "attacking" in highlight_zones:
            fig.add_shape(type="rect", x0=80, y0=0, x1=120, y1=80,
                         fillcolor=zone_colors["attacking"], line=dict(width=0))
        if "defensive" in highlight_zones:
            fig.add_shape(type="rect", x0=0, y0=0, x1=40, y1=80,
                         fillcolor=zone_colors["defensive"], line=dict(width=0))
        if "midfield" in highlight_zones:
            fig.add_shape(type="rect", x0=40, y0=0, x1=80, y1=80,
                         fillcolor=zone_colors["midfield"], line=dict(width=0))
    
    # Field annotations for coaches
    annotations = [
        {"x": 20, "y": 5, "text": "DEFENSIVE THIRD", "font": {"color": "white", "size": 12}},
        {"x": 60, "y": 5, "text": "MIDFIELD", "font": {"color": "white", "size": 12}},
        {"x": 100, "y": 5, "text": "ATTACKING THIRD", "font": {"color": "white", "size": 12}},
    ]
    
    fig.update_layout(
        annotations=annotations,
        xaxis=dict(range=[0, field_length], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, field_width], showgrid=False, zeroline=False, showticklabels=False),
        showlegend=True,
        plot_bgcolor='darkgreen',
        paper_bgcolor='white',
        height=500,
        title=dict(text="⚽ TACTICAL FIELD VIEW", x=0.5, font=dict(size=20, color="darkgreen"))
    )
    
    return fig

def visualize_player_positions_enhanced(data: Dict, selected_frame: int = 0):
    """Enhanced player position visualization with coaching insights"""
    fig = create_enhanced_field_layout()
    
    players_data = get_safe_data(data, ['raw_detections', 'players'], [])
    
    if not players_data or selected_frame >= len(players_data):
        st.warning("⚠️ No player data available for this frame")
        return fig
    
    frame_data = players_data[selected_frame]
    detections = frame_data.get('detections', [])
    
    # Enhanced team colors and symbols
    team_config = {
        0: {"color": "#1e88e5", "name": "Your Team", "symbol": "circle"},
        1: {"color": "#e53935", "name": "Opposition", "symbol": "square"},
        2: {"color": "#ffc107", "name": "Officials", "symbol": "diamond"}
    }
    
    team_positions = {0: [], 1: [], 2: []}
    
    for detection in detections:
        if detection.get('position_field', {}).get('x', 0) > 0:
            # Convert cm to meters
            x = detection['position_field']['x'] / 100
            y = detection['position_field']['y'] / 100
            
            team_id = detection.get('team_id', -1)
            if team_id in team_config:
                team_positions[team_id].append((x, y))
                
                config = team_config[team_id]
                player_id = detection.get('tracker_id', '?')
                
                # Add player marker with enhanced styling
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color=config["color"],
                        symbol=config["symbol"],
                        line=dict(width=3, color='white')
                    ),
                    text=[str(player_id)],
                    textposition='middle center',
                    textfont=dict(color='white', size=10, family="Arial Black"),
                    name=config["name"],
                    showlegend=team_id in [0, 1],
                    hovertemplate=f"<b>{config['name']} Player {player_id}</b><br>" +
                                f"Position: ({x:.1f}m, {y:.1f}m)<br>" +
                                "<extra></extra>"
                ))
    
    # Add tactical insights
    if team_positions[0] and team_positions[1]:
        your_team_x = [pos[0] for pos in team_positions[0]]
        opp_team_x = [pos[0] for pos in team_positions[1]]
        
        your_avg_x = np.mean(your_team_x)
        opp_avg_x = np.mean(opp_team_x)
        
        # Add team centroid lines
        fig.add_shape(type="line", x0=your_avg_x, y0=0, x1=your_avg_x, y1=80,
                     line=dict(color="#1e88e5", width=3, dash="dash"),
                     name="Your Team Line")
        fig.add_shape(type="line", x0=opp_avg_x, y0=0, x1=opp_avg_x, y1=80,
                     line=dict(color="#e53935", width=3, dash="dash"),
                     name="Opposition Line")
    
    fig.update_layout(
        title=f"🎯 Player Positions - Frame {selected_frame} (Time: {selected_frame/30:.1f}s)",
        title_x=0.5
    )
    
    return fig

def create_player_performance_dashboard(data: Dict, player_id: str):
    """Create comprehensive player performance analysis for coaches"""
    player_data = get_safe_data(data, ['tracking_data', 'players', player_id], {})
    
    if not player_data:
        st.error(f"❌ No data found for Player {player_id}")
        return
    
    # Player info header
    team_id = player_data.get('team_id', 'Unknown')
    team_name = "Your Team" if team_id == 0 else "Opposition" if team_id == 1 else "Officials"
    
    st.markdown(f"""
    <div class="player-card">
        <h2>👤 Player {player_id} Analysis</h2>
        <p><strong>Team:</strong> {team_name} | <strong>Tracking Quality:</strong> {player_data.get('tracking_quality', 0)*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics
    trajectory = player_data.get('trajectory', [])
    if not trajectory:
        st.warning("⚠️ No trajectory data available for this player")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate performance metrics
    total_distance = calculate_total_distance(trajectory)
    max_speed = calculate_max_speed(trajectory)
    avg_speed = calculate_avg_speed(trajectory)
    time_in_zones = calculate_zone_time(trajectory)
    
    with col1:
        distance_class = get_performance_class(total_distance, [8000, 10000, 12000])
        st.markdown(f"""
        <div class="coach-metric">
            <h3>🏃 Distance Covered</h3>
            <h2 class="{distance_class}">{total_distance:.0f}m</h2>
            <p>Match total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        speed_class = get_performance_class(max_speed, [5, 7, 9])
        st.markdown(f"""
        <div class="coach-metric">
            <h3>⚡ Max Speed</h3>
            <h2 class="{speed_class}">{max_speed:.1f} m/s</h2>
            <p>Peak velocity</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_speed_class = get_performance_class(avg_speed, [2, 3, 4])
        st.markdown(f"""
        <div class="coach-metric">
            <h3>📊 Avg Speed</h3>
            <h2 class="{avg_speed_class}">{avg_speed:.1f} m/s</h2>
            <p>Overall pace</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        frames_tracked = len(trajectory)
        st.markdown(f"""
        <div class="coach-metric">
            <h3>⏱️ Tracking Time</h3>
            <h2>{frames_tracked/30:.0f}s</h2>
            <p>{frames_tracked} frames</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Heat map and movement analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Player Heat Map")
        heatmap_fig = create_player_heatmap_enhanced(trajectory)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Speed Timeline")
        speed_fig = create_speed_timeline(trajectory)
        st.plotly_chart(speed_fig, use_container_width=True)
    
    # Tactical insights
    st.markdown(f"""
    <div class="tactical-insight">
        <h3>🧠 COACHING INSIGHTS for Player {player_id}</h3>
        <ul>
            <li><strong>Work Rate:</strong> {get_work_rate_insight(total_distance, avg_speed)}</li>
            <li><strong>Positioning:</strong> {get_positioning_insight(time_in_zones)}</li>
            <li><strong>Intensity:</strong> {get_intensity_insight(max_speed, avg_speed)}</li>
            <li><strong>Consistency:</strong> {get_consistency_insight(trajectory)}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def calculate_total_distance(trajectory: List[Dict]) -> float:
    """Calculate total distance covered by player"""
    if len(trajectory) < 2:
        return 0.0
    
    total_dist = 0.0
    for i in range(1, len(trajectory)):
        pos1 = trajectory[i-1]['position_field']
        pos2 = trajectory[i]['position_field']
        
        # Convert cm to meters and calculate distance
        dx = (pos2['x'] - pos1['x']) / 100
        dy = (pos2['y'] - pos1['y']) / 100
        dist = math.sqrt(dx*dx + dy*dy)
        
        # Filter out unrealistic jumps (likely tracking errors)
        if dist < 10:  # Max 10m between frames
            total_dist += dist
    
    return total_dist

def calculate_max_speed(trajectory: List[Dict]) -> float:
    """Calculate maximum speed achieved"""
    max_speed = 0.0
    for point in trajectory:
        velocity = point.get('velocity', {})
        if isinstance(velocity, dict):
            speed = velocity.get('magnitude', 0)
            max_speed = max(max_speed, speed)
    return max_speed

def calculate_avg_speed(trajectory: List[Dict]) -> float:
    """Calculate average speed"""
    speeds = []
    for point in trajectory:
        velocity = point.get('velocity', {})
        if isinstance(velocity, dict):
            speed = velocity.get('magnitude', 0)
            if speed > 0:
                speeds.append(speed)
    return np.mean(speeds) if speeds else 0.0

def calculate_zone_time(trajectory: List[Dict]) -> Dict[str, float]:
    """Calculate time spent in different field zones"""
    zone_time = {"defensive": 0, "midfield": 0, "attacking": 0}
    
    for point in trajectory:
        x = point.get('position_field', {}).get('x', 0) / 100  # Convert to meters
        
        if x < 40:
            zone_time["defensive"] += 1/30  # Convert frames to seconds
        elif x < 80:
            zone_time["midfield"] += 1/30
        else:
            zone_time["attacking"] += 1/30
    
    return zone_time

def get_performance_class(value: float, thresholds: List[float]) -> str:
    """Get CSS class based on performance thresholds"""
    if value >= thresholds[2]:
        return "performance-excellent"
    elif value >= thresholds[1]:
        return "performance-good"
    elif value >= thresholds[0]:
        return "performance-average"
    else:
        return "performance-poor"

def create_player_heatmap_enhanced(trajectory: List[Dict]):
    """Create enhanced player heatmap"""
    fig = create_enhanced_field_layout()
    
    if not trajectory:
        return fig
    
    # Extract positions and convert to meters
    x_positions = [point['position_field']['x']/100 for point in trajectory 
                   if point.get('position_field', {}).get('x', 0) > 0]
    y_positions = [point['position_field']['y']/100 for point in trajectory 
                   if point.get('position_field', {}).get('y', 0) > 0]
    
    if x_positions and y_positions:
        # Create 2D histogram for heatmap
        fig.add_trace(go.Histogram2d(
            x=x_positions, y=y_positions,
            nbinsx=40, nbinsy=25,
            colorscale='Reds',
            opacity=0.7,
            showscale=True,
            colorbar=dict(title="Activity Level")
        ))
        
        # Add movement trail
        fig.add_trace(go.Scatter(
            x=x_positions[::5], y=y_positions[::5],  # Sample every 5th point
            mode='lines',
            line=dict(color='yellow', width=2, dash='dot'),
            name='Movement Trail',
            opacity=0.6
        ))
    
    fig.update_layout(title="🔥 Player Movement Heat Map & Trail")
    return fig

def create_speed_timeline(trajectory: List[Dict]):
    """Create speed timeline chart"""
    times = []
    speeds = []
    
    for point in trajectory:
        times.append(point.get('timestamp', 0))
        velocity = point.get('velocity', {})
        if isinstance(velocity, dict):
            speeds.append(velocity.get('magnitude', 0))
        else:
            speeds.append(0)
    
    fig = go.Figure()
    
    # Speed line
    fig.add_trace(go.Scatter(
        x=times, y=speeds,
        mode='lines',
        line=dict(color='#1e88e5', width=2),
        name='Speed',
        fill='tonexty'
    ))
    
    # Speed zones
    fig.add_hline(y=5.5, line_dash="dash", line_color="red", 
                  annotation_text="Sprint Threshold (5.5 m/s)")
    fig.add_hline(y=4.0, line_dash="dash", line_color="orange", 
                  annotation_text="High Intensity (4.0 m/s)")
    
    fig.update_layout(
        title="📈 Speed Profile Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Speed (m/s)",
        hovermode='x unified'
    )
    
    return fig

def get_work_rate_insight(distance: float, avg_speed: float) -> str:
    """Generate work rate insight for coaches"""
    if distance > 12000:
        return "Excellent work rate - high distance covered"
    elif distance > 10000:
        return "Good work rate - solid contribution"
    elif distance > 8000:
        return "Average work rate - room for improvement"
    else:
        return "Low work rate - may need fitness work"

def get_positioning_insight(zone_time: Dict[str, float]) -> str:
    """Generate positioning insight"""
    total_time = sum(zone_time.values())
    if total_time == 0:
        return "Unable to determine positioning patterns"
    
    zone_percentages = {k: v/total_time*100 for k, v in zone_time.items()}
    dominant_zone = max(zone_percentages, key=zone_percentages.get)
    
    insights = {
        "defensive": "Defensive player - good coverage in own third",
        "midfield": "Box-to-box player - excellent midfield presence", 
        "attacking": "Attack-minded player - focus on final third"
    }
    
    return f"{insights[dominant_zone]} ({zone_percentages[dominant_zone]:.0f}% of time)"

def get_intensity_insight(max_speed: float, avg_speed: float) -> str:
    """Generate intensity insight"""
    if max_speed > 8:
        return "High intensity player - capable of explosive bursts"
    elif max_speed > 6:
        return "Good pace - solid acceleration when needed"
    else:
        return "Consider speed/acceleration training"

def get_consistency_insight(trajectory: List[Dict]) -> str:
    """Generate consistency insight"""
    if len(trajectory) > 1000:
        return "Excellent tracking consistency - reliable data"
    elif len(trajectory) > 500:
        return "Good tracking - sufficient data for analysis"
    else:
        return "Limited tracking data - may affect analysis accuracy"

def create_team_comparison_dashboard(data: Dict):
    """Enhanced team comparison for tactical analysis"""
    st.markdown("""
    <div class="team-comparison">
        <h2>⚔️ TEAM TACTICAL COMPARISON</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Get team data
    players_data = get_safe_data(data, ['tracking_data', 'players'], {})
    
    team_0_players = [pid for pid, pdata in players_data.items() if pdata.get('team_id') == 0]
    team_1_players = [pid for pid, pdata in players_data.items() if pdata.get('team_id') == 1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔵 YOUR TEAM")
        if team_0_players:
            team_0_metrics = calculate_team_metrics(data, team_0_players)
            display_team_metrics(team_0_metrics, "🔵")
        else:
            st.warning("No data available for your team")
    
    with col2:
        st.markdown("### 🔴 OPPOSITION")
        if team_1_players:
            team_1_metrics = calculate_team_metrics(data, team_1_players)
            display_team_metrics(team_1_metrics, "🔴")
        else:
            st.warning("No data available for opposition")
    
    # Tactical insights
    if team_0_players and team_1_players:
        st.markdown("### 🧠 TACTICAL INSIGHTS")
        create_tactical_comparison(data, team_0_players, team_1_players)

def calculate_team_metrics(data: Dict, player_ids: List[str]) -> Dict:
    """Calculate comprehensive team metrics"""
    players_data = get_safe_data(data, ['tracking_data', 'players'], {})
    
    total_distance = 0
    max_speed = 0
    total_players = len(player_ids)
    
    for player_id in player_ids:
        player_data = players_data.get(player_id, {})
        trajectory = player_data.get('trajectory', [])
        
        if trajectory:
            player_distance = calculate_total_distance(trajectory)
            player_max_speed = calculate_max_speed(trajectory)
            
            total_distance += player_distance
            max_speed = max(max_speed, player_max_speed)
    
    return {
        'total_distance': total_distance,
        'avg_distance_per_player': total_distance / max(total_players, 1),
        'max_speed': max_speed,
        'player_count': total_players
    }

def display_team_metrics(metrics: Dict, icon: str):
    """Display team metrics in coach-friendly format"""
    st.metric(f"{icon} Total Distance", f"{metrics['total_distance']:.0f}m")
    st.metric(f"{icon} Avg per Player", f"{metrics['avg_distance_per_player']:.0f}m")
    st.metric(f"{icon} Team Max Speed", f"{metrics['max_speed']:.1f} m/s")
    st.metric(f"{icon} Players Tracked", f"{metrics['player_count']}")

def create_tactical_comparison(data: Dict, team_0_players: List[str], team_1_players: List[str]):
    """Create tactical comparison insights"""
    # Zone occupancy analysis
    occupancy_data = get_safe_data(data, ['spatial_analytics', 'per_frame_occupancy'], [])
    
    if occupancy_data:
        # Calculate average zone occupancy
        team_0_zones = {"defensive_third": 0, "middle_third": 0, "attacking_third": 0}
        team_1_zones = {"defensive_third": 0, "middle_third": 0, "attacking_third": 0}
        
        for frame in occupancy_data:
            zones = frame.get('zone_occupancy', {})
            for zone in team_0_zones.keys():
                zone_data = zones.get(zone, {})
                team_0_zones[zone] += zone_data.get('team_0', 0)
                team_1_zones[zone] += zone_data.get('team_1', 0)
        
        # Average over all frames
        num_frames = len(occupancy_data)
        if num_frames > 0:
            for zone in team_0_zones.keys():
                team_0_zones[zone] /= num_frames
                team_1_zones[zone] /= num_frames
            
            # Create comparison chart
            fig = go.Figure()
            
            zones = list(team_0_zones.keys())
            your_team_values = list(team_0_zones.values())
            opp_team_values = list(team_1_zones.values())
            
            fig.add_trace(go.Bar(
                name='Your Team', x=zones, y=your_team_values,
                marker_color='#1e88e5'
            ))
            
            fig.add_trace(go.Bar(
                name='Opposition', x=zones, y=opp_team_values,
                marker_color='#e53935'
            ))
            
            fig.update_layout(
                title='⚔️ Zone Occupancy Comparison',
                xaxis_title='Field Zones',
                yaxis_title='Average Players in Zone',
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tactical insights
            insights = generate_tactical_insights(team_0_zones, team_1_zones)
            st.markdown(f"""
            <div class="tactical-insight">
                <h3>🎯 KEY TACTICAL INSIGHTS</h3>
                {insights}
            </div>
            """, unsafe_allow_html=True)

def generate_tactical_insights(your_team_zones: Dict, opp_zones: Dict) -> str:
    """Generate tactical insights based on zone occupancy"""
    insights = []
    
    # Attacking vs Defensive focus
    your_attack_focus = your_team_zones['attacking_third'] / max(sum(your_team_zones.values()), 1)
    opp_attack_focus = opp_zones['attacking_third'] / max(sum(opp_zones.values()), 1)
    
    if your_attack_focus > opp_attack_focus:
        insights.append("✅ <strong>More attacking presence than opposition</strong>")
    else:
        insights.append("⚠️ <strong>Opposition showing more attacking intent</strong>")
    
    # Midfield battle
    your_midfield = your_team_zones['middle_third']
    opp_midfield = opp_zones['middle_third']
    
    if your_midfield > opp_midfield:
        insights.append("✅ <strong>Winning the midfield battle</strong>")
    else:
        insights.append("⚠️ <strong>Opposition controlling midfield - consider tactical adjustment</strong>")
    
    # Defensive solidity
    your_defense = your_team_zones['defensive_third']
    if your_defense > 3:
        insights.append("✅ <strong>Solid defensive structure</strong>")
    else:
        insights.append("⚠️ <strong>Defensive line may be too high - risk of counter-attacks</strong>")
    
    return "<ul>" + "".join(f"<li>{insight}</li>" for insight in insights) + "</ul>"

def main():
    # Enhanced header
    st.markdown('<div class="main-header">⚽ COACH ANALYTICS DASHBOARD</div>', unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("📁 Match Analysis Setup")
        st.markdown("Upload your match analytics file to begin tactical analysis")
        
        uploaded_file = st.file_uploader(
            "Choose JSON Analytics File",
            type=['json'],
            help="Upload the JSON file generated by your RADAR analysis"
        )
        
        if uploaded_file is not None:
            with st.spinner("🔄 Loading match data..."):
                data = load_and_validate_json(uploaded_file)
            
            if data:
                st.session_state['soccer_data'] = data
                
                # Match info
                metadata = data.get('match_metadata', {})
                st.success(f"✅ **{metadata.get('video_file', 'Match')}** loaded")
                st.info(f"⏱️ Duration: {metadata.get('duration_seconds', 0)/60:.1f} minutes")
                
                # Timeline controls
                st.header("⏱️ Match Timeline")
                max_frames = len(data.get('raw_detections', {}).get('players', []))
                
                if max_frames > 0:
                    selected_frame = st.slider(
                        "Match Time",
                        min_value=0,
                        max_value=max_frames-1,
                        value=0,
                        format="%d",
                        help="Scrub through match timeline"
                    )
                    
                    # Display time in minutes:seconds
                    time_seconds = selected_frame / 30
                    minutes = int(time_seconds // 60)
                    seconds = int(time_seconds % 60)
                    st.write(f"⏰ **{minutes}:{seconds:02d}**")
                    
                    st.session_state['selected_frame'] = selected_frame
                
                # Player selection
                st.header("👤 Player Focus")
                tracking_data = data.get('tracking_data', {}).get('players', {})
                player_ids = list(tracking_data.keys())
                
                if player_ids:
                    selected_player = st.selectbox(
                        "Select Player for Analysis",
                        player_ids,
                        format_func=lambda x: f"Player {x} ({'Your Team' if tracking_data[x].get('team_id') == 0 else 'Opposition' if tracking_data[x].get('team_id') == 1 else 'Official'})",
                        help="Choose a player for detailed performance analysis"
                    )
                    st.session_state['selected_player'] = selected_player
                
                # Quick insights
                st.header("📊 Quick Insights")
                total_players = len(player_ids)
                your_team_count = len([p for p in player_ids if tracking_data[p].get('team_id') == 0])
                opp_team_count = len([p for p in player_ids if tracking_data[p].get('team_id') == 1])
                
                st.metric("🔵 Your Players", your_team_count)
                st.metric("🔴 Opposition", opp_team_count)
                st.metric("👥 Total Tracked", total_players)
    
    # Main content area
    if 'soccer_data' in st.session_state:
        data = st.session_state['soccer_data']
        
        # Match overview
        display_match_overview(data)
        
        # Main analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏟️ **TACTICAL FIELD**", "👤 **PLAYER FOCUS**", 
            "⚔️ **TEAM COMPARISON**", "📊 **MATCH DATA**"
        ])
        
        with tab1:
            st.subheader("🎯 Live Tactical View")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if 'selected_frame' in st.session_state:
                    fig = visualize_player_positions_enhanced(data, st.session_state['selected_frame'])
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 🧠 Tactical Options")
                
                highlight_zones = st.multiselect(
                    "Highlight Field Zones",
                    ["attacking", "midfield", "defensive"],
                    help="Overlay tactical zones on field view"
                )
                
                if st.button("🔄 Refresh View"):
                    st.rerun()
                
                # Quick tactical insights
                frame_num = st.session_state.get('selected_frame', 0)
                players_data = data.get('raw_detections', {}).get('players', [])
                
                if frame_num < len(players_data):
                    frame_data = players_data[frame_num]
                    detections = frame_data.get('detections', [])
                    
                    your_team_players = len([d for d in detections if d.get('team_id') == 0])
                    opp_players = len([d for d in detections if d.get('team_id') == 1])
                    
                    st.metric("🔵 Your Players", your_team_players)
                    st.metric("🔴 Opposition", opp_players)
                    
                    if your_team_players > opp_players:
                        st.success("✅ Numerical advantage!")
                    elif your_team_players < opp_players:
                        st.warning("⚠️ Outnumbered in frame")
        
        with tab2:
            st.subheader("👤 Individual Player Analysis")
            
            if 'selected_player' in st.session_state:
                create_player_performance_dashboard(data, st.session_state['selected_player'])
            else:
                st.info("👆 Select a player in the sidebar to view detailed analysis")
        
        with tab3:
            st.subheader("⚔️ Team vs Team Analysis")
            create_team_comparison_dashboard(data)
        
        with tab4:
            st.subheader("📊 Raw Match Data Explorer")
            
            data_section = st.selectbox(
                "Select Data Section",
                ['match_metadata', 'quality_metrics', 'tracking_data', 'spatial_analytics'],
                help="Explore the raw data structure"
            )
            
            if data_section in data:
                # Show summary first
                section_data = data[data_section]
                
                if data_section == 'tracking_data':
                    players = section_data.get('players', {})
                    st.info(f"📊 **{len(players)} players** tracked with movement data")
                    
                elif data_section == 'match_metadata':
                    st.info(f"📹 **Match Duration:** {section_data.get('duration_seconds', 0)/60:.1f} minutes")
                
                # Show raw data with expandable sections
                with st.expander(f"View Raw {data_section.replace('_', ' ').title()} Data"):
                    st.json(section_data)
            else:
                st.error(f"❌ Section '{data_section}' not found in data")
    
    else:
        # Welcome screen for coaches
        st.markdown("""
        ## 👋 Welcome, Coach!
        
        This dashboard helps you analyze your team's performance using advanced soccer analytics.
        
        ### 🎯 What You Can Analyze:
        
        **🏟️ Tactical Field View**
        - See where your players are positioned in real-time
        - Compare your formation vs opposition
        - Identify tactical advantages and weaknesses
        
        **👤 Individual Player Analysis**  
        - Track each player's movement and work rate
        - See heat maps showing where players spend time
        - Get performance insights and coaching recommendations
        
        **⚔️ Team Comparison**
        - Compare your team's metrics vs opposition
        - Analyze zone control and possession patterns
        - Get tactical insights for in-game adjustments
        
        ### 📁 Getting Started:
        1. Upload your match analytics JSON file (generated by RADAR mode)
        2. Use the timeline slider to scrub through the match
        3. Select individual players for detailed analysis
        4. Review tactical insights and coaching recommendations
        
        ### 🎥 Data Requirements:
        Your JSON file should be generated using the `main.py` script in **RADAR mode**:
        ```bash
        python main.py --source your_video.mp4 --device cpu --mode RADAR
        ```
        
        **Ready to analyze your team's performance? Upload your match data! ⚽**
        """)

if __name__ == "__main__":
    main() 