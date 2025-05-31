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

# Page configuration
st.set_page_config(
    page_title="⚽ Soccer Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1e88e5;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def load_and_validate_json(uploaded_file):
    """Load and validate JSON file according to our schema"""
    try:
        data = json.load(uploaded_file)
        
        # Validate required top-level keys
        required_keys = [
            'match_metadata', 'raw_detections', 'tracking_data',
            'spatial_analytics', 'quality_metrics'
        ]
        
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            st.error(f"Missing required keys: {missing_keys}")
            return None
        
        st.success("✅ JSON file loaded and validated successfully!")
        return data
        
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON file format")
        return None
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

def display_match_metadata(data):
    """Display match metadata and quality metrics"""
    metadata = data['match_metadata']
    quality = data['quality_metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📹 Video Duration", 
            f"{metadata.get('duration_seconds', 0):.0f}s",
            f"{metadata.get('total_frames', 0)} frames"
        )
    
    with col2:
        st.metric(
            "🎯 Player Detection", 
            f"{quality.get('player_detection_rate', 0):.1%}",
            f"FPS: {metadata.get('fps', 30)}"
        )
    
    with col3:
        st.metric(
            "⚽ Ball Detection", 
            f"{quality.get('ball_detection_rate', 0):.1%}",
            f"Tracking: {quality.get('tracking_stability', 0):.1%}"
        )
    
    with col4:
        st.metric(
            "🏟️ Field Analysis", 
            f"{quality.get('keypoint_accuracy', 0):.1%}",
            f"{metadata.get('resolution', {}).get('width', 0)}x{metadata.get('resolution', {}).get('height', 0)}"
        )

def create_field_layout():
    """Create soccer field layout for visualizations"""
    # Standard soccer field dimensions (in cm, converted to meters)
    field_length = 120  # 12000cm = 120m
    field_width = 80    # 8000cm = 80m
    
    fig = go.Figure()
    
    # Field outline
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=field_length, y1=field_width,
        line=dict(color="white", width=2),
        fillcolor="green",
        opacity=0.3
    )
    
    # Center line
    fig.add_shape(
        type="line",
        x0=field_length/2, y0=0, x1=field_length/2, y1=field_width,
        line=dict(color="white", width=2)
    )
    
    # Center circle
    fig.add_shape(
        type="circle",
        x0=field_length/2-9.15, y0=field_width/2-9.15,
        x1=field_length/2+9.15, y1=field_width/2+9.15,
        line=dict(color="white", width=2)
    )
    
    # Penalty boxes
    # Left penalty box
    fig.add_shape(
        type="rect",
        x0=0, y0=field_width/2-20.15, x1=16.5, y1=field_width/2+20.15,
        line=dict(color="white", width=2)
    )
    
    # Right penalty box
    fig.add_shape(
        type="rect",
        x0=field_length-16.5, y0=field_width/2-20.15, 
        x1=field_length, y1=field_width/2+20.15,
        line=dict(color="white", width=2)
    )
    
    # Goal boxes
    # Left goal box
    fig.add_shape(
        type="rect",
        x0=0, y0=field_width/2-9.16, x1=5.5, y1=field_width/2+9.16,
        line=dict(color="white", width=2)
    )
    
    # Right goal box
    fig.add_shape(
        type="rect",
        x0=field_length-5.5, y0=field_width/2-9.16, 
        x1=field_length, y1=field_width/2+9.16,
        line=dict(color="white", width=2)
    )
    
    fig.update_layout(
        xaxis=dict(range=[0, field_length], showgrid=False, zeroline=False),
        yaxis=dict(range=[0, field_width], showgrid=False, zeroline=False),
        showlegend=True,
        plot_bgcolor='darkgreen',
        paper_bgcolor='white',
        height=400
    )
    
    return fig

def visualize_player_positions(data, selected_frame=0):
    """Visualize player positions at a specific frame"""
    fig = create_field_layout()
    
    # Get player data for the selected frame
    players_data = data['raw_detections']['players']
    
    if selected_frame < len(players_data):
        frame_data = players_data[selected_frame]
        
        team_colors = {0: 'blue', 1: 'red', 2: 'yellow'}  # team_0, team_1, referees
        
        for detection in frame_data['detections']:
            if detection['position_field']['x'] > 0:  # Valid position
                # Convert cm to meters
                x = detection['position_field']['x'] / 100
                y = detection['position_field']['y'] / 100
                
                team_id = detection.get('team_id', -1)
                color = team_colors.get(team_id, 'gray')
                
                # Player marker
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color=color,
                        symbol='circle',
                        line=dict(width=2, color='white')
                    ),
                    text=[str(detection.get('tracker_id', '?'))],
                    textposition='middle center',
                    textfont=dict(color='white', size=8),
                    name=f"Team {team_id}" if team_id >= 0 else "Unknown",
                    showlegend=team_id >= 0
                ))
    
    fig.update_layout(title=f"Player Positions - Frame {selected_frame}")
    return fig

def create_possession_timeline(data):
    """Create ball possession timeline"""
    # Extract ball data
    ball_data = data['raw_detections']['ball']
    
    timestamps = []
    possession_team = []
    
    for frame_data in ball_data:
        if frame_data['detections']:
            timestamps.append(frame_data['timestamp'])
            # Simple possession detection based on closest player
            # This is a simplified version - you'd want more sophisticated logic
            possession_team.append(0)  # Placeholder
    
    if timestamps:
        df = pd.DataFrame({
            'timestamp': timestamps,
            'possession': possession_team
        })
        
        fig = px.line(df, x='timestamp', y='possession', 
                     title='Ball Possession Timeline',
                     labels={'timestamp': 'Time (seconds)', 'possession': 'Team'})
        
        fig.update_layout(
            yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Team 0', 'Team 1'])
        )
        
        return fig
    
    return go.Figure()

def create_player_heatmap(data, player_id):
    """Create heat map for a specific player"""
    # Extract player tracking data
    tracking_data = data['tracking_data']['players']
    
    if str(player_id) in tracking_data:
        player_data = tracking_data[str(player_id)]
        trajectory = player_data['trajectory']
        
        # Extract positions (convert cm to meters)
        positions = [(point['position_field']['x']/100, point['position_field']['y']/100) 
                    for point in trajectory if point['position_field']['x'] > 0]
        
        if positions:
            x_pos, y_pos = zip(*positions)
            
            # Create heatmap
            fig = create_field_layout()
            
            # Add heatmap
            fig.add_trace(go.Histogram2d(
                x=x_pos, y=y_pos,
                nbinsx=30, nbinsy=20,
                colorscale='Reds',
                opacity=0.7,
                name=f'Player {player_id} Heatmap'
            ))
            
            fig.update_layout(title=f"Player {player_id} Movement Heatmap")
            return fig
    
    return create_field_layout()

def create_team_statistics_dashboard(data):
    """Create team statistics comparison"""
    # Extract summary statistics
    summary = data.get('summary_statistics', {})
    match_summary = summary.get('match_summary', {})
    
    # Create metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔵 Team 0 Statistics")
        if 'possession' in match_summary:
            st.metric("Possession %", f"{match_summary['possession'].get('team_0', 0):.1f}%")
        if 'total_passes' in match_summary:
            st.metric("Total Passes", match_summary['total_passes'].get('team_0', 0))
        if 'distance_covered' in match_summary:
            st.metric("Distance Covered (m)", f"{match_summary['distance_covered'].get('team_0', 0):.0f}")
    
    with col2:
        st.subheader("🔴 Team 1 Statistics")
        if 'possession' in match_summary:
            st.metric("Possession %", f"{match_summary['possession'].get('team_1', 0):.1f}%")
        if 'total_passes' in match_summary:
            st.metric("Total Passes", match_summary['total_passes'].get('team_1', 0))
        if 'distance_covered' in match_summary:
            st.metric("Distance Covered (m)", f"{match_summary['distance_covered'].get('team_1', 0):.0f}")

def create_zone_occupancy_timeline(data):
    """Create zone occupancy over time"""
    spatial_data = data['spatial_analytics']['per_frame_occupancy']
    
    if spatial_data:
        # Prepare data for visualization
        timestamps = []
        defensive_third_team0 = []
        middle_third_team0 = []
        attacking_third_team0 = []
        
        for frame_data in spatial_data:
            timestamps.append(frame_data['timestamp'])
            zones = frame_data['zone_occupancy']
            defensive_third_team0.append(zones['defensive_third']['team_0'])
            middle_third_team0.append(zones['middle_third']['team_0'])
            attacking_third_team0.append(zones['attacking_third']['team_0'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps, y=defensive_third_team0,
            mode='lines', name='Defensive Third',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps, y=middle_third_team0,
            mode='lines', name='Middle Third',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps, y=attacking_third_team0,
            mode='lines', name='Attacking Third',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Team 0 Zone Occupancy Over Time',
            xaxis_title='Time (seconds)',
            yaxis_title='Number of Players',
            hovermode='x unified'
        )
        
        return fig
    
    return go.Figure()

def main():
    # Header
    st.markdown('<div class="main-header">⚽ Soccer Analytics Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar for file upload and controls
    with st.sidebar:
        st.header("📁 Upload Analytics Data")
        uploaded_file = st.file_uploader(
            "Choose JSON analytics file",
            type=['json'],
            help="Upload the JSON file generated by your soccer analysis script"
        )
        
        if uploaded_file is not None:
            data = load_and_validate_json(uploaded_file)
            
            if data:
                # Store data in session state
                st.session_state['soccer_data'] = data
                
                st.success(f"✅ Loaded: {data['match_metadata'].get('video_file', 'Unknown')}")
                
                # Timeline controls
                st.header("⏱️ Timeline Controls")
                max_frames = len(data['raw_detections']['players'])
                
                selected_frame = st.slider(
                    "Select Frame",
                    min_value=0,
                    max_value=max_frames-1,
                    value=0,
                    help="Scrub through the match timeline"
                )
                
                st.session_state['selected_frame'] = selected_frame
                
                # Player selection for detailed analysis
                st.header("👤 Player Analysis")
                tracking_data = data['tracking_data']['players']
                player_ids = list(tracking_data.keys())
                
                if player_ids:
                    selected_player = st.selectbox(
                        "Select Player",
                        player_ids,
                        help="Choose a player for detailed analysis"
                    )
                    st.session_state['selected_player'] = selected_player
    
    # Main content area
    if 'soccer_data' in st.session_state:
        data = st.session_state['soccer_data']
        
        # Match metadata and quality metrics
        st.header("📊 Match Overview")
        display_match_metadata(data)
        
        # Team statistics
        st.header("📈 Team Statistics")
        create_team_statistics_dashboard(data)
        
        # Main visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏟️ Field View", "📍 Player Heatmap", "⚽ Ball Analysis", 
            "📊 Zone Analysis", "📋 Raw Data"
        ])
        
        with tab1:
            st.subheader("Player Positions on Field")
            if 'selected_frame' in st.session_state:
                fig = visualize_player_positions(data, st.session_state['selected_frame'])
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Ball Possession Timeline")
                possession_fig = create_possession_timeline(data)
                st.plotly_chart(possession_fig, use_container_width=True)
            
            with col2:
                st.subheader("Zone Occupancy Timeline")
                zone_fig = create_zone_occupancy_timeline(data)
                st.plotly_chart(zone_fig, use_container_width=True)
        
        with tab2:
            st.subheader("Player Movement Heatmap")
            if 'selected_player' in st.session_state:
                heatmap_fig = create_player_heatmap(data, st.session_state['selected_player'])
                st.plotly_chart(heatmap_fig, use_container_width=True)
                
                # Player statistics
                player_data = data['tracking_data']['players'].get(st.session_state['selected_player'], {})
                if player_data:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Frames Tracked", player_data.get('total_frames_tracked', 0))
                    with col2:
                        st.metric("Team ID", player_data.get('team_id', 'Unknown'))
                    with col3:
                        st.metric("Role", player_data.get('role', 'Unknown'))
        
        with tab3:
            st.subheader("Ball Detection Analysis")
            ball_stats = data['tracking_data']['ball']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Frames Detected", ball_stats.get('total_frames_detected', 0))
            with col2:
                st.metric("Detection Rate", f"{ball_stats.get('detection_rate', 0):.1%}")
            with col3:
                st.metric("Trajectory Points", len(ball_stats.get('trajectory', [])))
            
            # Ball trajectory visualization
            if ball_stats.get('trajectory'):
                trajectory = ball_stats['trajectory']
                times = [point['timestamp'] for point in trajectory]
                velocities = [point['velocity']['magnitude'] for point in trajectory]
                
                fig = px.line(x=times, y=velocities, 
                             title='Ball Velocity Over Time',
                             labels={'x': 'Time (seconds)', 'y': 'Velocity (m/s)'})
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Spatial Zone Analysis")
            
            # Show field zones definition
            zones = data['spatial_analytics']['field_zones']
            
            zone_df = pd.DataFrame([
                {'Zone': 'Defensive Third', 'X Min': 0, 'X Max': 40, 'Area (m²)': 2800},
                {'Zone': 'Middle Third', 'X Min': 40, 'X Max': 80, 'Area (m²)': 2800},
                {'Zone': 'Attacking Third', 'X Min': 80, 'X Max': 120, 'Area (m²)': 2800},
            ])
            
            st.dataframe(zone_df, use_container_width=True)
        
        with tab5:
            st.subheader("Raw Data Explorer")
            
            # Allow users to explore the raw JSON structure
            section = st.selectbox(
                "Select Data Section",
                ['match_metadata', 'quality_metrics', 'tracking_data', 'spatial_analytics']
            )
            
            if section in data:
                st.json(data[section])
    
    else:
        # Welcome screen
        st.info("""
        👋 **Welcome to the Soccer Analytics Dashboard!**
        
        📁 **Upload your JSON analytics file** in the sidebar to get started.
        
        🎯 **Features:**
        - Interactive field visualization with player positions
        - Player movement heatmaps and trajectory analysis  
        - Ball possession and trajectory tracking
        - Zone occupancy analysis over time
        - Team statistics comparison
        - Timeline scrubbing through match events
        
        📊 **Generated by your `main.py` script using RADAR mode**
        """)

if __name__ == "__main__":
    main() 