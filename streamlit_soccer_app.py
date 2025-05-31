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
    """Load and validate comprehensive match data"""
    try:
        content = uploaded_file.read()
        data = json.loads(content)
        
        # Validate comprehensive data structure
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
        for section, default in required_sections.items():
            if section not in data:
                data[section] = default
                st.warning(f"⚠️ {section.replace('_', ' ').title()} data not found - using defaults")
        
        return data
        
    except json.JSONDecodeError as e:
        st.error(f"❌ Invalid JSON format: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

def safe_get(data: Dict, path: str, default=None):
    """Safely get nested data with dot notation"""
    keys = path.split('.')
    current = data
    try:
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key, {})
            else:
                return default
        return current if current != {} else default
    except:
        return default

def format_time(seconds: float) -> str:
    """Format seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"

def get_quality_class(value: float, thresholds: List[float]) -> str:
    """Get CSS class for quality indicators"""
    if value >= thresholds[0]:
        return "quality-excellent"
    elif value >= thresholds[1]:
        return "quality-good"
    elif value >= thresholds[2]:
        return "quality-fair"
    else:
        return "quality-poor"

def create_match_header(data: Dict):
    """Create professional match header"""
    metadata = safe_get(data, 'match_metadata', {})
    quality = safe_get(data, 'quality_metrics', {})
    
    duration = metadata.get('duration_seconds', 0)
    video_file = metadata.get('video_file', 'Match Analysis')
    
    st.markdown(f"""
    <div class="analytics-header">
        <div class="analytics-title">⚽ {video_file}</div>
        <div class="analytics-subtitle">
            Duration: {format_time(duration)} | 
            Processed: {datetime.now().strftime('%Y-%m-%d %H:%M')} |
            Quality: {quality.get('player_detection_rate', 0)*100:.0f}% Detection Rate
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_overview_metrics(data: Dict):
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
    """Create timeline navigation controls"""
    st.markdown('<div class="timeline-controls">', unsafe_allow_html=True)
    
    # Get frame count
    raw_players = safe_get(data, 'raw_detections.players', [])
    max_frames = len(raw_players) if raw_players else 1000
    fps = safe_get(data, 'match_metadata.fps', 30)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🕐 Timeline Navigation")
        frame_number = st.slider(
            "Select Match Moment",
            min_value=0,
            max_value=max_frames-1,
            value=0,
            format="%d",
            help="Scrub through match timeline"
        )
        
        # Display time info
        time_seconds = frame_number / fps
        st.info(f"⏱️ **{format_time(time_seconds)}** | Frame {frame_number:,}/{max_frames:,}")
    
    with col2:
        st.subheader("🔍 Quick Jump")
        jump_options = {
            "Match Start": 0,
            "First Quarter": max_frames // 4,
            "Half Time": max_frames // 2,
            "Third Quarter": 3 * max_frames // 4,
            "Match End": max_frames - 1
        }
        
        for label, frame in jump_options.items():
            if st.button(label, key=f"jump_{frame}"):
                st.session_state.selected_frame = frame
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    return frame_number

def create_field_visualization(data: Dict, frame_number: int):
    """Create interactive field visualization"""
    # Get frame data
    raw_players = safe_get(data, 'raw_detections.players', [])
    
    if not raw_players or frame_number >= len(raw_players):
        st.warning("⚠️ No player data available for this time")
        return
    
    frame_data = raw_players[frame_number]
    detections = frame_data.get('detections', [])
    
    # Create field layout
    fig = go.Figure()
    
    # Field dimensions (120m x 80m)
    field_length, field_width = 120, 80
    
    # Field background
    fig.add_shape(
        type="rect", x0=0, y0=0, x1=field_length, y1=field_width,
        fillcolor="#1a472a", line=dict(color="white", width=2), opacity=0.8
    )
    
    # Field markings
    markings = [
        {"type": "line", "x0": 60, "y0": 0, "x1": 60, "y1": 80},  # Center line
        {"type": "circle", "x0": 50.85, "y0": 30.85, "x1": 69.15, "y1": 49.15},  # Center circle
        {"type": "rect", "x0": 0, "y0": 20.15, "x1": 16.5, "y1": 59.85},  # Left penalty box
        {"type": "rect", "x0": 103.5, "y0": 20.15, "x1": 120, "y1": 59.85},  # Right penalty box
    ]
    
    for marking in markings:
        fig.add_shape(line=dict(color="white", width=2), **marking)
    
    # Team colors and tracking
    team_colors = {0: "#3b82f6", 1: "#ef4444", 2: "#fbbf24"}  # Blue, Red, Yellow
    team_names = {0: "Team A", 1: "Team B", 2: "Officials"}
    team_counts = {0: 0, 1: 0, 2: 0}
    
    # Plot players
    for detection in detections:
        if detection.get('position_field', {}).get('x', 0) > 0:
            x = detection['position_field']['x'] / 100  # Convert cm to meters
            y = detection['position_field']['y'] / 100
            
            team_id = detection.get('team_id', -1)
            if team_id in team_colors:
                team_counts[team_id] += 1
                player_id = detection.get('tracker_id', '?')
                
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(size=14, color=team_colors[team_id], 
                              line=dict(width=2, color='white')),
                    text=[str(player_id)],
                    textfont=dict(color='white', size=9),
                    name=team_names[team_id],
                    showlegend=team_id in [0, 1],
                    hovertemplate=f"<b>{team_names[team_id]} #{player_id}</b><br>" +
                                f"Position: ({x:.1f}m, {y:.1f}m)<extra></extra>"
                ))
    
    fig.update_layout(
        title=f"⚽ Live Match View - {format_time(frame_number/30)}",
        xaxis=dict(range=[0, field_length], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, field_width], showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#1a472a',
        paper_bgcolor='white',
        height=400,
        showlegend=True
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Live Stats")
        st.metric("Team A Players", team_counts[0])
        st.metric("Team B Players", team_counts[1])
        st.metric("Officials", team_counts[2])
        
        if team_counts[0] > team_counts[1]:
            st.success("🔵 Team A Advantage")
        elif team_counts[1] > team_counts[0]:
            st.error("🔴 Team B Advantage")
        else:
            st.info("⚖️ Equal Numbers")

def create_possession_timeline(data: Dict):
    """Create possession timeline visualization"""
    possession_segments = safe_get(data, 'possession_analytics.possession_segments', [])
    
    if not possession_segments:
        st.warning("⚠️ No possession data available")
        return
    
    # Prepare timeline data
    timeline_data = []
    for segment in possession_segments:
        timeline_data.append({
            'start': segment.get('start_frame', 0) / 30,  # Convert to seconds
            'duration': segment.get('duration', 0),
            'team': f"Team {'A' if segment.get('team_id') == 0 else 'B'}",
            'team_id': segment.get('team_id', 0)
        })
    
    if timeline_data:
        df = pd.DataFrame(timeline_data)
        
        # Create Gantt-style timeline
        fig = go.Figure()
        
        colors = {'Team A': '#3b82f6', 'Team B': '#ef4444'}
        
        for i, row in df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['start'], row['start'] + row['duration']],
                y=[row['team'], row['team']],
                mode='lines',
                line=dict(color=colors[row['team']], width=8),
                name=row['team'],
                showlegend=i == 0 or (i > 0 and row['team'] != df.iloc[i-1]['team']),
                hovertemplate=f"<b>{row['team']}</b><br>" +
                            f"Time: {row['start']:.1f}s - {row['start'] + row['duration']:.1f}s<br>" +
                            f"Duration: {row['duration']:.1f}s<extra></extra>"
            ))
        
        fig.update_layout(
            title="⚽ Ball Possession Timeline",
            xaxis_title="Match Time (seconds)",
            yaxis_title="Team",
            height=200,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_player_performance_table(data: Dict):
    """Create player performance rankings table"""
    movement_data = safe_get(data, 'movement_analytics', [])
    
    if not movement_data:
        st.warning("⚠️ No player performance data available")
        return
    
    # Convert to DataFrame
    players_df = pd.DataFrame([
        {
            'Player': f"#{p['player_id']}",
            'Team': 'A' if p['team_id'] == 0 else 'B' if p['team_id'] == 1 else 'Ref',
            'Distance (m)': round(p['movement_stats']['total_distance'], 0),
            'Max Speed (m/s)': round(p['movement_stats']['max_speed'], 1),
            'Avg Speed (m/s)': round(p['movement_stats']['average_speed'], 1),
            'Sprints': p['movement_stats']['sprint_count'],
            'High Intensity': p['movement_stats']['high_intensity_runs']
        }
        for p in movement_data
        if p['team_id'] in [0, 1]  # Exclude referees
    ])
    
    if not players_df.empty:
        # Sort by distance covered
        players_df = players_df.sort_values('Distance (m)', ascending=False)
        
        st.subheader("🏃‍♂️ Player Performance Rankings")
        st.dataframe(
            players_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Distance (m)': st.column_config.ProgressColumn(
                    'Distance (m)',
                    help='Total distance covered',
                    min_value=0,
                    max_value=players_df['Distance (m)'].max(),
                    format='%.0f'
                ),
                'Max Speed (m/s)': st.column_config.ProgressColumn(
                    'Max Speed (m/s)',
                    help='Maximum speed achieved',
                    min_value=0,
                    max_value=15,
                    format='%.1f'
                )
            }
        )

def create_tactical_events_timeline(data: Dict):
    """Create tactical events timeline"""
    pressing_events = safe_get(data, 'tactical_events.pressing_events', [])
    offside_events = safe_get(data, 'tactical_events.offside_events', [])
    
    if not pressing_events and not offside_events:
        st.warning("⚠️ No tactical events detected")
        return
    
    events_data = []
    
    # Add pressing events
    for event in pressing_events:
        events_data.append({
            'time': event.get('frame_start', 0) / 30,
            'type': 'Pressing',
            'team': f"Team {'A' if event.get('team_id') == 0 else 'B'}",
            'intensity': event.get('intensity', 0),
            'details': f"Intensity: {event.get('intensity', 0):.1f}"
        })
    
    # Add offside events
    for event in offside_events:
        events_data.append({
            'time': event.get('frame_id', 0) / 30,
            'type': 'Offside',
            'team': f"Team {'A' if event.get('team_id') == 0 else 'B'}",
            'intensity': 0.8,
            'details': f"Distance: {event.get('offside_distance', 0):.1f}m"
        })
    
    if events_data:
        df = pd.DataFrame(events_data)
        
        fig = px.scatter(
            df, x='time', y='type', color='team',
            size='intensity', size_max=15,
            title='⚡ Tactical Events Timeline',
            labels={'time': 'Match Time (seconds)', 'type': 'Event Type'},
            color_discrete_map={'Team A': '#3b82f6', 'Team B': '#ef4444'}
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def create_coaching_insights(data: Dict, frame_number: int):
    """Generate AI coaching insights"""
    possession_stats = safe_get(data, 'possession_analytics.possession_stats', {})
    movement_data = safe_get(data, 'movement_analytics', [])
    
    insights = []
    
    # Possession insights
    if possession_stats:
        team_a_poss = possession_stats.get('team_0', {}).get('percentage', 0)
        team_b_poss = possession_stats.get('team_1', {}).get('percentage', 0)
        
        if team_a_poss > 60:
            insights.append("Team A dominating possession - consider defensive counter-attacking strategy for Team B")
        elif team_b_poss > 60:
            insights.append("Team B controlling the game - Team A should focus on pressing and quick transitions")
        else:
            insights.append("Evenly matched possession - tactical discipline will be key")
    
    # Movement insights
    if movement_data:
        team_a_distances = [p['movement_stats']['total_distance'] for p in movement_data if p['team_id'] == 0]
        team_b_distances = [p['movement_stats']['total_distance'] for p in movement_data if p['team_id'] == 1]
        
        if team_a_distances and team_b_distances:
            avg_a = np.mean(team_a_distances)
            avg_b = np.mean(team_b_distances)
            
            if avg_a > avg_b * 1.1:
                insights.append("Team A showing higher work rate - may lead to fatigue advantage for Team B later")
            elif avg_b > avg_a * 1.1:
                insights.append("Team B covering more ground - consider rotation to maintain intensity")
    
    # Tactical events insights
    pressing_count = len(safe_get(data, 'tactical_events.pressing_events', []))
    if pressing_count > 10:
        insights.append("High pressing intensity detected - monitor player stamina and spacing")
    
    # Display insights
    if insights:
        st.markdown("""
        <div class="insights-panel">
            <div class="insights-title">🧠 AI Coaching Insights</div>
        """, unsafe_allow_html=True)
        
        for insight in insights[:3]:  # Show top 3 insights
            st.markdown(f'<div class="insight-item">{insight}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Professional header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="margin: 0; color: #0f172a;">⚽ Football Analytics Pro</h1>
        <p style="margin: 0.5rem 0 0 0; color: #64748b;">Professional match analysis for coaches and analysts</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload Match Analytics JSON",
        type=['json'],
        help="Upload the comprehensive analytics file generated by RADAR mode",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Load data
        with st.spinner("🔄 Loading match data..."):
            data = load_match_data(uploaded_file)
        
        if data:
            # Store in session state
            st.session_state['match_data'] = data
            
            # Create match header
            create_match_header(data)
            
            # Overview metrics
            create_overview_metrics(data)
            
            # Timeline controls
            frame_number = create_timeline_controls(data)
            st.session_state['selected_frame'] = frame_number
            
            # Main analysis tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🏟️ Live View", "📊 Performance", "⚽ Possession", "⚡ Events", "📈 Analytics"
            ])
            
            with tab1:
                st.subheader("🏟️ Live Match Visualization")
                create_field_visualization(data, frame_number)
                create_coaching_insights(data, frame_number)
            
            with tab2:
                st.subheader("📊 Player Performance Analysis")
                create_player_performance_table(data)
                
                # Additional performance metrics
                col1, col2 = st.columns(2)
                with col1:
                    movement_data = safe_get(data, 'movement_analytics', [])
                    if movement_data:
                        distances = [p['movement_stats']['total_distance'] for p in movement_data if p['team_id'] in [0,1]]
                        fig = px.histogram(distances, nbins=10, title="Distance Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if movement_data:
                        speeds = [p['movement_stats']['max_speed'] for p in movement_data if p['team_id'] in [0,1]]
                        fig = px.box(y=speeds, title="Speed Analysis")
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("⚽ Possession Analysis")
                create_possession_timeline(data)
                
                # Possession stats
                possession_stats = safe_get(data, 'possession_analytics.possession_stats', {})
                if possession_stats:
                    col1, col2 = st.columns(2)
                    with col1:
                        team_a = possession_stats.get('team_0', {})
                        st.metric("Team A Possession", f"{team_a.get('percentage', 0):.1f}%")
                        st.metric("Team A Segments", team_a.get('segments', 0))
                    
                    with col2:
                        team_b = possession_stats.get('team_1', {})
                        st.metric("Team B Possession", f"{team_b.get('percentage', 0):.1f}%")
                        st.metric("Team B Segments", team_b.get('segments', 0))
            
            with tab4:
                st.subheader("⚡ Tactical Events")
                create_tactical_events_timeline(data)
                
                # Event summary
                pressing_events = len(safe_get(data, 'tactical_events.pressing_events', []))
                offside_events = len(safe_get(data, 'tactical_events.offside_events', []))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pressing Events", pressing_events)
                with col2:
                    st.metric("Offside Events", offside_events)
                with col3:
                    formations = len(safe_get(data, 'formation_analytics', []))
                    st.metric("Formation Changes", formations)
            
            with tab5:
                st.subheader("📈 Advanced Analytics")
                
                # Data quality overview
                quality = safe_get(data, 'quality_metrics', {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    player_rate = quality.get('player_detection_rate', 0) * 100
                    quality_class = get_quality_class(player_rate, [90, 80, 70])
                    st.markdown(f"""
                    **Player Detection Quality**  
                    <span class="{quality_class}">{player_rate:.1f}%</span>
                    """, unsafe_allow_html=True)
                
                with col2:
                    ball_rate = quality.get('ball_detection_rate', 0) * 100
                    quality_class = get_quality_class(ball_rate, [85, 70, 55])
                    st.markdown(f"""
                    **Ball Detection Quality**  
                    <span class="{quality_class}">{ball_rate:.1f}%</span>
                    """, unsafe_allow_html=True)
                
                with col3:
                    tracking_rate = quality.get('tracking_stability', 0) * 100
                    quality_class = get_quality_class(tracking_rate, [90, 80, 70])
                    st.markdown(f"""
                    **Tracking Stability**  
                    <span class="{quality_class}">{tracking_rate:.1f}%</span>
                    """, unsafe_allow_html=True)
                
                # Data completeness
                st.subheader("📊 Data Completeness")
                completeness = safe_get(data, 'summary_statistics.data_completeness', {})
                
                if completeness:
                    completion_df = pd.DataFrame([
                        {'Metric': 'Player Tracking', 'Frames': completeness.get('player_tracking_frames', 0)},
                        {'Metric': 'Ball Detection', 'Frames': completeness.get('ball_detection_frames', 0)},
                        {'Metric': 'Spatial Analysis', 'Frames': completeness.get('spatial_analysis_frames', 0)},
                        {'Metric': 'Formation Analysis', 'Frames': completeness.get('formation_analysis_frames', 0)},
                    ])
                    
                    fig = px.bar(completion_df, x='Metric', y='Frames', 
                               title='Data Collection Completeness')
                    st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to Football Analytics Pro
        
        **Professional match analysis platform for coaches and analysts**
        
        ### 🎯 Key Features:
        - **Timeline Analysis**: Scrub through match moments like video analysis software
        - **Live Tactical View**: See player positions and movements in real-time
        - **Performance Metrics**: Individual and team performance analytics
        - **Possession Analysis**: Detailed ball control and team dynamics
        - **Tactical Events**: Pressing, offside, and formation changes
        - **AI Insights**: Automated coaching recommendations
        
        ### 📁 Getting Started:
        1. **Generate Data**: Use `python main.py --mode RADAR` to create comprehensive analytics
        2. **Upload JSON**: Upload your analytics file using the button above
        3. **Analyze**: Navigate through tabs to explore different aspects of the match
        
        ### 🎥 Data Requirements:
        Your JSON file should be generated using RADAR mode for complete analysis:
        ```bash
        python main.py --source match.mp4 --target output.mp4 --device cuda --mode RADAR
        ```
        
        **Ready to analyze your match? Upload your analytics file above! ⚽**
        """)

if __name__ == "__main__":
    main() 