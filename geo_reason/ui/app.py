"""
Geo-Reason Streamlit Application.

Enterprise-grade web interface for the Geo-Reason GeoAI platform.

Features:
- Natural language query input
- Chain-of-thought visualization
- Interactive map display with Folium
- Analysis history and export
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

# Configure page before other imports
st.set_page_config(
    page_title="Geo-Reason | GeoAI Analysis Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

from loguru import logger

# Conditional imports with fallbacks
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    logger.warning("Folium not available. Map visualization will be limited.")

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False


# Custom CSS for enterprise look
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .stTitle {
        color: #1e3a5f;
    }
    
    /* Query input styling */
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 10px;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    /* Card styling */
    .info-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1e3a5f;
    }
    
    /* Step indicator */
    .step-indicator {
        background-color: #e8f4f8;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
    
    /* Success/Error badges */
    .success-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    
    .error-badge {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
    }
</style>
"""


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    if "use_mock" not in st.session_state:
        st.session_state.use_mock = True  # Default to mock for safety


def get_agent():
    """Get or create the GeoReason agent."""
    if st.session_state.agent is None:
        try:
            from geo_reason.core.agent import create_agent
            st.session_state.agent = create_agent(
                use_mock_llm=st.session_state.use_mock
            )
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            st.error(f"Failed to initialize agent: {e}")
            return None
    return st.session_state.agent


def create_map(
    geojson_data: Optional[Dict] = None,
    center: Optional[List[float]] = None,
    zoom: int = 5
) -> "folium.Map":
    """
    Create a Folium map with optional GeoJSON overlay.
    
    Args:
        geojson_data: GeoJSON data to display
        center: Map center [lat, lon]
        zoom: Initial zoom level
        
    Returns:
        Folium Map object
    """
    if not FOLIUM_AVAILABLE:
        return None
    
    # Default center on India
    center = center or [20.5937, 78.9629]
    
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap"
    )
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add GeoJSON if provided
    if geojson_data:
        try:
            def geojson_style(feature):
                """Style function for GeoJSON features."""
                return {
                    'fillColor': '#3388ff',
                    'color': '#0000ff',
                    'weight': 2,
                    'fillOpacity': 0.3
                }
            
            geojson_layer = folium.GeoJson(
                geojson_data,
                name="Analysis Result",
                style_function=geojson_style,
                tooltip=folium.GeoJsonTooltip(
                    fields=list(geojson_data.get('features', [{}])[0].get('properties', {}).keys())[:5] if geojson_data.get('features') else [],
                    aliases=[f.replace('_', ' ').title() for f in list(geojson_data.get('features', [{}])[0].get('properties', {}).keys())[:5]] if geojson_data.get('features') else []
                )
            )
            geojson_layer.add_to(m)
            
            # Fit bounds to GeoJSON
            if geojson_data.get('features'):
                bounds = folium.GeoJson(geojson_data).get_bounds()
                if bounds:
                    m.fit_bounds(bounds)
                    
        except Exception as e:
            logger.warning(f"Error adding GeoJSON to map: {e}")
    
    return m


def render_sidebar() -> Dict[str, Any]:
    """
    Render the sidebar with settings and history.
    
    Returns:
        Dict with sidebar settings
    """
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Model settings
        with st.expander("🤖 Model Configuration", expanded=False):
            use_mock = st.checkbox(
                "Use Mock LLM (for testing)",
                value=st.session_state.use_mock,
                help="Use a mock LLM that returns sample responses"
            )
            if use_mock != st.session_state.use_mock:
                st.session_state.use_mock = use_mock
                st.session_state.agent = None  # Reset agent
            
            st.text_input(
                "Model Name",
                value="meta-llama/Llama-3.2-3B-Instruct",
                disabled=True,
                help="HuggingFace model for inference"
            )
            
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.1,
                disabled=True
            )
        
        # Analysis settings
        with st.expander("📊 Analysis Settings", expanded=False):
            max_iterations = st.slider(
                "Max Iterations",
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum ReAct loop iterations"
            )
            
            timeout = st.slider(
                "Timeout (seconds)",
                min_value=60,
                max_value=600,
                value=300,
                step=60
            )
        
        st.markdown("---")
        
        # Analysis history
        st.markdown("## 📜 History")
        
        if st.session_state.analysis_history:
            for i, entry in enumerate(reversed(st.session_state.analysis_history[-5:])):
                with st.expander(f"Query {len(st.session_state.analysis_history) - i}", expanded=False):
                    st.markdown(f"**Query:** {entry['query'][:50]}...")
                    st.markdown(f"**Time:** {entry['time']}")
                    st.markdown(f"**Status:** {'✅' if entry['success'] else '❌'}")
        else:
            st.info("No analysis history yet")
        
        if st.button("Clear History"):
            st.session_state.analysis_history = []
            st.rerun()
        
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666; font-size: 0.8rem;'>
            Geo-Reason v1.0.0<br>
            Enterprise GeoAI Platform
            </div>
            """,
            unsafe_allow_html=True
        )
        
        return {
            "max_iterations": max_iterations,
            "timeout": timeout,
            "use_mock": use_mock
        }


def render_thinking_process(steps: List) -> None:
    """
    Render the chain-of-thought thinking process.
    
    Args:
        steps: List of AgentStep objects
    """
    st.markdown("### 🧠 Thinking Process")
    
    for step in steps:
        with st.expander(f"Step {step.step_number}", expanded=step.step_number == len(steps)):
            # Thought section
            if step.thought:
                st.markdown("**💭 Thought:**")
                st.markdown(f"```\n{step.thought}\n```")
            
            # Plan section
            if step.plan:
                st.markdown("**📋 Plan:**")
                st.json(step.plan)
            
            # Code section
            if step.code:
                st.markdown("**💻 Code:**")
                st.code(step.code, language="python")
            
            # Execution result
            if step.execution_result:
                if step.execution_result.success:
                    st.markdown(
                        '<span class="success-badge">✅ Execution Successful</span>',
                        unsafe_allow_html=True
                    )
                    if step.execution_result.output:
                        st.markdown("**Output:**")
                        st.text(step.execution_result.output[:500])
                else:
                    st.markdown(
                        '<span class="error-badge">❌ Execution Failed</span>',
                        unsafe_allow_html=True
                    )
                    st.error(step.execution_result.error[:500])
            
            # Explanation
            if step.explanation:
                st.markdown("**📝 Explanation:**")
                st.markdown(step.explanation)


def render_results(result) -> None:
    """
    Render analysis results with map visualization.
    
    Args:
        result: AgentResponse object
    """
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Analysis Results")
        
        # Status card
        if result.success:
            st.success(f"Analysis completed successfully in {result.total_time:.2f}s")
        else:
            st.error(f"Analysis failed: {result.error}")
        
        # Final result
        if result.final_result:
            st.markdown("**Result:**")
            if isinstance(result.final_result, str):
                st.text(result.final_result)
            else:
                st.json(result.final_result if isinstance(result.final_result, (dict, list)) else str(result.final_result))
        
        # Export options
        st.markdown("### 📤 Export")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if result.geojson_output:
                geojson_str = json.dumps(result.geojson_output, indent=2)
                st.download_button(
                    "📥 Download GeoJSON",
                    geojson_str,
                    file_name=f"geo_reason_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson",
                    mime="application/geo+json"
                )
        
        with export_col2:
            report = {
                "query": result.query,
                "success": result.success,
                "total_time": result.total_time,
                "steps": len(result.steps),
                "timestamp": datetime.now().isoformat()
            }
            st.download_button(
                "📥 Download Report",
                json.dumps(report, indent=2),
                file_name=f"geo_reason_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        st.markdown("### 🗺️ Map Visualization")
        
        if FOLIUM_AVAILABLE:
            m = create_map(
                geojson_data=result.geojson_output,
                zoom=5
            )
            st_folium(m, width=None, height=500)
        else:
            st.warning("Folium is not installed. Map visualization unavailable.")
            if result.geojson_output:
                st.json(result.geojson_output)


def render_example_queries() -> Optional[str]:
    """
    Render example query buttons.
    
    Returns:
        Selected example query or None
    """
    st.markdown("#### 💡 Example Queries")
    
    examples = [
        "Find areas prone to flooding in Kerala",
        "Identify building density in Mumbai",
        "Calculate distance to nearest hospital in Delhi",
        "Show all water bodies in Rajasthan",
        "Analyze land use patterns in Bangalore"
    ]
    
    cols = st.columns(len(examples))
    
    for i, (col, example) in enumerate(zip(cols, examples)):
        with col:
            if st.button(f"📍 {example.split()[0]}...", key=f"example_{i}", use_container_width=True):
                return example
    
    return None


def main():
    """Main application entry point."""
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    settings = render_sidebar()
    
    # Main content area
    st.markdown(
        """
        # 🌍 Geo-Reason
        ### Autonomous GeoAI Analysis Platform
        
        Enter a natural language query to perform complex geospatial analysis.
        The system uses Chain-of-Thought reasoning to plan and execute analysis tasks.
        """
    )
    
    st.markdown("---")
    
    # Example queries
    example_query = render_example_queries()
    
    # Query input
    query = st.text_area(
        "Enter your geospatial analysis query:",
        value=example_query or "",
        height=100,
        placeholder="e.g., Find areas prone to flooding in Kerala based on proximity to water bodies",
        key="query_input"
    )
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        analyze_button = st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True
        )
    
    # Run analysis
    if analyze_button and query:
        with st.spinner("🔄 Running analysis..."):
            agent = get_agent()
            
            if agent:
                # Progress indicator
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(step):
                    progress = min(step.step_number / settings["max_iterations"], 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Step {step.step_number}: {'✅' if step.execution_result and step.execution_result.success else '⏳'}")
                
                try:
                    result = agent.run(
                        query,
                        max_iterations=settings["max_iterations"],
                        callback=update_progress
                    )
                    
                    # Store in session state
                    st.session_state.current_result = result
                    
                    # Add to history
                    st.session_state.analysis_history.append({
                        "query": query,
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "success": result.success
                    })
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                except Exception as e:
                    logger.error(f"Analysis error: {e}")
                    st.error(f"Analysis failed: {e}")
    
    # Display results
    if st.session_state.current_result:
        st.markdown("---")
        
        result = st.session_state.current_result
        
        # Tabs for different views
        tab1, tab2 = st.tabs(["📊 Results", "🧠 Thinking Process"])
        
        with tab1:
            render_results(result)
        
        with tab2:
            render_thinking_process(result.steps)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        Geo-Reason uses advanced AI to perform geospatial analysis.<br>
        Always verify results for critical applications.
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
