# 🌍 Geo-Reason

**Enterprise GeoAI Analysis Platform**

Geo-Reason is an advanced AI-powered system that uses Large Language Models (LLMs) to autonomously plan and execute complex geospatial analysis tasks based on natural language queries.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io/)

## 🎯 Features

- **Natural Language Queries**: Analyze geospatial data using plain English
- **Chain-of-Thought Reasoning**: Transparent, step-by-step analysis planning
- **ReAct Loop**: Iterative reasoning, action, and observation for robust analysis
- **Secure Code Execution**: Sandboxed environment for safe Python execution
- **RAG Integration**: Documentation-enhanced responses for accurate tool usage
- **Interactive Maps**: Folium-powered visualization of analysis results
- **Enterprise-Ready**: Production-grade logging, configuration, and error handling

## 🏗️ Architecture

```
geo_reason/
├── __init__.py              # Package initialization
├── core/                    # Core modules
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── agent.py            # ReAct agent implementation
│   ├── sandbox.py          # Secure code execution
│   └── tools.py            # GIS tool implementations
├── knowledge/              # RAG and knowledge base
│   ├── __init__.py
│   └── rag_engine.py       # Document retrieval engine
├── prompts/                # LLM prompts
│   ├── __init__.py
│   └── brain.py            # System prompt templates
├── ui/                     # User interface
│   ├── __init__.py
│   └── app.py              # Streamlit application
└── utils/                  # Utilities
    ├── __init__.py
    └── logging.py          # Logging configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for LLM inference)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Anamitra-Sarkar/wings_of_ai.git
   cd wings_of_ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment** (optional)
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

### Running the Application

**Start the Streamlit UI:**
```bash
streamlit run geo_reason/ui/app.py
```

The application will be available at `http://localhost:8501`.

### Using the API

```python
from geo_reason.core.agent import create_agent

# Create an agent (use_mock_llm=True for testing)
agent = create_agent(use_mock_llm=True)

# Run an analysis
result = agent.run("Find areas prone to flooding in Kerala")

# Access results
print(f"Success: {result.success}")
print(f"Steps: {len(result.steps)}")
if result.geojson_output:
    print("GeoJSON output available")
```

## 📖 Example Queries

Here are some example queries you can try:

1. **Flood Risk Analysis**
   ```
   Find areas prone to flooding in Kerala based on proximity to water bodies
   ```

2. **Building Density**
   ```
   Identify building density in Mumbai
   ```

3. **Distance Analysis**
   ```
   Calculate distance to nearest hospital in Delhi
   ```

4. **Feature Extraction**
   ```
   Show all water bodies in Rajasthan
   ```

5. **Land Use Analysis**
   ```
   Analyze land use patterns in Bangalore
   ```

## 🔧 Configuration

Configuration can be set via environment variables or `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | HuggingFace model name | `meta-llama/Llama-3.2-3B-Instruct` |
| `MODEL_DEVICE` | Device for inference | `auto` |
| `MODEL_MAX_TOKENS` | Maximum generation tokens | `4096` |
| `MODEL_TEMPERATURE` | Sampling temperature | `0.1` |
| `USE_QUANTIZATION` | Enable 4-bit quantization | `True` |
| `SANDBOX_TIMEOUT` | Execution timeout (seconds) | `300` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

## 🛡️ Security

Geo-Reason includes multiple security features:

- **Import Whitelist**: Only approved Python modules can be imported
- **Code Analysis**: AST-based analysis blocks dangerous operations
- **Timeout Protection**: Execution is terminated after timeout
- **Process Isolation**: Code runs in separate processes when enabled
- **No Network Access**: Sandbox blocks unauthorized network operations

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=geo_reason --cov-report=html
```

## 📚 API Reference

### GeoReasonAgent

The main agent class for running analyses.

```python
class GeoReasonAgent:
    def run(
        self,
        query: str,
        max_iterations: int = 5,
        callback: Optional[Callable] = None
    ) -> AgentResponse:
        """
        Run the ReAct loop for a given query.
        
        Args:
            query: Natural language geospatial analysis query
            max_iterations: Maximum number of iterations
            callback: Optional callback for step updates
            
        Returns:
            AgentResponse with complete analysis results
        """
```

### GeoTools

Collection of geospatial analysis tools.

```python
class GeoTools:
    @staticmethod
    def load_vector_layer(path, layer=None, bbox=None) -> GeoDataFrame
    
    @staticmethod
    def buffer_geometry(gdf, distance, resolution=16, target_crs=None) -> GeoDataFrame
    
    @staticmethod
    def calculate_raster_stats(raster_path, vector, stats=None) -> DataFrame
    
    def fetch_osm_data(tags, bbox=None, place=None, timeout=180) -> GeoDataFrame
    
    @staticmethod
    def spatial_join(left, right, predicate='intersects', how='inner') -> GeoDataFrame
    
    @staticmethod
    def dissolve(gdf, by=None, aggfunc='first') -> GeoDataFrame
```

### SecureSandbox

Secure code execution environment.

```python
class SecureSandbox:
    def execute(
        self,
        code: str,
        context: Optional[Dict] = None,
        validate: bool = True
    ) -> ExecutionResult:
        """
        Execute Python code in the sandbox.
        
        Args:
            code: Python code to execute
            context: Initial variables/context
            validate: Whether to validate code first
            
        Returns:
            ExecutionResult with execution status and outputs
        """
```

## 🗺️ Data Sources

Geo-Reason supports multiple geospatial data sources:

- **OpenStreetMap (OSM)**: Via OSMnx for vector data
- **User-uploaded Files**: GeoJSON, Shapefile, GeoPackage, etc.
- **Raster Data**: GeoTIFF and other formats via Rasterio

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for agent framework inspiration
- [GeoPandas](https://geopandas.org/) for geospatial data handling
- [Streamlit](https://streamlit.io/) for the web interface
- [Folium](https://python-visualization.github.io/folium/) for map visualization
- [HuggingFace](https://huggingface.co/) for transformer models

## 📧 Support

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for the geospatial community**
