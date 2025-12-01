"""Tests for the agent module."""

import pytest

from geo_reason.core.agent import (
    GeoReasonAgent,
    AgentStep,
    AgentResponse,
    ResponseParser,
    MockLLMInterface,
    create_agent
)


class TestResponseParser:
    """Test cases for ResponseParser."""
    
    def test_parse_complete_response(self):
        """Test parsing a complete LLM response."""
        response = """### THOUGHT:
I need to analyze the flood risk areas.
This requires fetching water body data.

### PLAN:
```json
{
    "task_description": "Flood risk analysis",
    "target_crs": "EPSG:4326"
}
```

### CODE:
```python
import geopandas as gpd
result = gpd.GeoDataFrame()
print("Done")
```

### EXPLANATION:
This code creates an empty GeoDataFrame for testing.
"""
        
        parsed = ResponseParser.parse(response)
        
        assert "flood risk" in parsed["thought"].lower()
        assert parsed["plan"] is not None
        assert parsed["plan"]["task_description"] == "Flood risk analysis"
        assert "geopandas" in parsed["code"]
        assert "GeoDataFrame" in parsed["explanation"]
    
    def test_parse_partial_response(self):
        """Test parsing a partial response."""
        response = """### THOUGHT:
Just a thought without other sections.
"""
        
        parsed = ResponseParser.parse(response)
        
        assert "thought" in parsed["thought"].lower()
        assert parsed["plan"] is None
        assert parsed["code"] == ""
    
    def test_parse_malformed_json(self):
        """Test parsing with malformed JSON."""
        response = """### PLAN:
```json
{not valid json}
```
"""
        
        parsed = ResponseParser.parse(response)
        
        assert parsed["plan"] is None


class TestMockLLMInterface:
    """Test cases for MockLLMInterface."""
    
    def test_mock_generate(self):
        """Test that mock interface generates responses."""
        mock = MockLLMInterface()
        
        messages = [
            {"role": "system", "content": "You are a GeoAI agent."},
            {"role": "user", "content": "Analyze flood risk"}
        ]
        
        response = mock.generate(messages)
        
        assert "THOUGHT" in response
        assert "PLAN" in response
        assert "CODE" in response
        assert "EXPLANATION" in response


class TestAgentStep:
    """Test cases for AgentStep."""
    
    def test_agent_step_creation(self):
        """Test creating an AgentStep."""
        step = AgentStep(
            step_number=1,
            thought="Analyzing the query",
            code="print('hello')"
        )
        
        assert step.step_number == 1
        assert step.thought == "Analyzing the query"
        assert step.code == "print('hello')"
        assert step.execution_result is None


class TestAgentResponse:
    """Test cases for AgentResponse."""
    
    def test_agent_response_creation(self):
        """Test creating an AgentResponse."""
        response = AgentResponse(query="Find flood areas")
        
        assert response.query == "Find flood areas"
        assert response.steps == []
        assert response.success is False
        assert response.final_result is None


class TestGeoReasonAgent:
    """Test cases for GeoReasonAgent."""
    
    def test_agent_initialization(self):
        """Test agent initialization with mock LLM."""
        agent = GeoReasonAgent(use_mock_llm=True)
        
        assert agent.llm is not None
        assert agent.sandbox is not None
        assert agent.tools is not None
    
    def test_agent_run_mock(self):
        """Test running agent with mock LLM."""
        agent = GeoReasonAgent(use_mock_llm=True)
        
        result = agent.run("Find flood areas in Kerala", max_iterations=2)
        
        assert isinstance(result, AgentResponse)
        assert result.query == "Find flood areas in Kerala"
        assert len(result.steps) > 0
        assert result.total_time > 0
    
    def test_agent_reset(self):
        """Test agent state reset."""
        agent = GeoReasonAgent(use_mock_llm=True)
        
        # Add something to history
        agent.conversation_history.append({"role": "user", "content": "test"})
        
        agent.reset()
        
        assert len(agent.conversation_history) == 0


class TestCreateAgent:
    """Test cases for create_agent factory."""
    
    def test_create_agent_mock(self):
        """Test creating agent with mock LLM."""
        agent = create_agent(use_mock_llm=True)
        
        assert isinstance(agent, GeoReasonAgent)
        assert isinstance(agent.llm, MockLLMInterface)
    
    def test_create_agent_with_custom_settings(self):
        """Test creating agent with custom settings."""
        from geo_reason.core.config import Settings
        
        settings = Settings(
            sandbox_timeout=120,
            max_retries=5
        )
        
        agent = create_agent(use_mock_llm=True, settings=settings)
        
        assert agent.settings.sandbox_timeout == 120
        assert agent.settings.max_retries == 5
