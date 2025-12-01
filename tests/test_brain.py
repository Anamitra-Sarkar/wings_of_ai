"""Tests for the brain/prompts module."""

import pytest

from geo_reason.prompts.brain import (
    SystemPrompt,
    get_system_prompt,
    get_tool_context,
    SYSTEM_PROMPT_TEMPLATE
)


class TestSystemPrompt:
    """Test cases for SystemPrompt class."""
    
    def test_system_prompt_creation(self):
        """Test creating a SystemPrompt."""
        prompt = SystemPrompt(
            base_prompt="You are a GeoAI agent."
        )
        
        assert prompt.base_prompt == "You are a GeoAI agent."
        assert prompt.tool_context is None
        assert prompt.rag_context is None
    
    def test_system_prompt_format_base_only(self):
        """Test formatting with base prompt only."""
        prompt = SystemPrompt(
            base_prompt="Base prompt content."
        )
        
        formatted = prompt.format()
        
        assert formatted == "Base prompt content."
    
    def test_system_prompt_format_with_rag(self):
        """Test formatting with RAG context."""
        prompt = SystemPrompt(
            base_prompt="Base prompt.",
            rag_context="Retrieved documentation."
        )
        
        formatted = prompt.format()
        
        assert "Base prompt." in formatted
        assert "RETRIEVED DOCUMENTATION CONTEXT" in formatted
        assert "Retrieved documentation." in formatted
    
    def test_system_prompt_format_with_tools(self):
        """Test formatting with tool context."""
        prompt = SystemPrompt(
            base_prompt="Base prompt.",
            tool_context="Tool documentation."
        )
        
        formatted = prompt.format()
        
        assert "Base prompt." in formatted
        assert "AVAILABLE TOOLS" in formatted
        assert "Tool documentation." in formatted
    
    def test_system_prompt_format_full(self):
        """Test formatting with all contexts."""
        prompt = SystemPrompt(
            base_prompt="Base prompt.",
            rag_context="RAG context.",
            tool_context="Tool context."
        )
        
        formatted = prompt.format()
        
        assert "Base prompt." in formatted
        assert "RAG context." in formatted
        assert "Tool context." in formatted


class TestSystemPromptTemplate:
    """Test cases for the system prompt template."""
    
    def test_template_contains_core_sections(self):
        """Test that template contains core instruction sections."""
        assert "GEO-REASON" in SYSTEM_PROMPT_TEMPLATE
        assert "CORE PRINCIPLES" in SYSTEM_PROMPT_TEMPLATE
        assert "RESPONSE FORMAT" in SYSTEM_PROMPT_TEMPLATE
        assert "CRS HANDLING" in SYSTEM_PROMPT_TEMPLATE
    
    def test_template_contains_response_format(self):
        """Test that template specifies response format."""
        assert "THOUGHT:" in SYSTEM_PROMPT_TEMPLATE
        assert "PLAN:" in SYSTEM_PROMPT_TEMPLATE
        assert "CODE:" in SYSTEM_PROMPT_TEMPLATE
        assert "EXPLANATION:" in SYSTEM_PROMPT_TEMPLATE
    
    def test_template_contains_crs_guidance(self):
        """Test that template includes CRS guidance."""
        assert "EPSG:4326" in SYSTEM_PROMPT_TEMPLATE
        assert "to_crs" in SYSTEM_PROMPT_TEMPLATE
        assert "UTM" in SYSTEM_PROMPT_TEMPLATE
    
    def test_template_contains_example(self):
        """Test that template includes example query."""
        assert "Kerala" in SYSTEM_PROMPT_TEMPLATE
        assert "flooding" in SYSTEM_PROMPT_TEMPLATE.lower()


class TestGetSystemPrompt:
    """Test cases for get_system_prompt function."""
    
    def test_get_system_prompt_default(self):
        """Test getting default system prompt."""
        prompt = get_system_prompt()
        
        assert isinstance(prompt, SystemPrompt)
        assert prompt.base_prompt == SYSTEM_PROMPT_TEMPLATE
    
    def test_get_system_prompt_with_rag(self):
        """Test getting system prompt with RAG context."""
        rag = "Relevant GIS documentation."
        prompt = get_system_prompt(rag_context=rag)
        
        assert prompt.rag_context == rag
    
    def test_get_system_prompt_with_tools(self):
        """Test getting system prompt with tool context."""
        tools = "Custom tool documentation."
        prompt = get_system_prompt(tool_context=tools)
        
        assert prompt.tool_context == tools


class TestGetToolContext:
    """Test cases for get_tool_context function."""
    
    def test_get_tool_context(self):
        """Test getting tool context."""
        context = get_tool_context()
        
        assert "load_vector_layer" in context
        assert "buffer_geometry" in context
        assert "calculate_raster_stats" in context
        assert "fetch_osm_data" in context
