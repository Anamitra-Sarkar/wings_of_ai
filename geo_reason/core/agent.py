"""
GeoReason Agent - ReAct Loop Implementation.

Implements the Reasoning-Acting-Observation (ReAct) loop for
autonomous geospatial analysis using LLMs.

Features:
- Chain-of-Thought reasoning
- Tool execution with error recovery
- RAG integration for documentation retrieval
- Iterative refinement of analysis plans
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from loguru import logger

from geo_reason.core.config import Settings, get_settings
from geo_reason.core.sandbox import ExecutionResult, SecureSandbox
from geo_reason.core.tools import GeoTools
from geo_reason.prompts.brain import get_system_prompt, get_tool_context


@dataclass
class AgentStep:
    """
    Represents a single step in the ReAct loop.
    
    Attributes:
        step_number: Sequential step number
        thought: Agent's reasoning
        plan: Structured analysis plan
        code: Generated Python code
        explanation: Code explanation
        execution_result: Result of code execution
        timestamp: When the step was executed
    """
    step_number: int
    thought: str = ""
    plan: Optional[Dict[str, Any]] = None
    code: str = ""
    explanation: str = ""
    execution_result: Optional[ExecutionResult] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResponse:
    """
    Complete response from the agent.
    
    Attributes:
        query: Original user query
        steps: List of all ReAct steps
        final_result: Final analysis result
        success: Whether the analysis succeeded
        error: Error message if failed
        total_time: Total execution time
        geojson_output: GeoJSON output if available
    """
    query: str
    steps: List[AgentStep] = field(default_factory=list)
    final_result: Any = None
    success: bool = False
    error: str = ""
    total_time: float = 0.0
    geojson_output: Optional[Dict] = None


class ResponseParser:
    """
    Parse LLM responses into structured components.
    
    Extracts THOUGHT, PLAN, CODE, and EXPLANATION sections
    from the model output.
    """
    
    # Patterns for extracting sections
    THOUGHT_PATTERN = re.compile(
        r'(?:###\s*)?THOUGHT:?\s*\n?(.*?)(?=(?:###\s*)?PLAN:|$)',
        re.DOTALL | re.IGNORECASE
    )
    
    PLAN_PATTERN = re.compile(
        r'(?:###\s*)?PLAN:?\s*\n?```json\s*(.*?)```',
        re.DOTALL | re.IGNORECASE
    )
    
    CODE_PATTERN = re.compile(
        r'(?:###\s*)?CODE:?\s*\n?```python\s*(.*?)```',
        re.DOTALL | re.IGNORECASE
    )
    
    EXPLANATION_PATTERN = re.compile(
        r'(?:###\s*)?EXPLANATION:?\s*\n?(.*?)$',
        re.DOTALL | re.IGNORECASE
    )
    
    @classmethod
    def parse(cls, response: str) -> Dict[str, Any]:
        """
        Parse an LLM response into components.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Dict with thought, plan, code, explanation keys
        """
        result = {
            "thought": "",
            "plan": None,
            "code": "",
            "explanation": ""
        }
        
        # Extract thought
        thought_match = cls.THOUGHT_PATTERN.search(response)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # Extract plan (JSON)
        plan_match = cls.PLAN_PATTERN.search(response)
        if plan_match:
            try:
                result["plan"] = json.loads(plan_match.group(1).strip())
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse plan JSON: {e}")
        
        # Extract code
        code_match = cls.CODE_PATTERN.search(response)
        if code_match:
            result["code"] = code_match.group(1).strip()
        
        # Extract explanation
        explanation_match = cls.EXPLANATION_PATTERN.search(response)
        if explanation_match:
            result["explanation"] = explanation_match.group(1).strip()
        
        return result


class LLMInterface:
    """
    Interface for LLM interactions.
    
    Supports both local models (via transformers) and API-based models.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: str = "auto",
        max_tokens: int = 4096,
        temperature: float = 0.1,
        use_quantization: bool = True
    ):
        """
        Initialize the LLM interface.
        
        Args:
            model_name: HuggingFace model name
            device: Device for inference (cuda, cpu, auto)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_quantization: Use 4-bit quantization
        """
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_quantization = use_quantization
        
        self.model = None
        self.tokenizer = None
        self._initialized = False
    
    def _initialize_model(self) -> None:
        """Initialize the model and tokenizer."""
        if self._initialized:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
            
            logger.info(f"Loading model: {self.model_name}")
            
            # Configure quantization if requested
            quantization_config = None
            if self.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=self.device,
                torch_dtype=torch.float16
            )
            
            self._initialized = True
            logger.info("Model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Missing dependencies for local LLM: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Override for max tokens
            
        Returns:
            Generated text response
        """
        if not self._initialized:
            self._initialize_model()
        
        try:
            # Format messages for chat
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=8192
            ).to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise


class MockLLMInterface:
    """
    Mock LLM interface for testing without actual model.
    """
    
    def __init__(self, **kwargs):
        """Initialize mock interface."""
        self.model_name = "mock"
        logger.info("Using MockLLMInterface")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None
    ) -> str:
        """Generate a mock response."""
        # Return a template response for testing
        return """### THOUGHT:
I need to analyze the user's request and create a plan for geospatial analysis.
The query asks for identifying areas of interest.

### PLAN:
```json
{
    "task_description": "Sample geospatial analysis",
    "target_crs": "EPSG:4326",
    "data_requirements": [],
    "analysis_steps": [
        {
            "step_id": 1,
            "tool": "fetch_osm_data",
            "description": "Fetch sample data",
            "inputs": {},
            "expected_output": "GeoDataFrame"
        }
    ],
    "output_format": "GeoJSON",
    "potential_issues": []
}
```

### CODE:
```python
import geopandas as gpd
from shapely.geometry import Point

# Create sample data
data = gpd.GeoDataFrame({
    'name': ['Location A', 'Location B'],
    'geometry': [Point(78.9629, 20.5937), Point(77.5946, 12.9716)]
}, crs="EPSG:4326")

result = data
print(f"Created {len(result)} sample features")
```

### EXPLANATION:
This is a sample analysis that creates test geospatial data.
The result contains two sample points in India.
"""


class GeoReasonAgent:
    """
    Main agent class implementing the ReAct loop for geospatial analysis.
    
    The agent:
    1. Receives natural language queries
    2. Uses chain-of-thought reasoning to plan analysis
    3. Generates and executes Python code
    4. Observes results and iterates if needed
    5. Returns structured results with visualization data
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm: Optional[Union[LLMInterface, MockLLMInterface]] = None,
        sandbox: Optional[SecureSandbox] = None,
        tools: Optional[GeoTools] = None,
        rag_engine: Any = None,
        use_mock_llm: bool = False
    ):
        """
        Initialize the GeoReason agent.
        
        Args:
            settings: Configuration settings
            llm: LLM interface for generation
            sandbox: Sandbox for code execution
            tools: GeoTools instance for geospatial operations
            rag_engine: RAG engine for documentation retrieval
            use_mock_llm: Use mock LLM for testing
        """
        self.settings = settings or get_settings()
        
        # Initialize LLM
        if llm:
            self.llm = llm
        elif use_mock_llm:
            self.llm = MockLLMInterface()
        else:
            self.llm = LLMInterface(
                model_name=self.settings.model_name,
                device=self.settings.model_device,
                max_tokens=self.settings.model_max_tokens,
                temperature=self.settings.model_temperature,
                use_quantization=self.settings.use_quantization
            )
        
        # Initialize sandbox
        self.sandbox = sandbox or SecureSandbox(
            allowed_imports=set(self.settings.allowed_imports),
            timeout=self.settings.sandbox_timeout
        )
        
        # Initialize tools
        self.tools = tools or GeoTools(cache_dir=self.settings.osm_cache_dir)
        
        # RAG engine (optional)
        self.rag_engine = rag_engine
        
        # Conversation history for context
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info("GeoReasonAgent initialized")
    
    def _get_system_prompt(self, query: str) -> str:
        """
        Build the system prompt with optional RAG context.
        
        Args:
            query: User's query for context retrieval
            
        Returns:
            Formatted system prompt
        """
        rag_context = None
        if self.rag_engine:
            try:
                rag_context = self.rag_engine.get_gis_tool_context(query)
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        
        system_prompt = get_system_prompt(
            rag_context=rag_context,
            tool_context=get_tool_context()
        )
        
        return system_prompt.format()
    
    def _build_error_feedback_prompt(
        self,
        error: str,
        code: str,
        step: AgentStep
    ) -> str:
        """
        Build a prompt for error recovery.
        
        Args:
            error: Error message from execution
            code: Code that failed
            step: The step that produced the error
            
        Returns:
            Formatted error feedback prompt
        """
        return f"""The previous code execution failed with the following error:

ERROR:
{error}

FAILED CODE:
```python
{code}
```

Please analyze the error and provide a corrected solution following the same THOUGHT -> PLAN -> CODE -> EXPLANATION format.
Focus on fixing the specific error while maintaining the overall analysis goal.
"""
    
    def _execute_step(
        self,
        parsed_response: Dict[str, Any],
        step_number: int
    ) -> AgentStep:
        """
        Execute a single step of the ReAct loop.
        
        Args:
            parsed_response: Parsed LLM response with thought, plan, code
            step_number: Current step number
            
        Returns:
            AgentStep with execution result
        """
        step = AgentStep(
            step_number=step_number,
            thought=parsed_response.get("thought", ""),
            plan=parsed_response.get("plan"),
            code=parsed_response.get("code", ""),
            explanation=parsed_response.get("explanation", "")
        )
        
        if step.code:
            # Execute the code in sandbox
            logger.info(f"Executing code for step {step_number}")
            
            # Prepare context with tools
            context = {
                "tools": self.tools,
                "GeoTools": GeoTools,
            }
            
            step.execution_result = self.sandbox.execute(
                step.code,
                context=context,
                validate=True
            )
            
            if step.execution_result.success:
                logger.info(f"Step {step_number} executed successfully")
            else:
                logger.warning(
                    f"Step {step_number} failed: "
                    f"{step.execution_result.error[:200]}..."
                )
        
        return step
    
    def run(
        self,
        query: str,
        max_iterations: int = 5,
        callback: Optional[Callable[[AgentStep], None]] = None
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
        start_time = time.time()
        response = AgentResponse(query=query)
        
        logger.info(f"Starting analysis for query: {query[:100]}...")
        
        try:
            # Build initial messages
            system_prompt = self._get_system_prompt(query)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            for iteration in range(max_iterations):
                logger.info(f"Iteration {iteration + 1}/{max_iterations}")
                
                # Generate response from LLM
                llm_response = self.llm.generate(messages)
                
                # Parse the response
                parsed = ResponseParser.parse(llm_response)
                
                # Execute the step
                step = self._execute_step(parsed, iteration + 1)
                response.steps.append(step)
                
                # Notify callback
                if callback:
                    callback(step)
                
                # Check if successful
                if step.execution_result and step.execution_result.success:
                    # Extract GeoJSON if available
                    if step.execution_result.variables:
                        result_var = step.execution_result.variables.get(
                            'result',
                            step.execution_result.variables.get('gdf')
                        )
                        if result_var is not None:
                            try:
                                # Convert GeoDataFrame to GeoJSON
                                if hasattr(result_var, '__geo_interface__'):
                                    response.geojson_output = result_var.__geo_interface__
                                elif hasattr(result_var, 'to_json'):
                                    response.geojson_output = json.loads(result_var.to_json())
                            except Exception as e:
                                logger.warning(f"Could not convert result to GeoJSON: {e}")
                    
                    response.final_result = step.execution_result.result or step.execution_result.output
                    response.success = True
                    break
                
                # If failed, prepare error feedback for next iteration
                if step.execution_result and not step.execution_result.success:
                    error_feedback = self._build_error_feedback_prompt(
                        step.execution_result.error,
                        step.code,
                        step
                    )
                    messages.append({"role": "assistant", "content": llm_response})
                    messages.append({"role": "user", "content": error_feedback})
            
            if not response.success:
                response.error = "Failed to complete analysis within maximum iterations"
            
        except Exception as e:
            logger.error(f"Agent error: {e}")
            response.error = str(e)
        
        response.total_time = time.time() - start_time
        logger.info(
            f"Analysis completed: success={response.success}, "
            f"steps={len(response.steps)}, time={response.total_time:.2f}s"
        )
        
        return response
    
    def reset(self) -> None:
        """Reset the agent state for a new conversation."""
        self.conversation_history = []
        logger.debug("Agent state reset")


def create_agent(
    use_mock_llm: bool = False,
    settings: Optional[Settings] = None
) -> GeoReasonAgent:
    """
    Factory function to create a configured GeoReason agent.
    
    Args:
        use_mock_llm: Use mock LLM for testing
        settings: Optional settings override
        
    Returns:
        Configured GeoReasonAgent instance
    """
    return GeoReasonAgent(
        settings=settings,
        use_mock_llm=use_mock_llm
    )
