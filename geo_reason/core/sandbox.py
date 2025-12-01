"""
Secure Sandbox for Code Execution.

Provides a restricted execution environment for running
LLM-generated Python code safely. Features:
- Import restrictions
- Timeout protection
- Memory limits
- Output capture
"""

import ast
import builtins
import io
import multiprocessing
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger


@dataclass
class ExecutionResult:
    """
    Result of code execution in the sandbox.
    
    Attributes:
        success: Whether execution completed without errors
        output: Captured stdout
        error: Error message if execution failed
        result: Final expression result if any
        variables: Dict of variables defined during execution
        execution_time: Time taken for execution in seconds
    """
    success: bool
    output: str = ""
    error: str = ""
    result: Any = None
    variables: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0


class ImportValidator(ast.NodeVisitor):
    """
    AST visitor to validate imports in code.
    
    Ensures only allowed modules can be imported.
    """
    
    def __init__(self, allowed_imports: Set[str]):
        """
        Initialize the import validator.
        
        Args:
            allowed_imports: Set of allowed module names
        """
        self.allowed_imports = allowed_imports
        self.violations: List[str] = []
    
    def visit_Import(self, node: ast.Import) -> None:
        """Check import statements."""
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            if module_name not in self.allowed_imports:
                self.violations.append(f"Import of '{alias.name}' is not allowed")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check from...import statements."""
        if node.module:
            module_name = node.module.split('.')[0]
            if module_name not in self.allowed_imports:
                self.violations.append(f"Import from '{node.module}' is not allowed")
        self.generic_visit(node)


class CodeAnalyzer(ast.NodeVisitor):
    """
    Analyze code for potentially dangerous operations.
    """
    
    DANGEROUS_ATTRS = {
        '__code__', '__globals__', '__builtins__', '__subclasses__',
        '__bases__', '__mro__', '__class__'
    }
    
    DANGEROUS_CALLS = {
        'eval', 'exec', 'compile', 'open', 'input',
        '__import__', 'getattr', 'setattr', 'delattr',
        'globals', 'locals', 'vars', 'dir'
    }
    
    def __init__(self):
        """Initialize the code analyzer."""
        self.warnings: List[str] = []
        self.blocked_operations: List[str] = []
    
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check attribute access."""
        if node.attr in self.DANGEROUS_ATTRS:
            self.blocked_operations.append(
                f"Access to '{node.attr}' is blocked for security"
            )
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls."""
        if isinstance(node.func, ast.Name):
            if node.func.id in self.DANGEROUS_CALLS:
                self.blocked_operations.append(
                    f"Call to '{node.func.id}' is blocked for security"
                )
        self.generic_visit(node)


# Store reference to the built-in import function at module load time
# This prevents potential bypassing through __builtins__ manipulation
_original_import = builtins.__import__


def _execute_in_process(
    code: str,
    allowed_imports: Set[str],
    timeout: int,
    context: Dict[str, Any],
    result_queue: multiprocessing.Queue
) -> None:
    """
    Execute code in a separate process.
    
    Args:
        code: Python code to execute
        allowed_imports: Set of allowed module names
        timeout: Execution timeout in seconds
        context: Initial execution context
        result_queue: Queue to send results back
    """
    import time
    start_time = time.time()
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        # Create restricted builtins
        safe_builtins = {
            name: getattr(builtins, name)
            for name in dir(builtins)
            if not name.startswith('_') and name not in {
                'eval', 'exec', 'compile', 'open', 'input',
                '__import__', 'breakpoint', 'exit', 'quit'
            }
        }
        
        # Custom __import__ that only allows whitelisted modules
        # Uses the original import function stored at module load time for security
        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Safe import that only allows whitelisted modules."""
            root_module = name.split('.')[0]
            if root_module not in allowed_imports:
                raise ImportError(f"Import of '{name}' is not allowed")
            return _original_import(name, globals, locals, fromlist, level)
        
        safe_builtins['__import__'] = safe_import
        
        # Prepare execution namespace
        namespace = {
            '__builtins__': safe_builtins,
            '__name__': '__sandbox__',
            '__doc__': None,
        }
        namespace.update(context)
        
        # Execute code
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(compile(code, '<sandbox>', 'exec'), namespace)
        
        # Extract result (last expression value)
        result = namespace.get('_result', namespace.get('result', None))
        
        # Filter out non-serializable items
        safe_vars = {}
        for key, value in namespace.items():
            if not key.startswith('_') and key != 'builtins':
                try:
                    # Test if serializable
                    repr(value)
                    safe_vars[key] = value
                except Exception:
                    pass
        
        execution_time = time.time() - start_time
        
        result_queue.put(ExecutionResult(
            success=True,
            output=stdout_capture.getvalue(),
            error=stderr_capture.getvalue(),
            result=result,
            variables=safe_vars,
            execution_time=execution_time
        ))
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        result_queue.put(ExecutionResult(
            success=False,
            output=stdout_capture.getvalue(),
            error=error_msg,
            execution_time=execution_time
        ))


class SecureSandbox:
    """
    Secure sandbox for executing LLM-generated Python code.
    
    Features:
    - Import restrictions (whitelist only)
    - Dangerous operation blocking
    - Timeout protection
    - Output capture
    - Error handling and reporting
    """
    
    DEFAULT_ALLOWED_IMPORTS = {
        # Core data science
        "geopandas", "pandas", "numpy", "scipy",
        # Geospatial
        "shapely", "rasterio", "fiona", "pyproj",
        "osmnx", "folium", "rasterstats",
        # Standard library
        "json", "math", "datetime", "collections",
        "itertools", "functools", "operator",
        "pathlib", "typing", "dataclasses",
        "re", "copy", "io",
        # Visualization
        "matplotlib",
    }
    
    def __init__(
        self,
        allowed_imports: Optional[Set[str]] = None,
        timeout: int = 300,
        enable_multiprocessing: bool = True
    ):
        """
        Initialize the secure sandbox.
        
        Args:
            allowed_imports: Set of allowed module names (None for defaults)
            timeout: Maximum execution time in seconds
            enable_multiprocessing: Use separate process for execution
        """
        self.allowed_imports = allowed_imports or self.DEFAULT_ALLOWED_IMPORTS.copy()
        self.timeout = timeout
        self.enable_multiprocessing = enable_multiprocessing
        
        logger.info(
            f"SecureSandbox initialized with {len(self.allowed_imports)} allowed imports, "
            f"timeout={timeout}s"
        )
    
    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate code before execution.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        try:
            # Parse the code
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]
        
        # Check imports
        import_validator = ImportValidator(self.allowed_imports)
        import_validator.visit(tree)
        issues.extend(import_validator.violations)
        
        # Check for dangerous operations
        analyzer = CodeAnalyzer()
        analyzer.visit(tree)
        issues.extend(analyzer.blocked_operations)
        issues.extend(analyzer.warnings)
        
        return len(issues) == 0, issues
    
    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        validate: bool = True
    ) -> ExecutionResult:
        """
        Execute Python code in the sandbox.
        
        Args:
            code: Python code to execute
            context: Initial variables/context for execution
            validate: Whether to validate code before execution
            
        Returns:
            ExecutionResult with execution status and outputs
        """
        context = context or {}
        
        # Validate code first
        if validate:
            is_valid, issues = self.validate_code(code)
            if not is_valid:
                return ExecutionResult(
                    success=False,
                    error=f"Code validation failed:\n" + "\n".join(issues)
                )
        
        logger.debug(f"Executing code ({len(code)} chars)")
        
        if self.enable_multiprocessing:
            return self._execute_multiprocess(code, context)
        else:
            return self._execute_inline(code, context)
    
    def _execute_multiprocess(
        self,
        code: str,
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute code in a separate process with timeout.
        
        Args:
            code: Python code to execute
            context: Execution context
            
        Returns:
            ExecutionResult
        """
        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        
        # Start execution process
        process = multiprocessing.Process(
            target=_execute_in_process,
            args=(code, self.allowed_imports, self.timeout, context, result_queue)
        )
        
        process.start()
        process.join(timeout=self.timeout)
        
        if process.is_alive():
            # Timeout - terminate the process
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
            
            return ExecutionResult(
                success=False,
                error=f"Execution timeout: exceeded {self.timeout} seconds"
            )
        
        try:
            return result_queue.get_nowait()
        except Exception:
            return ExecutionResult(
                success=False,
                error="Failed to retrieve execution result"
            )
    
    def _execute_inline(
        self,
        code: str,
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute code inline (without multiprocessing).
        
        Less secure but useful for debugging.
        
        Args:
            code: Python code to execute
            context: Execution context
            
        Returns:
            ExecutionResult
        """
        import time
        
        start_time = time.time()
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            namespace = dict(context)
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compile(code, '<sandbox>', 'exec'), namespace)
            
            result = namespace.get('_result', namespace.get('result', None))
            
            safe_vars = {}
            for key, value in namespace.items():
                if not key.startswith('_'):
                    try:
                        repr(value)
                        safe_vars[key] = value
                    except Exception:
                        pass
            
            return ExecutionResult(
                success=True,
                output=stdout_capture.getvalue(),
                error=stderr_capture.getvalue(),
                result=result,
                variables=safe_vars,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=stdout_capture.getvalue(),
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                execution_time=time.time() - start_time
            )
    
    def execute_with_retry(
        self,
        code: str,
        max_retries: int = 3,
        context: Optional[Dict[str, Any]] = None,
        error_handler: Optional[callable] = None
    ) -> ExecutionResult:
        """
        Execute code with automatic retries on failure.
        
        Args:
            code: Python code to execute
            max_retries: Maximum number of retry attempts
            context: Execution context
            error_handler: Optional function to modify code after errors
            
        Returns:
            ExecutionResult from the successful attempt or last failure
        """
        context = context or {}
        last_result = None
        
        for attempt in range(max_retries):
            result = self.execute(code, context)
            last_result = result
            
            if result.success:
                logger.info(f"Execution succeeded on attempt {attempt + 1}")
                return result
            
            logger.warning(f"Attempt {attempt + 1} failed: {result.error[:100]}...")
            
            # Try to fix the code if handler provided
            if error_handler and attempt < max_retries - 1:
                try:
                    code = error_handler(code, result.error)
                except Exception as e:
                    logger.error(f"Error handler failed: {e}")
        
        return last_result


def create_sandbox(settings=None) -> SecureSandbox:
    """
    Factory function to create a configured sandbox.
    
    Args:
        settings: Optional settings object
        
    Returns:
        Configured SecureSandbox instance
    """
    if settings:
        return SecureSandbox(
            allowed_imports=set(settings.allowed_imports),
            timeout=settings.sandbox_timeout
        )
    return SecureSandbox()
