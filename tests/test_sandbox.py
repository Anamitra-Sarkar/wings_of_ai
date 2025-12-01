"""Tests for the sandbox module."""

import pytest

from geo_reason.core.sandbox import (
    SecureSandbox,
    ExecutionResult,
    ImportValidator,
    CodeAnalyzer,
    create_sandbox
)


class TestImportValidator:
    """Test cases for ImportValidator."""
    
    def test_allowed_import(self):
        """Test that allowed imports pass validation."""
        import ast
        
        code = "import geopandas as gpd"
        tree = ast.parse(code)
        
        validator = ImportValidator({"geopandas"})
        validator.visit(tree)
        
        assert len(validator.violations) == 0
    
    def test_disallowed_import(self):
        """Test that disallowed imports are caught."""
        import ast
        
        code = "import subprocess"
        tree = ast.parse(code)
        
        validator = ImportValidator({"geopandas"})
        validator.visit(tree)
        
        assert len(validator.violations) == 1
        assert "subprocess" in validator.violations[0]
    
    def test_from_import_validation(self):
        """Test from...import statement validation."""
        import ast
        
        code = "from os import system"
        tree = ast.parse(code)
        
        validator = ImportValidator({"geopandas"})
        validator.visit(tree)
        
        assert len(validator.violations) == 1


class TestCodeAnalyzer:
    """Test cases for CodeAnalyzer."""
    
    def test_dangerous_attribute_access(self):
        """Test that dangerous attribute access is blocked."""
        import ast
        
        code = "obj.__globals__"
        tree = ast.parse(code)
        
        analyzer = CodeAnalyzer()
        analyzer.visit(tree)
        
        assert len(analyzer.blocked_operations) == 1
        assert "__globals__" in analyzer.blocked_operations[0]
    
    def test_dangerous_function_call(self):
        """Test that dangerous function calls are blocked."""
        import ast
        
        code = "eval('print(1)')"
        tree = ast.parse(code)
        
        analyzer = CodeAnalyzer()
        analyzer.visit(tree)
        
        assert len(analyzer.blocked_operations) == 1
        assert "eval" in analyzer.blocked_operations[0]
    
    def test_safe_code(self):
        """Test that safe code passes analysis."""
        import ast
        
        code = """
x = 1 + 2
y = x * 3
print(y)
"""
        tree = ast.parse(code)
        
        analyzer = CodeAnalyzer()
        analyzer.visit(tree)
        
        assert len(analyzer.blocked_operations) == 0


class TestSecureSandbox:
    """Test cases for SecureSandbox."""
    
    def test_sandbox_initialization(self):
        """Test sandbox initialization."""
        sandbox = SecureSandbox()
        
        assert sandbox.timeout == 300
        assert "geopandas" in sandbox.allowed_imports
    
    def test_validate_code_syntax_error(self):
        """Test validation catches syntax errors."""
        sandbox = SecureSandbox()
        
        code = "def foo(:"  # Invalid syntax
        is_valid, issues = sandbox.validate_code(code)
        
        assert is_valid is False
        assert any("Syntax error" in issue for issue in issues)
    
    def test_validate_code_disallowed_import(self):
        """Test validation catches disallowed imports."""
        sandbox = SecureSandbox()
        
        code = "import subprocess"
        is_valid, issues = sandbox.validate_code(code)
        
        assert is_valid is False
        assert any("subprocess" in issue for issue in issues)
    
    def test_validate_code_valid(self):
        """Test validation passes for valid code."""
        sandbox = SecureSandbox()
        
        code = """
x = 1 + 2
result = x * 3
print(result)
"""
        is_valid, issues = sandbox.validate_code(code)
        
        assert is_valid is True
        assert len(issues) == 0
    
    def test_execute_simple_code(self):
        """Test executing simple valid code."""
        sandbox = SecureSandbox(enable_multiprocessing=False)
        
        code = """
x = 1 + 2
result = x * 3
print(f"Result: {result}")
"""
        
        result = sandbox.execute(code)
        
        assert result.success is True
        assert "Result: 9" in result.output
    
    def test_execute_returns_variables(self):
        """Test that execution returns defined variables."""
        sandbox = SecureSandbox(enable_multiprocessing=False)
        
        code = """
x = 42
y = "hello"
result = x * 2
"""
        
        result = sandbox.execute(code)
        
        assert result.success is True
        assert result.variables.get("x") == 42
        assert result.variables.get("y") == "hello"
        assert result.variables.get("result") == 84
    
    def test_execute_code_with_error(self):
        """Test that execution captures errors."""
        sandbox = SecureSandbox(enable_multiprocessing=False)
        
        code = """
x = 1 / 0
"""
        
        result = sandbox.execute(code)
        
        assert result.success is False
        assert "ZeroDivisionError" in result.error
    
    def test_execute_disallowed_import_blocked(self):
        """Test that disallowed imports are blocked at validation."""
        sandbox = SecureSandbox(enable_multiprocessing=False)
        
        code = "import subprocess"
        
        result = sandbox.execute(code)
        
        assert result.success is False
        assert "validation failed" in result.error.lower()


class TestCreateSandbox:
    """Test cases for create_sandbox factory."""
    
    def test_create_sandbox_default(self):
        """Test creating sandbox with defaults."""
        sandbox = create_sandbox()
        
        assert isinstance(sandbox, SecureSandbox)
        assert sandbox.timeout == 300
    
    def test_create_sandbox_with_settings(self):
        """Test creating sandbox with custom settings."""
        from geo_reason.core.config import Settings
        
        settings = Settings(
            sandbox_timeout=600,
            allowed_imports=["geopandas", "pandas"]
        )
        
        sandbox = create_sandbox(settings)
        
        assert sandbox.timeout == 600
        assert "geopandas" in sandbox.allowed_imports
        assert "pandas" in sandbox.allowed_imports
