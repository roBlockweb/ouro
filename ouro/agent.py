"""
Agent capabilities for Ouro.
"""
import json
import os
import re
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Tuple

import requests
from pydantic import BaseModel, Field

from ouro.logger import get_logger
from ouro.config import (
    AGENT_SYSTEM_PROMPT,
    TOOLS_ENABLED,
    ENABLE_WEB_SEARCH,
    WEB_SEARCH_PROVIDER,
    MAX_SEARCH_RESULTS
)

logger = get_logger()


class ToolStatus(str, Enum):
    """Status of a tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    THINKING = "thinking"


class ToolResult(BaseModel):
    """Result of a tool execution."""
    status: ToolStatus
    result: Optional[str] = None
    error: Optional[str] = None


class AgentTool:
    """Base class for agent tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def run(self, **kwargs) -> ToolResult:
        """Run the tool with the given parameters."""
        try:
            result = self._execute(**kwargs)
            return ToolResult(status=ToolStatus.SUCCESS, result=result)
        except Exception as e:
            logger.error(f"Error running tool {self.name}: {e}")
            return ToolResult(status=ToolStatus.ERROR, error=str(e))
    
    def _execute(self, **kwargs) -> str:
        """Execute the tool with the given parameters."""
        raise NotImplementedError("Tool must implement _execute method")


class WebSearchTool(AgentTool):
    """Tool for performing web searches."""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information on a specific topic"
        )
    
    def _execute(self, query: str, num_results: int = MAX_SEARCH_RESULTS) -> str:
        """Execute a web search."""
        if not ENABLE_WEB_SEARCH:
            return "Web search is disabled in configuration."
        
        if WEB_SEARCH_PROVIDER == "duckduckgo":
            return self._search_duckduckgo(query, num_results)
        elif WEB_SEARCH_PROVIDER == "google":
            return "Google search is not implemented yet."
        elif WEB_SEARCH_PROVIDER == "bing":
            return "Bing search is not implemented yet."
        else:
            return f"Unknown search provider: {WEB_SEARCH_PROVIDER}"
    
    def _search_duckduckgo(self, query: str, num_results: int) -> str:
        """Search DuckDuckGo."""
        try:
            # This is a basic implementation using DuckDuckGo's HTML
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'no_redirect': '1'
            }
            
            response = requests.get('https://api.duckduckgo.com/', params=params, headers=headers)
            
            if response.status_code != 200:
                return f"Error: Received status code {response.status_code}"
            
            data = response.json()
            
            # Process and format results
            results = []
            
            # Add abstract if available
            if data.get('Abstract'):
                results.append(f"ABSTRACT: {data['Abstract']}")
                if data.get('AbstractSource'):
                    results[-1] += f" (Source: {data['AbstractSource']})"
            
            # Add related topics
            if data.get('RelatedTopics'):
                for i, topic in enumerate(data['RelatedTopics']):
                    if i >= num_results:
                        break
                    if 'Text' in topic:
                        results.append(f"TOPIC: {topic['Text']}")
            
            if not results:
                return "No results found for the query."
            
            return "\n\n".join(results)
        
        except Exception as e:
            logger.error(f"Error in DuckDuckGo search: {e}")
            return f"Error performing search: {str(e)}"


class MathTool(AgentTool):
    """Tool for performing mathematical calculations."""
    
    def __init__(self):
        super().__init__(
            name="math",
            description="Perform mathematical calculations"
        )
    
    def _execute(self, expression: str) -> str:
        """Evaluate a mathematical expression."""
        # Clean the expression to avoid security issues
        sanitized = self._sanitize_expression(expression)
        
        try:
            # Use eval with restricted globals to evaluate the expression
            # This is safer than raw eval but still has limitations
            result = eval(sanitized, {"__builtins__": {}}, {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
                "int": int,
                "float": float,
            })
            return f"Result: {result}"
        except Exception as e:
            logger.error(f"Error evaluating math expression: {e}")
            return f"Error: Could not evaluate the expression. {str(e)}"
    
    def _sanitize_expression(self, expression: str) -> str:
        """Sanitize a mathematical expression to avoid security issues."""
        # Remove all characters except digits, operators, and common math symbols
        sanitized = re.sub(r'[^0-9+\-*/().,\s]', '', expression)
        return sanitized


class QueryGenerationTool(AgentTool):
    """Tool for generating effective search queries."""
    
    def __init__(self):
        super().__init__(
            name="query_generation",
            description="Generate effective search queries from a user question"
        )
    
    def _execute(self, question: str) -> str:
        """Generate effective search queries from a user question."""
        # Extract key components from the question
        words = question.lower().split()
        stopwords = {"a", "an", "the", "and", "or", "but", "is", "are", "was", "were", 
                     "how", "what", "why", "when", "where", "who", "whom", "which", 
                     "can", "could", "would", "should", "will", "shall", "may", "might",
                     "do", "does", "did", "done", "am", "be", "been", "being", "have", "has", "had"}
        
        # Remove stopwords and punctuation
        keywords = [word.strip('.,?!;:"\'()[]{}') for word in words if word not in stopwords]
        
        # Generate 3 different queries
        queries = []
        
        # Query 1: All keywords combined
        if keywords:
            queries.append(" ".join(keywords))
        
        # Query 2: All keywords with quotes around key phrases (if any)
        if len(keywords) >= 2:
            query2 = " ".join(keywords)
            for i in range(len(keywords) - 1):
                phrase = f"{keywords[i]} {keywords[i+1]}"
                if len(phrase) > 5:  # Only quote meaningful phrases
                    query2 = query2.replace(phrase, f'"{phrase}"')
            queries.append(query2)
        
        # Query 3: Rephrase as a statement
        if keywords:
            if question.lower().startswith("how to"):
                queries.append(question)
            elif question.lower().startswith("what is"):
                queries.append(question.replace("what is", "define").replace("What is", "Define"))
            else:
                queries.append(question)
        
        return "Generated search queries:\n1. " + "\n2. ".join(queries)


class SummarizationTool(AgentTool):
    """Tool for summarizing text."""
    
    def __init__(self):
        super().__init__(
            name="summarization",
            description="Summarize a block of text to extract the key points"
        )
    
    def _execute(self, text: str, max_length: int = 200) -> str:
        """Summarize text."""
        # This is a very basic extractive summarization
        # In a real implementation, we'd use a proper NLP model
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) <= 3:
            return text  # Text is already short enough
        
        # Score sentences based on simple metrics
        scores = {}
        for i, sentence in enumerate(sentences):
            # Score based on position (earlier sentences are more important)
            position_score = 1.0 if i < 3 else 0.5
            
            # Score based on length (not too short, not too long)
            length = len(sentence)
            length_score = 0.8 if 10 < length < 30 else 0.5
            
            # Score based on presence of numbers (often important)
            number_score = 1.2 if re.search(r'\d', sentence) else 1.0
            
            # Calculate total score
            scores[i] = position_score * length_score * number_score
        
        # Select top sentences
        top_indices = sorted(scores, key=scores.get, reverse=True)[:3]
        top_indices.sort()  # Restore original order
        
        # Create summary
        summary = " ".join([sentences[i] for i in top_indices])
        
        return f"Summary: {summary}"


class OuroAgent:
    """Main agent class to coordinate tool usage and reasoning."""
    
    def __init__(self, llm=None):
        """Initialize the agent with tools."""
        self.llm = llm  # The language model to use
        
        # Initialize tools
        self.tools = {}
        if "web_search" in TOOLS_ENABLED:
            self.tools["web_search"] = WebSearchTool()
        if "math" in TOOLS_ENABLED:
            self.tools["math"] = MathTool()
        if "query_generation" in TOOLS_ENABLED:
            self.tools["query_generation"] = QueryGenerationTool()
        if "summarization" in TOOLS_ENABLED:
            self.tools["summarization"] = SummarizationTool()
        
        self.logger = logger
    
    def get_tool_descriptions(self) -> str:
        """Get descriptions of available tools."""
        if not self.tools:
            return "No tools available."
        
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"{name}: {tool.description}")
        
        return "\n".join(descriptions)
    
    def run_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Run a specific tool with given parameters."""
        if tool_name not in self.tools:
            return ToolResult(
                status=ToolStatus.ERROR, 
                error=f"Tool '{tool_name}' not found. Available tools: {', '.join(self.tools.keys())}"
            )
        
        return self.tools[tool_name].run(**kwargs)
    
    def solve_task(self, task: str) -> str:
        """Solve a complex task using tools and reasoning."""
        if not self.llm:
            return "Agent requires an LLM to solve tasks."
        
        # Generate a plan using the LLM
        plan = self._generate_plan(task)
        
        # Extract steps from the plan
        steps = self._extract_steps(plan)
        
        # Execute each step
        results = []
        for i, step in enumerate(steps):
            self.logger.info(f"Executing step {i+1}: {step}")
            
            # Determine if this step requires a tool
            tool_to_use, params = self._extract_tool_and_params(step)
            
            if tool_to_use:
                # Execute tool
                tool_result = self.run_tool(tool_to_use, **params)
                results.append(f"Step {i+1} Result: {tool_result.result if tool_result.status == ToolStatus.SUCCESS else tool_result.error}")
            else:
                # Let the LLM handle this step
                reasoning = self._generate_reasoning(step, results)
                results.append(f"Step {i+1} Reasoning: {reasoning}")
        
        # Generate final answer
        final_answer = self._generate_final_answer(task, steps, results)
        
        return final_answer
    
    def _generate_plan(self, task: str) -> str:
        """Generate a plan to solve the task."""
        # Prepare context for the LLM
        tools_desc = self.get_tool_descriptions()
        prompt = f"{AGENT_SYSTEM_PROMPT}\n\nAVAILABLE TOOLS:\n{tools_desc}\n\nTASK: {task}\n\nPlease create a step-by-step plan to solve this task:"
        
        # Let the LLM generate a plan
        response = self.llm.generate_text(prompt)
        
        return response
    
    def _extract_steps(self, plan: str) -> List[str]:
        """Extract individual steps from the plan."""
        # Look for numbered or bulleted steps
        step_pattern = r'(?:^|\n)(?:\d+\.|-)?\s*(.*?)(?=(?:\n\d+\.|\n-|\n\n|\Z))'
        steps = re.findall(step_pattern, plan)
        
        # Filter out empty or irrelevant matches
        steps = [step.strip() for step in steps if step.strip()]
        
        # If no steps were found, try treating paragraphs as steps
        if not steps:
            steps = [para.strip() for para in plan.split('\n\n') if para.strip()]
        
        # If still no steps, use the entire plan
        if not steps:
            steps = [plan]
        
        return steps
    
    def _extract_tool_and_params(self, step: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Extract tool name and parameters from a step description."""
        # Check if the step mentions any of our tools
        tool_to_use = None
        params = {}
        
        for tool_name in self.tools:
            if tool_name.lower() in step.lower():
                tool_to_use = tool_name
                break
        
        if tool_to_use == "web_search":
            # Extract query parameter
            match = re.search(r'(?:search|query)(?:\s+for)?[:\s]+"([^"]+)"', step, re.IGNORECASE)
            if not match:
                match = re.search(r'(?:search|query)(?:\s+for)?[:\s]+([^.;]+)', step, re.IGNORECASE)
            
            if match:
                params["query"] = match.group(1).strip()
            else:
                # Fallback: use the entire step as query
                params["query"] = step
        
        elif tool_to_use == "math":
            # Extract expression parameter
            match = re.search(r'(?:calculate|compute|evaluate)[:\s]+([^.;]+)', step, re.IGNORECASE)
            
            if match:
                params["expression"] = match.group(1).strip()
            else:
                # Fallback to looking for a mathematical expression
                match = re.search(r'(\d[\d\s+\-*/().]+\d)', step)
                if match:
                    params["expression"] = match.group(1).strip()
        
        elif tool_to_use == "query_generation":
            # Extract question parameter
            match = re.search(r'(?:generate|create)(?:\s+a)?(?:\s+query)?(?:\s+for)?[:\s]+([^.;]+)', step, re.IGNORECASE)
            
            if match:
                params["question"] = match.group(1).strip()
            else:
                params["question"] = step
        
        elif tool_to_use == "summarization":
            # Extract text parameter
            match = re.search(r'(?:summarize|summary)[:\s]+([^.;]+)', step, re.IGNORECASE)
            
            if match:
                params["text"] = match.group(1).strip()
            else:
                params["text"] = step
        
        return tool_to_use, params
    
    def _generate_reasoning(self, step: str, previous_results: List[str]) -> str:
        """Generate reasoning for a step that doesn't require a tool."""
        # Prepare context for the LLM
        previous_context = "\n".join(previous_results)
        prompt = f"{AGENT_SYSTEM_PROMPT}\n\nPrevious steps and results:\n{previous_context}\n\nCurrent step: {step}\n\nPlease provide reasoning for this step:"
        
        # Let the LLM generate reasoning
        response = self.llm.generate_text(prompt)
        
        return response
    
    def _generate_final_answer(self, task: str, steps: List[str], results: List[str]) -> str:
        """Generate a final answer based on all steps and results."""
        # Prepare context for the LLM
        steps_and_results = []
        for i, (step, result) in enumerate(zip(steps, results)):
            steps_and_results.append(f"Step {i+1}: {step}")
            steps_and_results.append(f"Result: {result}")
        
        step_context = "\n".join(steps_and_results)
        
        prompt = f"{AGENT_SYSTEM_PROMPT}\n\nTask: {task}\n\nSteps and results:\n{step_context}\n\nBased on the above information, please provide a final answer to the task:"
        
        # Let the LLM generate the final answer
        response = self.llm.generate_text(prompt)
        
        return response