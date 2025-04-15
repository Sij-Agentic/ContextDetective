import os
import asyncio
import json
import logging
import datetime
from pathlib import Path
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
from concurrent.futures import TimeoutError
from typing import List, Dict, Any, Optional
import inspect
import sys

def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"context_detective_{timestamp}.log"
    
    # Create simple log.txt file with UTF-8 encoding
    with open("log.txt", "w", encoding='utf-8') as f:
        f.write(f"Log started at {datetime.datetime.now()}\n")
        f.write("="*50 + "\n")
    
    # Configure logging with UTF-8 encoding
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(),
            logging.FileHandler("log.txt", mode='a', encoding='utf-8')
        ]
    )
    
    # Create logger
    logger = logging.getLogger("ContextDetective")
    logger.setLevel(logging.INFO)
    
    logger.info("Logging setup complete")
    logger.info(f"Detailed logs will be written to: {log_file}")
    logger.info("Simple log will be written to: log.txt")
    
    return logger

class WorkflowState:
    def __init__(self):
        self.iteration = 0
        self.visual_elements = None
        self.style_analysis = None
        self.scenario_analysis = None
        self.search_terms = None
        self.web_findings = None
        self.context_inference = None
        self.final_output = None
        self.errors = []

    def update(self, key: str, value: Any):
        setattr(self, key, value)
        logger.info(f"Updated state: {key} = {value}")

    def get_progress(self) -> str:
        progress = f"""Current Progress:
    - Visual Elements Analysis: {'[X]' if self.visual_elements and len(str(self.visual_elements)) > 2 else '[ ]'}
    - Style Analysis: {'[X]' if self.style_analysis and len(str(self.style_analysis)) > 2 else '[ ]'}
    - Scenario Analysis: {'[X]' if self.scenario_analysis and len(str(self.scenario_analysis)) > 2 else '[ ]'}
    - Search Terms: {'[X]' if self.search_terms and len(str(self.search_terms)) > 2 else '[ ]'}
    - Web Findings: {'[X]' if self.web_findings and len(str(self.web_findings)) > 2 else '[ ]'}
    - Context Inference: {'[X]' if self.context_inference and len(str(self.context_inference)) > 2 else '[ ]'}
    - Final Output: {'[X]' if self.final_output and len(str(self.final_output)) > 2 else '[ ]'}"""
        logger.info("\nWorkflow Progress:")
        logger.info(progress)
        return progress

def normalize_path(path: str) -> str:
    """Normalize path for consistent handling across tools."""
    # Convert backslashes to forward slashes and resolve any relative paths
    normalized = os.path.normpath(path).replace('\\', '/')
    logger.info(f"Normalized path: {path} -> {normalized}")
    return normalized




# Initialize logger
logger = setup_logging()

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# System prompt for the context detection workflow
SYSTEM_PROMPT = """You are an agentic context detector that analyzes images to determine their context, meaning, and significance.

### Tool Usage:
- FUNCTION_CALL: tool_name|param1|param2|... (one per line)
- Include 'FUNCTION_CALL:', function_name|, and parameters (if necessary)
- FINAL_ANSWER: result

### Workflow Steps:
1. Analyze the image using multiple perspectives (-- IMPORTANT -- Ensure visual analysis performed before moving on to search terms):
   - Visual elements (objects, people, colors, text)
   - Style and aesthetics (artistic style, cultural elements)
   - Possible scenarios (what might be happening) 
2. Generate search terms based on findings -- Perform all visual analysis before search term query
3. Explore the web for supporting evidence
4. Infer the most likely context
5. Provide structured output with confidence rating

### Tools:
{tools_block}

### Response Format:
For each step, use FUNCTION_CALL to execute tools.
When complete, provide FINAL_ANSWER with the structured output.
"""

async def generate_with_timeout(prompt: str, timeout: int = 15) -> str:
    """Generate response from Gemini with timeout."""
    logger.info("Generating response from Gemini")
    try:
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
            ),
            timeout=timeout
        )
        logger.info("Successfully received response from Gemini")
        return response.text.strip()
    except TimeoutError:
        logger.error("Gemini generation timed out")
        raise Exception("LLM generation timed out!")
    except Exception as e:
        logger.error(f"Error in Gemini generation: {str(e)}", exc_info=True)
        raise

def format_tools_for_prompt(tools: List[Any]) -> str:
    """Format tools information for the system prompt."""
    tool_descriptions = []
    for i, tool in enumerate(tools):
        try:
            params = tool.inputSchema
            desc = getattr(tool, 'description', 'No description')
            name = getattr(tool, 'name', f'tool_{i}')

            if 'properties' in params:
                param_list = [
                    f"{param}: {info.get('type', 'unknown')}"
                    for param, info in params['properties'].items()
                ]
                param_str = ", ".join(param_list)
            else:
                param_str = "no parameters"

            tool_descriptions.append(f"{i+1}. {name}({param_str}) - {desc}")
        except Exception as e:
            logger.error(f"Error formatting tool {i}: {str(e)}")
            tool_descriptions.append(f"{i+1}. Error loading tool: {str(e)}")

    return "\n".join(tool_descriptions)

def construct_prompt(system_prompt: str, image_path: str, state: WorkflowState) -> str:
    """Construct the prompt for the current iteration."""
    prompt = f"{system_prompt}\n\nImage Path: {image_path}\n"
    prompt += state.get_progress()
    
    if state.errors:
        prompt += "\n\nPrevious Errors:\n"
        for error in state.errors:
            prompt += f"- {error}\n"
    
    prompt += "\nWhat should you do next?\n"
    return prompt

async def execute_tool_call(
    session: ClientSession,
    tool_name: str,
    params: List[str],
    tools: List[Any],
    state: WorkflowState
) -> Optional[str]:
    """Execute a single tool call and return the result."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Executing tool: {tool_name}")
    logger.info(f"Parameters received: {params}")
    logger.info("Current analysis state:")
    logger.info(f"- Visual Elements: {state.visual_elements}")
    logger.info(f"- Style Analysis: {state.style_analysis}")
    logger.info(f"- Scenario Analysis: {state.scenario_analysis}")
    logger.info(f"- Search Terms: {state.search_terms}")
    logger.info(f"{'='*50}\n")
    
    try:
        tool = next((t for t in tools if t.name == tool_name), None)
        if not tool:
            error_msg = f"Tool '{tool_name}' not found"
            logger.error(error_msg)
            return error_msg

        arguments = {}
        schema = tool.inputSchema.get("properties", {})
        param_names = list(schema.keys())
        
        logger.info(f"Tool schema: {schema}")
        
        # Special handling for generate_search_terms
        if tool_name == "generate_search_terms":
            logger.info("Processing search terms generation")
            
            # Ensure we have the previous analyses
            if not state.visual_elements or not state.style_analysis or not state.scenario_analysis:
                logger.error("Missing required analyses for search term generation")
                return "Error: Missing required analyses for search term generation"
            
            # Create a structured context from previous analyses
            context = f"""Previous Image Analyses:

1. Visual Elements Analysis:
{state.visual_elements}

2. Style Analysis:
{state.style_analysis}

3. Scenario Analysis:
{state.scenario_analysis}

Based on these analyses, generate specific search terms for each category."""

            logger.info("Constructed context for search terms:")
            logger.info(context)
            
            arguments["descriptions"] = [context]
            
###
        # Special handling for search_web
        elif tool_name == "search_web":
            logger.info("Processing web search")
            if not state.search_terms:
                logger.error("No search terms available for web search")
                return "Error: No search terms available for web search"
            
            # Extract the actual search terms text from the state
            search_terms_text = state.search_terms
            if isinstance(search_terms_text, str):
                # If it's a string representation of a list/dict, try to parse it
                try:
                    if search_terms_text.startswith('['):
                        # Remove outer brackets and parse the content
                        content_str = search_terms_text.strip('[]')
                        if '"content":' in content_str:
                            # Extract the text from the content field
                            import json
                            parsed = json.loads(content_str)
                            if isinstance(parsed, dict) and 'content' in parsed:
                                content = parsed['content']
                                if isinstance(content, list) and len(content) > 0:
                                    if isinstance(content[0], dict) and 'text' in content[0]:
                                        search_terms_text = content[0]['text']
                except Exception as e:
                    logger.error(f"Error parsing search terms: {e}")
                    return f"Error parsing search terms: {e}"
            
            # Set the query parameter
            arguments["query"] = search_terms_text
            logger.info(f"Using search terms for web search: {search_terms_text}")
###
                    
        else:
            # Normal parameter handling for other tools
            for i, (param_name, param_info) in enumerate(schema.items()):
                if i >= len(params):
                    raise ValueError(f"Not enough parameters for {tool_name}")
                value = params[i]
                
                # Clean up file paths if this is an image_path parameter
                if param_name == "image_path":
                    # Remove any parameter name prefix
                    if '=' in value:
                        value = value.split('=')[1]
                    # Normalize the path
                    value = os.path.normpath(value)
                    logger.info(f"Cleaned and normalized image path: {value}")
                
                arguments[param_name] = value

        logger.info(f"Calling {tool_name} with arguments: {arguments}")
        result = await session.call_tool(tool_name, arguments=arguments)
        
        # Process result
        texts = []
        if hasattr(result, 'content'):
            for item in result.content:
                if hasattr(item, 'text'):
                    texts.append(item.text)
                else:
                    texts.append(str(item))
        else:
            texts = [str(result)]

        result_str = "[" + ", ".join(texts) + "]"
        logger.info(f"Tool result: {result_str}")
        
        # Update state based on tool name
        if tool_name == "describe_visual_elements":
            state.update('visual_elements', result_str)
        elif tool_name == "describe_style_or_aesthetics":
            state.update('style_analysis', result_str)
        elif tool_name == "describe_possible_scenario":
            state.update('scenario_analysis', result_str)
        elif tool_name == "generate_search_terms":
            state.update('search_terms', result_str)
        elif tool_name == "search_web":
            state.update('web_findings', result_str)
        elif tool_name == "infer_context":
            state.update('context_inference', result_str)
        elif tool_name == "format_final_output":
            state.update('final_output', result_str)
            
        logger.info(f"{'='*50}")
        logger.info(f"Completed execution of {tool_name}")
        logger.info(f"{'='*50}\n")
        return result_str
        
    except Exception as e:
        error_msg = f"Error executing {tool_name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

async def process_image(image_path: str):
    """Main workflow for processing an image."""
    # Normalize the image path
    image_path = os.path.normpath(image_path)
    logger.info(f"Starting context detection for image: {image_path}")
    state = WorkflowState()
    
    # Add logging for state updates
    def log_state_update(key: str, value: Any):
        logger.info(f"State update - {key}:")
        logger.info(f"Previous value: {getattr(state, key, None)}")
        logger.info(f"New value: {value}")
        state.update(key, value)
    
    server_params = StdioServerParameters(command="python", args=["example2-detective.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Get available tools
            tools_result = await session.list_tools()
            tools = tools_result.tools
            
            # Format tools for the prompt
            tools_block = format_tools_for_prompt(tools)
            system_prompt = SYSTEM_PROMPT.replace('{tools_block}', tools_block)
            
            # Main workflow loop
            while state.iteration < 5:  # Max 5 iterations
                logger.info(f"Starting iteration {state.iteration + 1}")
                
                try:
                    # Generate next action
                    prompt = construct_prompt(system_prompt, image_path, state)
                    logger.info("Generated prompt for Gemini:")
                    logger.info(prompt)
                    
                    response = await generate_with_timeout(prompt)
                    logger.info("Received response from Gemini:")
                    logger.info(response)
                    
                    # Check for final answer
                    if response.strip().startswith("FINAL_ANSWER:"):
                        answer = response.strip().replace("FINAL_ANSWER:", "").strip()
                        logger.info(f"Received final answer: {answer}")
                        log_state_update('final_output', answer)
                        break
                    
                    # Process function calls
                    function_lines = [
                        line.strip() for line in response.splitlines()
                        if line.strip().startswith("FUNCTION_CALL:")
                    ]
                    
                    if not function_lines:
                        logger.warning("No valid FUNCTION_CALL lines detected")
                        break
                    
                    # Execute each function call
                    for func_line in function_lines:
                        logger.info(f"Processing function call: {func_line}")
                        _, call = func_line.split(":", 1)
                        parts = [p.strip() for p in call.split("|") if p.strip()]
                        func_name = parts[0]
                        params = parts[1:] if len(parts) > 1 else []
                        
                        logger.info(f"Function: {func_name}")
                        logger.info(f"Parameters: {params}")
                        
                        result = await execute_tool_call(session, func_name, params, tools, state)
                        logger.info(f"Tool execution result: {result}")
                        
                        if result and result.startswith("Error"):
                            state.errors.append(result)
                        else:
                            # Update state based on the tool that was called
                            if func_name == "describe_visual_elements":
                                log_state_update('visual_elements', result)
                            elif func_name == "describe_style_or_aesthetics":
                                log_state_update('style_analysis', result)
                            elif func_name == "describe_possible_scenario":
                                log_state_update('scenario_analysis', result)
                            elif func_name == "generate_search_terms":
                                log_state_update('search_terms', result)
                            elif func_name == "search_web":
                                log_state_update('web_findings', result)
                                # After search_web completes successfully, automatically trigger infer_context
                                if state.visual_elements and state.style_analysis and state.scenario_analysis and not state.context_inference:
                                    logger.info("\n=== Automatically triggering infer_context after search_web ===")
                                    # Extract text from each state value
                                    visual_elements_text = extract_text_content(state.visual_elements)
                                    style_analysis_text = extract_text_content(state.style_analysis)
                                    scenario_analysis_text = extract_text_content(state.scenario_analysis)
                                    web_findings_text = extract_text_content(state.web_findings)
                                    
                                    # Handle empty web findings
                                    if not web_findings_text or web_findings_text == "[]":
                                        web_findings_text = "No relevant web findings were found for this image."

                                    infer_context_result = await execute_tool_call(
                                        session,
                                        "infer_context",
                                        [
                                            visual_elements_text,
                                            style_analysis_text,
                                            scenario_analysis_text,
                                            web_findings_text
                                        ],
                                        tools,
                                        state
                                    )
                                    if infer_context_result and not infer_context_result.startswith("Error"):
                                        log_state_update('context_inference', infer_context_result)
                            elif func_name == "infer_context":
                                log_state_update('context_inference', result)
                            elif func_name == "format_final_output":
                                log_state_update('final_output', result)
                                # Log detailed information about the final output
                                try:
                                    # Parse the JSON string
                                    output_json = json.loads(result)
                                    logger.info("\n=== Final Output Structure ===")
                                    logger.info(f"Type: {type(output_json)}")
                                    logger.info("\nKeys and Values:")
                                    for key, value in output_json.items():
                                        logger.info(f"\nKey: {key}")
                                        logger.info(f"Type: {type(value)}")
                                        logger.info(f"Value: {value}")
                                except Exception as e:
                                    logger.error(f"Error parsing final output: {e}")
                    
                    state.iteration += 1
                    
                except Exception as e:
                    logger.error(f"Error in iteration {state.iteration + 1}: {str(e)}", exc_info=True)
                    state.errors.append(str(e))
                    break


def extract_text_content(value):
    """Extract text content from a state value that might be in JSON format."""
    if isinstance(value, str):
        try:
            if value.startswith('['):
                # Parse the JSON-like structure
                import json
                parsed = json.loads(value)
                if isinstance(parsed, list) and len(parsed) > 0:
                    if isinstance(parsed[0], dict) and 'content' in parsed[0]:
                        content = parsed[0]['content']
                        if isinstance(content, list) and len(content) > 0:
                            if isinstance(content[0], dict) and 'text' in content[0]:
                                return content[0]['text']
        except Exception as e:
            logger.error(f"Error extracting text content: {e}")
    elif isinstance(value, list) and len(value) == 0:
        # Handle empty lists by returning a default string
        return "No content available"
    return str(value)  # Convert any other type to string

async def main():
    """Main entry point."""
    try:
        if len(sys.argv) < 2:
            logger.error("No image path provided")
            return
        
        image_path = sys.argv[1]
        logger.info(f"Attempting to open image at: {image_path}")
        await process_image(image_path)
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())