import os
import sys
import json
import aiohttp
import asyncio
import requests
import PIL.Image
from google import genai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.genai import types
from mcp.types import TextContent
from typing import List, Dict, Any, Tuple
from mcp.server.fastmcp.prompts import base
from mcp.server.fastmcp import FastMCP, Image
from typing import List, Dict, Any, Optional
from mcp import ClientSession, StdioServerParameters

import logging
import datetime
from pathlib import Path

def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"context_detective_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create logger
    logger = logging.getLogger("ContextDetective")
    logger.setLevel(logging.INFO)
    
    return logger


logger = setup_logging()

# Load environment variables
load_dotenv()

# Initialize MCP
mcp = FastMCP("ContextDetective")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """You are an agentic context detector that analyzes images to determine their context, meaning, and significance.

### Tool Usage:
- FUNCTION_CALL: tool_name|param1|param2|... (one per line)
- Include 'FUNCTION_CALL:', function_name|, and parameters (if necessary)
- FINAL_ANSWER: result

### Workflow Steps:
1. Analyze the image using multiple perspectives:
   - Visual elements (objects, people, colors, text)
   - Style and aesthetics (artistic style, cultural elements)
   - Possible scenarios (what might be happening)
2. Generate search terms based on findings
3. Explore the web for supporting evidence
4. Infer the most likely context
5. Provide structured output with confidence rating

### Tools:
{tools_block}

### Response Format:
For each step, use FUNCTION_CALL to execute tools.
When complete, provide FINAL_ANSWER with the structured output.
"""

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
        return f"""Current Progress:
- Visual Elements Analysis: {'✅' if self.visual_elements else '❌'}
- Style Analysis: {'✅' if self.style_analysis else '❌'}
- Scenario Analysis: {'✅' if self.scenario_analysis else '❌'}
- Search Terms: {'✅' if self.search_terms else '❌'}
- Web Findings: {'✅' if self.web_findings else '❌'}
- Context Inference: {'✅' if self.context_inference else '❌'}
- Final Output: {'✅' if self.final_output else '❌'}"""


@mcp.tool()
async def describe_visual_elements(image_path: str) -> Dict[str, Any]:
    """Analyze and describe the visual elements in an image."""
    try:
        logger.info(f"Input image_path: {image_path}")
        if '=' in image_path:
            image_path = image_path.split('=')[1]
            logger.info(f"Cleaned image_path: {image_path}")
        image_path = os.path.normpath(image_path)
        logger.info(f"Normalized image_path: {image_path}")
        
        # Load and prepare the image
        logger.info(f"Attempting to open image at: {image_path}")
        image = PIL.Image.open(image_path)
        
        # Generate content with Gemini
        logger.info("Sending request to Gemini API to describe visual elements")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Describe the visual elements in this image in detail. Focus on objects, people, colors, text, and structural elements.", image]
        )
        logger.info("Successfully received response from Gemini")
        
        return {
            "content": [
                TextContent(
                    type="text",
                    text=response.text
                )
            ]
        }
    except Exception as e:
        error_msg = f"Error analyzing visual elements: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "content": [
                TextContent(
                    type="text",
                    text=f"❌ {error_msg}"
                )
            ]
        }

@mcp.tool()
async def describe_style_or_aesthetics(image_path: str) -> Dict[str, Any]:
    """Analyze and describe the style and aesthetic features of an image."""
    try:
        # Load and prepare the image
        image = PIL.Image.open(image_path)
        
        # Generate content with Gemini
        logger.info("Sending request to Gemini API to describe style/aesthetics")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Analyze the style and aesthetics of this image. Consider artistic style, cultural elements, composition, and visual techniques.", image]
        )
        logger.info("Successfully received response from Gemini")
        return {
            "content": [
                TextContent(
                    type="text",
                    text=response.text
                )
            ]
        }
    except Exception as e:
        logger.error(f"Error in describe_style_or_aesthetics: {str(e)}", exc_info=True)
        return {
            "content": [
                TextContent(
                    type="text",
                    text=f"❌ Error analyzing style: {e}"
                )
            ]
        }

@mcp.tool()
async def describe_possible_scenario(image_path: str) -> Dict[str, Any]:
    """Analyze and describe possible scenarios or contexts for the image."""
    try:
        # Load and prepare the image
        image = PIL.Image.open(image_path)
        
        # Generate content with Gemini
        logger.info("Sending request to Gemini API to describe possible_scenario")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["What might this image be depicting? Consider the context, setting, and possible scenarios. What story or situation might this image represent?", image]
        )
        logger.info("Successfully received response from Gemini")
        
        return {
            "content": [
                TextContent(
                    type="text",
                    text=response.text
                )
            ]
        }
    except Exception as e:
        logger.error(f"Error in possible_scenario: {str(e)}", exc_info=True)
        return {
            "content": [
                TextContent(
                    type="text",
                    text=f"❌ Error analyzing scenario: {e}"
                )
            ]
        }

async def execute_tool_call(
    session: ClientSession,
    tool_name: str,
    params: List[str],
    tools: List[Any],
    state: WorkflowState
) -> Optional[str]:
    """Execute a single tool call and return the result."""
    logger.info(f"Executing tool: {tool_name}")
    logger.info(f"Parameters received: {params}")
    logger.info("Current analysis state:")
    logger.info(f"- Visual Elements: {state.visual_elements}")
    logger.info(f"- Style Analysis: {state.style_analysis}")
    logger.info(f"- Scenario Analysis: {state.scenario_analysis}")
    
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
            
        # Special handling for search_web
        elif tool_name == "search_web":
            logger.info("Processing web search")
            if not state.search_terms:
                logger.error("No search terms available for web search")
                return "Error: No search terms available for web search"
                
            # Use the most relevant search terms from the previous step
            search_query = params[0] if params else state.search_terms
            logger.info(f"Using search query: {search_query}")
            arguments["query"] = search_query
            
        else:
            # Normal parameter handling for other tools
            for i, (param_name, param_info) in enumerate(schema.items()):
                if i >= len(params):
                    raise ValueError(f"Not enough parameters for {tool_name}")
                value = params[i]
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
        return result_str
        
    except Exception as e:
        error_msg = f"Error executing {tool_name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg
    
@mcp.tool()
async def generate_search_terms(descriptions: List[str]) -> Dict[str, Any]:
    """Generate relevant search terms from the image descriptions."""
    try:
        logger.info("\n=== Starting Search Terms Generation ===")
        logger.info("Raw input descriptions:")
        for i, desc in enumerate(descriptions):
            logger.info(f"\nDescription {i+1}:")
            logger.info(desc)
        
        # Get the context from the input
        context = descriptions[0] if descriptions else ""
        
        prompt = f"""Based on the following image analyses, generate specific search terms to identify this image's context.

{context}

Generate search terms in these categories:
WHO: [key people, groups, or organizations visible or implied]
WHAT: [specific objects, actions, or events shown]
WHERE: [location, setting, or place indicated]
WHEN: [time period, era, or date suggested]
WHY: [purpose, cause, or significance of the scene]
HOW: [methods, techniques, or manner shown]

Important:
- Be specific and precise
- Use terms directly from the analyses
- Focus on what is actually shown/described
- Avoid speculation or unrelated terms
- If a category doesn't have clear evidence, leave it empty

Format your response as:
WHO: term1, term2, ...
WHAT: term1, term2, ...
WHERE: term1, term2, ...
WHEN: term1, term2, ...
WHY: term1, term2, ...
HOW: term1, term2, ..."""

        logger.info("\nGenerated prompt for Gemini:")
        logger.info(f"{prompt}\n")
        
        # Generate content with Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        logger.info("Received response from Gemini:")
        logger.info(f"{response.text}\n")
        logger.info("=== End Search Terms Generation ===\n")
        
        return {
            "content": [
                TextContent(
                    type="text",
                    text=response.text
                )
            ]
        }
    except Exception as e:
        logger.error(f"Error in generate_search_terms: {str(e)}", exc_info=True)
        return {
            "content": [
                TextContent(
                    type="text",
                    text=f"❌ Error generating search terms: {e}"
                )
            ]
        }

@mcp.tool()
async def search_web(query: str) -> Dict[str, Any]:
    """Search the web for information related to the query."""
    try:
        logger.info(f"Starting web search with query: {query}")
        
        # Parse the structured search terms if available
        search_categories = {}
        search_queries = []
        
        if "WHO:" in query or "WHAT:" in query:  # Check if it's our structured format
            categories = query.split('\n')
            for category in categories:
                if ':' in category:
                    key, terms = category.split(':', 1)
                    search_categories[key.strip()] = terms.strip()
            
            # Create focused search queries for each category
            for category, terms in search_categories.items():
                if terms and terms != "[terms]":
                    search_queries.append(f"{category} {terms}")
        
        # Use either structured queries or original query
        queries = search_queries if search_queries else [query]
        logger.info(f"Processed search queries: {queries}")
        
        all_results = []
        async with aiohttp.ClientSession() as session:
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            for search_query in queries:
                search_url = f"https://api.duckduckgo.com/?q={search_query}&format=json"
                logger.info(f"Searching: {search_url}")
                
                async with session.get(search_url, headers=headers) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            
                            # Extract relevant information
                            if data.get('Abstract'):
                                all_results.append({
                                    'query': search_query,
                                    'title': 'Abstract',
                                    'content': data['Abstract'],
                                    'url': data.get('AbstractURL')
                                })
                            
                            if data.get('RelatedTopics'):
                                for topic in data['RelatedTopics'][:3]:  # Limit to top 3 per query
                                    if 'Text' in topic and 'FirstURL' in topic:
                                        all_results.append({
                                            'query': search_query,
                                            'title': topic.get('Text', '').split(' - ')[0],
                                            'content': topic['Text'],
                                            'url': topic['FirstURL']
                                        })
                        except aiohttp.ContentTypeError:
                            logger.warning(f"Failed to parse JSON for query: {search_query}")
                            continue
            
            logger.info(f"Found {len(all_results)} total results")
            return {
                "content": [
                    TextContent(
                        type="text",
                        text=json.dumps(all_results, indent=2)
                    )
                ]
            }
    except Exception as e:
        logger.error(f"Error in web search: {str(e)}", exc_info=True)
        return {
            "content": [
                TextContent(
                    type="text",
                    text=f"❌ Error performing web search: {e}"
                )
            ]
        }
    


# Link fetching implementation
@mcp.tool()
async def fetch_links_and_snippets(query: str) -> Dict[str, Any]:
    """Fetch and extract relevant information from search results."""
    logger.info("Fetching Links and Snippets")
    try:
        async with aiohttp.ClientSession() as session:
            # First get search results
            search_url = f"https://api.duckduckgo.com/?q={query}&format=json"
            async with session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    # Process each result
                    for topic in data.get('RelatedTopics', [])[:3]:  # Limit to top 3
                        if 'FirstURL' in topic:
                            try:
                                async with session.get(topic['FirstURL']) as page_response:
                                    if page_response.status == 200:
                                        html = await page_response.text()
                                        soup = BeautifulSoup(html, 'html.parser')
                                        
                                        # Extract title and first few paragraphs
                                        title = soup.title.string if soup.title else "No title"
                                        paragraphs = [p.text.strip() for p in soup.find_all('p')[:3]]
                                        
                                        results.append({
                                            'url': topic['FirstURL'],
                                            'title': title,
                                            'snippets': paragraphs
                                        })
                            except Exception as e:
                                print(f"Error fetching {topic['FirstURL']}: {e}")
                    logger.info("Links and Snippets fetched")
                    return {
                        "content": [
                            TextContent(
                                type="text",
                                text=json.dumps(results, indent=2)
                            )
                        ]
                    }
                else:
                    raise Exception(f"Search API returned status {response.status}")
    except Exception as e:
        logger.error(f"Error in fetch_links_and_snippets: {str(e)}", exc_info=True)
        return {
            "content": [
                TextContent(
                    type="text",
                    text=f"❌ Error fetching links and snippets: {e}"
                )
            ]
        }

# Context inference implementation
@mcp.tool()
async def infer_context(
    visual_elements: str,
    style_analysis: str,
    scenario_analysis: str,
    web_findings: str
) -> Dict[str, Any]:
    """Combine all analyses to infer the most likely context."""
    try:
        # Prepare the prompt for Gemini
        prompt = f"""Based on the following analyses, infer the most likely context of the image:

Visual Elements:
{visual_elements}

Style Analysis:
{style_analysis}

Scenario Analysis:
{scenario_analysis}

Web Findings:
{web_findings}

Please provide:
1. A clear statement of the most likely context
2. A confidence score (0-1)
3. A detailed explanation of your reasoning
4. Key supporting evidence from the analyses
"""
        
        # Generate content with Gemini
        logger.info("Sending request to Gemini API to infer context")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        return {
            "content": [
                TextContent(
                    type="text",
                    text=response.text
                )
            ]
        }
    except Exception as e:
        logger.error(f"Error in infer_context: {str(e)}", exc_info=True)
        return {
            "content": [
                TextContent(
                    type="text",
                    text=f"❌ Error inferring context: {e}"
                )
            ]
        }

# Output formatting implementation
@mcp.tool()
async def format_final_output(
    context_guess: str,
    confidence: float,
    explanation: str,
    related_links: List[str],
    search_terms: List[str]
) -> Dict[str, Any]:
    """Format the final output in the required JSON structure."""
    try:
        output = {
            "context_guess": context_guess,
            "confidence": confidence,
            "explanation": explanation,
            "related_links": related_links,
            "search_terms_used": search_terms
        }
        
        # Validate the output
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be a number between 0 and 1")
        
        if not isinstance(related_links, list) or not isinstance(search_terms, list):
            raise ValueError("related_links and search_terms must be lists")
        logger.info("Return final output")
        return {
            "content": [
                TextContent(
                    type="text",
                    text=json.dumps(output, indent=2)
                )
            ]
        }
    except Exception as e:
        logger.error(f"Error in format_final_output: {str(e)}", exc_info=True)
        return {
            "content": [
                TextContent(
                    type="text",
                    text=f"❌ Error formatting output: {e}"
                )
            ]
        }

# --------------------
# System Prompts
# --------------------
@mcp.prompt()
def get_system_prompt() -> str:
    return """You are an agentic context detector that analyzes images to determine their context, meaning, and significance.
                Your workflow:
                1. Analyze visual elements, style, and possible scenarios
                2. Generate search terms based on your analysis
                3. Explore the web for supporting evidence
                4. Infer the most likely context
                5. Provide a structured output with explanation and confidence

                Always maintain a skeptical and analytical mindset, considering multiple possibilities before reaching conclusions."""

if __name__ == "__main__":
    print("MCP Server Starting...")
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        mcp.run()
    else:
        mcp.run(transport="stdio")