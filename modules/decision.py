import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from mcp.types import TextContent

logger = logging.getLogger("ContextDetective.Decision")

class DecisionModule:
    """Handles search term generation, web search, and context inference."""
    
    def __init__(self, api_key: str = None):
        """Initialize the decision module with Gemini API."""
        from google import genai
        
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.client = genai.Client(api_key=api_key)
        logger.info("Decision module initialized")

    def _get_user_preferences(self):
        """Get user preferences from environment."""
        try:
            prefs_json = os.getenv("USER_PREFERENCES", '{"interests": ["general knowledge"]}')
            return json.loads(prefs_json)
        except:
            logger.warning("Failed to parse user preferences, using defaults")
            return {"interests": ["general knowledge"]}

    async def generate_search_terms(self, descriptions: List[str]) -> Dict[str, Any]:
        """Generate relevant search terms from the image descriptions."""
        try:
            logger.info("Generating search terms...")
            
            # Combine descriptions into a single context string
            context = "\n\n".join(descriptions)
            
            user_prefs = self._get_user_preferences()
            interests = ", ".join(user_prefs.get("interests", []))
            
            prompt = f"""
            Based on the following combined image analyses, generate 5-7 specific search terms:
            
            ANALYSES:
            {context}
            
            USER INTERESTS:
            {interests}
            
            Return only the search terms, one per line. Choose terms that would help identify the specific context of this image.
            """
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            search_terms = [term.strip() for term in response.text.strip().split('\n') if term.strip()]
            logger.info(f"Generated {len(search_terms)} search terms")
            
            return {
                "content": [TextContent(type="text", text="\n".join(search_terms))]
            }
            
        except Exception as e:
            error_msg = f"Error generating search terms: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "content": [TextContent(type="text", text=f"❌ {error_msg}")]
            }
            
    async def search_web(self, query: str) -> Dict[str, Any]:
        """Search the web for information related to the query."""
        try:
            logger.info(f"Performing web search for query: {query}")
            
            all_results = []
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            search_url = f"https://api.duckduckgo.com/?q={query}&format=json"
            logger.info(f"Searching: {search_url}")
            
            response = requests.get(search_url, headers=headers)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Extract relevant information
                    if data.get('Abstract'):
                        all_results.append({
                            'query': query,
                            'title': 'Abstract',
                            'content': data['Abstract'],
                            'url': data.get('AbstractURL', 'https://duckduckgo.com')
                        })
                    
                    if data.get('RelatedTopics'):
                        for topic in data['RelatedTopics'][:3]: # Limit results
                            if isinstance(topic, dict) and 'Text' in topic and 'FirstURL' in topic:
                                all_results.append({
                                    'query': query,
                                    'title': topic.get('Text', '').split(' - ')[0],
                                    'content': topic['Text'],
                                    'url': topic['FirstURL']
                                })
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON for query: {query}")
            
            logger.info(f"Found {len(all_results)} results for query: {query}")
            
            return {
                "content": [TextContent(type="text", text=json.dumps(all_results, indent=2))]
            }
            
        except Exception as e:
            error_msg = f"Error performing web search: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "content": [TextContent(type="text", text=f"❌ {error_msg}")]
            }
            
    async def infer_context(
        self,
        visual_elements: str,
        style_analysis: str,
        scenario_analysis: str,
        web_findings: str
    ) -> Dict[str, Any]:
        """Combine all analyses to infer the most likely context."""
        try:
            logger.info("Inferring context...")
            
            user_prefs = self._get_user_preferences()
            interests = ", ".join(user_prefs.get("interests", []))
            
            prompt = f"""
            Based on the following analyses and web findings, infer the most likely context of the image:

            VISUAL ELEMENTS:
            {visual_elements}

            STYLE ANALYSIS:
            {style_analysis}

            SCENARIO ANALYSIS:
            {scenario_analysis}

            WEB FINDINGS:
            {web_findings}
            
            USER INTERESTS:
            {interests}

            Please provide:
            1. A clear statement of the most likely context
            2. A confidence score (0-1) based on the coherence and evidence
            3. A detailed explanation of your reasoning
            4. Key supporting evidence from the analyses and web findings
            """
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            logger.info("Context inference complete")
            
            return {
                "content": [TextContent(type="text", text=response.text)]
            }
            
        except Exception as e:
            error_msg = f"Error inferring context: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "content": [TextContent(type="text", text=f"❌ {error_msg}")]
            }