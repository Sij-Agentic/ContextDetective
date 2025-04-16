import os
import json
import logging
from PIL import Image
from typing import Dict, Any, List
from mcp.types import TextContent

logger = logging.getLogger("ContextDetective.Perception")

class PerceptionModule:
    """Handles visual analysis of images using Gemini API."""
    
    def __init__(self, api_key: str = None):
        """Initialize the perception module with Gemini API."""
        from google import genai
        
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.client = genai.Client(api_key=api_key)
        logger.info("Perception module initialized")
    
    def _get_user_preferences(self):
        """Get user preferences from environment."""
        try:
            prefs_json = os.getenv("USER_PREFERENCES", '{"interests": ["general knowledge"]}')
            return json.loads(prefs_json)
        except:
            logger.warning("Failed to parse user preferences, using defaults")
            return {"interests": ["general knowledge"]}
    
    async def describe_visual_elements(self, image_path: str) -> Dict[str, Any]:
        """Analyze and describe the visual elements in an image."""
        try:
            logger.info(f"Analyzing visual elements in {image_path}")
            
            # Clean up path if needed
            if '=' in image_path:
                image_path = image_path.split('=')[1]
            image_path = os.path.normpath(image_path)
            
            # Load image
            image = Image.open(image_path)
            
            # Get user preferences
            user_prefs = self._get_user_preferences()
            interests = ", ".join(user_prefs.get("interests", []))
            
            # Enhanced prompt with user preferences
            prompt = f"""
            Describe the visual elements in this image in detail. 
            Focus on objects, people, colors, text, and structural elements.
            
            The user is interested in: {interests}. Consider this when analyzing the image.
            
            Format your response to clearly identify:
            - Objects: List all visible objects
            - Colors: Describe dominant colors
            - Text: Transcribe any visible text
            - People: Describe any people present
            """
            
            # Generate content with Gemini
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, image]
            )
            
            logger.info("Visual elements analysis complete")
            
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
    
    async def describe_style_or_aesthetics(self, image_path: str) -> Dict[str, Any]:
        """Analyze and describe the style and aesthetic features of an image."""
        try:
            logger.info(f"Analyzing style and aesthetics in {image_path}")
            
            # Clean up path if needed
            if '=' in image_path:
                image_path = image_path.split('=')[1]
            image_path = os.path.normpath(image_path)
            
            # Load image
            image = Image.open(image_path)
            
            # Get user preferences
            user_prefs = self._get_user_preferences()
            interests = ", ".join(user_prefs.get("interests", []))
            
            # Enhanced prompt with user preferences
            prompt = f"""
            Analyze the style and aesthetics of this image.
            Consider artistic style, cultural elements, composition, and visual techniques.
            
            The user is interested in: {interests}. Consider this when analyzing the style.
            
            Format your response to clearly identify:
            - Artistic Style: The overall artistic style
            - Composition: How the image is composed
            - Cultural Elements: Any cultural influences visible
            """
            
            # Generate content with Gemini
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, image]
            )
            
            logger.info("Style and aesthetics analysis complete")
            
            return {
                "content": [
                    TextContent(
                        type="text",
                        text=response.text
                    )
                ]
            }
        except Exception as e:
            error_msg = f"Error analyzing style/aesthetics: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "content": [
                    TextContent(
                        type="text",
                        text=f"❌ {error_msg}"
                    )
                ]
            }
    
    async def describe_possible_scenario(self, image_path: str) -> Dict[str, Any]:
        """Analyze and describe possible scenarios or contexts for the image."""
        try:
            logger.info(f"Analyzing possible scenarios in {image_path}")
            
            # Clean up path if needed
            if '=' in image_path:
                image_path = image_path.split('=')[1]
            image_path = os.path.normpath(image_path)
            
            # Load image
            image = Image.open(image_path)
            
            # Get user preferences
            user_prefs = self._get_user_preferences()
            interests = ", ".join(user_prefs.get("interests", []))
            
            # Enhanced prompt with user preferences
            prompt = f"""
            What might this image be depicting? Consider the context, setting, and possible scenarios.
            What story or situation might this image represent?
            
            The user is interested in: {interests}. Consider this when analyzing the scenario.
            
            Format your response to clearly identify:
            - Possible Scenario: The main scenario depicted
            - Setting: The environment or location
            - Activity: What might be happening
            """
            
            # Generate content with Gemini
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, image]
            )
            
            logger.info("Scenario analysis complete")
            
            return {
                "content": [
                    TextContent(
                        type="text",
                        text=response.text
                    )
                ]
            }
        except Exception as e:
            error_msg = f"Error analyzing scenario: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "content": [
                    TextContent(
                        type="text",
                        text=f"❌ {error_msg}"
                    )
                ]
            }