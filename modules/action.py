import json
import logging
from typing import List, Dict, Any
from mcp.types import TextContent

logger = logging.getLogger("ContextDetective.Action")

class ActionModule:
    """Handles formatting the final output."""
    
    def __init__(self):
        logger.info("Action module initialized")
        
    async def format_final_output(
        self,
        context_guess: str,
        confidence: float,
        explanation: str,
        related_links: List[str],
        search_terms: List[str]
    ) -> Dict[str, Any]:
        """Format the final output in the required JSON structure."""
        try:
            logger.info("Formatting final output...")
            
            # Validate confidence
            try:
                confidence_float = float(confidence)
                if not 0 <= confidence_float <= 1:
                    raise ValueError("Confidence must be between 0 and 1")
            except ValueError as ve:
                logger.warning(f"Invalid confidence value '{confidence}', setting to 0.5. Error: {ve}")
                confidence_float = 0.5
                
            # Ensure links and terms are lists
            if not isinstance(related_links, list):
                related_links = [str(related_links)]
            if not isinstance(search_terms, list):
                search_terms = [str(search_terms)]

            output = {
                "context_guess": str(context_guess),
                "confidence": confidence_float,
                "explanation": str(explanation),
                "related_links": related_links,
                "search_terms_used": search_terms
            }
            
            output_json = json.dumps(output, indent=2)
            logger.info("Final output formatted successfully")
            
            return {
                "content": [TextContent(type="text", text=output_json)]
            }
            
        except Exception as e:
            error_msg = f"Error formatting final output: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Return error message in the expected format
            error_output = {
                "error": error_msg,
                "context_guess": "Error during formatting",
                "confidence": 0.0,
                "explanation": f"Failed to format output: {error_msg}",
                "related_links": [],
                "search_terms_used": []
            }
            return {
                "content": [TextContent(type="text", text=json.dumps(error_output, indent=2))]
            }
