import os
import json
import logging
import hashlib
import chromadb
import requests
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from mcp.types import TextContent
from chromadb.utils import embedding_functions

logger = logging.getLogger("ContextDetective.Memory")

class MemoryModule:
    """Handles storage and retrieval of analysis data with dual-layer memory system."""
    
    def __init__(self, storage_path: str = "memory_storage", ollama_url: str = "http://localhost:11434"):
        """Initialize the memory module with both file-based and vector storage."""
        # File-based storage (legacy)
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.index_file = self.storage_path / "index.json"
        self._index = self._load_index()
        
        # Short-term memory (session-based)
        self.short_term = {}
        
        # Vector database (ChromaDB)
        self.chroma_path = Path(storage_path) / "chromadb"
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
        
        # Create collections if they don't exist
        self.collections = {
            "complete_analysis": self.chroma_client.get_or_create_collection(
                name="complete_analysis",
                metadata={"description": "Complete analysis results"}
            ),
            "visual_elements": self.chroma_client.get_or_create_collection(
                name="visual_elements",
                metadata={"description": "Visual elements analysis"}
            ),
            "style_analysis": self.chroma_client.get_or_create_collection(
                name="style_analysis", 
                metadata={"description": "Style analysis"}
            ),
            "scenario_analysis": self.chroma_client.get_or_create_collection(
                name="scenario_analysis",
                metadata={"description": "Scenario analysis"}
            ),
            "inferred_contexts": self.chroma_client.get_or_create_collection(
                name="inferred_contexts",
                metadata={"description": "Inferred contexts"}
            )
        }
        
        # Initialize Ollama embedding function
        self.ollama_url = ollama_url
        self.model_name = "mxbai-embed-large"
        
        # Log initialization and memory statistics
        logger.info(f"🚀 Memory module initializing...")
        logger.info(f"   └─ File storage: {self.storage_path}")
        logger.info(f"   └─ ChromaDB: {self.chroma_path}")
        logger.info(f"   └─ Ollama URL: {self.ollama_url}")
        
        # Log statistics by calling our new method
        self.log_memory_statistics()
    
    def _load_index(self) -> Dict[str, str]:
        """Load the index of stored analyses."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Index file corrupted, creating new index")
                return {}
        else:
            logger.info("No index file found, creating new index")
            return {}
    
    def _save_index(self):
        """Save the current index to disk."""
        with open(self.index_file, 'w') as f:
            json.dump(self._index, f)
    
    def _compute_image_hash(self, image_path: str) -> str:
        """Compute a hash for an image to uniquely identify it."""
        logger.info(f"[HASH] Starting hash computation for: {image_path}")
        start_time = time.time()
        try:
            logger.info(f"Computing hash for image at {image_path}")
            
            # Verify file exists first
            if not os.path.exists(image_path):
                logger.error(f"File not found: {image_path}")
                return "error_file_not_found"
            
            # Verify file is readable
            if not os.access(image_path, os.R_OK):
                logger.error(f"File not readable: {image_path}")
                return "error_file_not_readable"
            
            # Get file size for debug info
            file_size = os.path.getsize(image_path)
            logger.info(f"File size: {file_size} bytes")
            
            with open(image_path, 'rb') as f:
                image_data = f.read()
                image_hash = hashlib.md5(image_data).hexdigest()
                logger.info(f"Computed hash {image_hash} for {image_path}")
                logger.info(f"[HASH] Completed hash computation for: {image_path} in {time.time() - start_time:.3f}s")
                return image_hash
        except FileNotFoundError:
            logger.error(f"File not found: {image_path}")
            return "error_file_not_found"
        except PermissionError:
            logger.error(f"Permission denied for: {image_path}")
            return "error_permission_denied"
        except Exception as e:
            logger.error(f"Error computing hash for {image_path}: {e}", exc_info=True)
            return f"error_{str(e).replace(' ', '_')[:20]}"  # Return error code instead of throwing
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using Ollama's mxbai-embed-large model."""
        try:
            url = f"{self.ollama_url}/api/embeddings"
            response = requests.post(
                url,
                json={"model": self.model_name, "prompt": text}
            )
            
            if response.status_code == 200:
                return response.json()["embedding"]
            else:
                logger.error(f"Error generating embedding: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []
    
    def store_in_short_term(self, session_id: str, component: str, data: Dict[str, Any]) -> None:
        """Store analysis component in short-term memory."""
        logger.info(f"[SHORT-TERM] Storing component '{component}' for session {session_id[:8]}")
        if session_id not in self.short_term:
            self.short_term[session_id] = {}
            logger.info(f"🧠 Created new short-term memory session: {session_id[:8]}")
        
        # Get data summary for logging
        data_summary = ""
        if isinstance(data, dict):
            keys = list(data.keys())
            data_summary = f"containing keys: {', '.join(keys[:3])}" + ("..." if len(keys) > 3 else "")
        elif isinstance(data, str):
            data_summary = f"text ({len(data)} chars)"
        else:
            data_summary = f"data of type {type(data).__name__}"
        
        self.short_term[session_id][component] = {
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"[SHORT-TERM] Stored '{component}' for session {session_id[:8]}")
    
    def get_from_short_term(self, session_id: str, component: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis component from short-term memory."""
        if session_id in self.short_term and component in self.short_term[session_id]:
            timestamp = self.short_term[session_id][component]["timestamp"]
            logger.info(f"🔍 SHORT-TERM: Retrieved '{component}' for session {session_id[:8]} (from {timestamp})")
            return self.short_term[session_id][component]["data"]
        
        logger.info(f"⚠️ SHORT-TERM: Component '{component}' not found for session {session_id[:8]}")
        return None
    
    def clear_short_term(self, session_id: str) -> None:
        """Clear short-term memory for a session."""
        if session_id in self.short_term:
            components = list(self.short_term[session_id].keys())
            del self.short_term[session_id]
            logger.info(f"🧹 SHORT-TERM: Cleared memory for session {session_id[:8]}")
            logger.info(f"   └─ Removed components: {', '.join(components)}")
        else:
            logger.info(f"⚠️ SHORT-TERM: No session {session_id[:8]} to clear")
    
    async def check_exact_match(self, image_hash: str) -> Dict[str, Any]:
        """Check if we have an exact match for this image hash."""
        operation = "EXACT_MATCH_CHECK"
        start_time = time.time()
        logger.info(f"[{operation}] START - Checking hash: {image_hash[:8]}")
        
        try:
            if image_hash in self._index:
                entry_path = self._index[image_hash]
                if os.path.exists(entry_path):
                    with open(entry_path, 'r') as f:
                        analysis_data = json.load(f)
                    
                    duration = time.time() - start_time
                    logger.info(f"[{operation}] SUCCESS - Found match for {image_hash[:8]}")
                    logger.info(f"[{operation}] DETAILS:")
                    logger.info(f"   └─ Timestamp: {analysis_data.get('timestamp', 'unknown')}")
                    logger.info(f"   └─ Context: {analysis_data.get('context_guess', 'unknown')}")
                    logger.info(f"   └─ Confidence: {analysis_data.get('confidence', 'unknown')}")
                    logger.info(f"[{operation}] END - Duration: {duration:.3f}s")
                    
                    return {
                        "match_found": True,
                        "content": [TextContent(type="text", text=json.dumps(analysis_data, indent=2))],
                        "data": analysis_data
                    }
            
            duration = time.time() - start_time
            logger.info(f"[{operation}] NO_MATCH - Hash {image_hash[:8]} not found")
            logger.info(f"[{operation}] END - Duration: {duration:.3f}s")
            return {
                "match_found": False,
                "content": [TextContent(type="text", text=f"No exact match found")]
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{operation}] ERROR - Failed to check match for {image_hash[:8]}: {str(e)}")
            logger.error(f"[{operation}] STACKTRACE:", exc_info=True)
            logger.info(f"[{operation}] END - Duration: {duration:.3f}s")
            return {
                "match_found": False,
                "content": [TextContent(type="text", text=f"Error during match check: {str(e)}")]
            }
    
    async def store_analysis(self, image_hash: str, analysis_json: str) -> Dict[str, Any]:
        """
        Store an analysis result in both file storage and vector database.
        
        Args:
            image_hash: Unique hash of the image.
            analysis_json: JSON string containing the complete analysis results.
        """
        logger.info(f"[LONG-TERM] Storing analysis for hash: {image_hash[:8]}")
        start_time = time.time()
        try:
            logger.info(f"💾 Storing analysis for image hash {image_hash[:8]}...")
            
            # Parse and validate JSON
            analysis_data = json.loads(analysis_json)
            
            # Add timestamp if not present
            if "timestamp" not in analysis_data:
                analysis_data["timestamp"] = datetime.now().isoformat()
            
            # Extract key information for logging
            context = analysis_data.get("context_guess", "unknown context")
            confidence = analysis_data.get("confidence", "unknown confidence")
            
            # Save to a file (legacy storage)
            entry_path = self.storage_path / f"{image_hash}.json"
            with open(entry_path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            logger.info(f"📁 LONG-TERM: Saved analysis to file storage: {entry_path}")
            
            # Update index
            self._index[image_hash] = str(entry_path)
            self._save_index()
            logger.info(f"📇 LONG-TERM: Updated index with entry for {image_hash[:8]}")
            
            # Extract components for vector storage
            complete_analysis = json.dumps(analysis_data)
            
            # Generate embeddings for the complete analysis
            logger.info(f"🔢 LONG-TERM: Generating embeddings for analysis...")
            complete_embedding = self._generate_embedding(complete_analysis)
            
            # Store in ChromaDB with appropriate metadata
            if complete_embedding:
                timestamp = datetime.now().isoformat()
                
                # Add to complete_analysis collection
                logger.info(f"🧮 LONG-TERM: Storing in ChromaDB complete_analysis collection...")
                self.collections["complete_analysis"].add(
                    ids=[image_hash],
                    embeddings=[complete_embedding],
                    documents=[complete_analysis],
                    metadatas=[{
                        "image_hash": image_hash,
                        "timestamp": timestamp,
                        "confidence": analysis_data.get("confidence", 0.0),
                        "context": context
                    }]
                )
                
                # Extract and store individual components if available
                components_stored = []
                
                if "visual_elements" in analysis_data:
                    visual_text = json.dumps(analysis_data["visual_elements"])
                    visual_embedding = self._generate_embedding(visual_text)
                    if visual_embedding:
                        logger.info(f"🧮 LONG-TERM: Storing in ChromaDB visual_elements collection...")
                        self.collections["visual_elements"].add(
                            ids=[f"{image_hash}_visual"],
                            embeddings=[visual_embedding],
                            documents=[visual_text],
                            metadatas=[{
                                "image_hash": image_hash,
                                "timestamp": timestamp
                            }]
                        )
                        components_stored.append("visual_elements")
                
                if "style_analysis" in analysis_data:
                    style_text = json.dumps(analysis_data["style_analysis"])
                    style_embedding = self._generate_embedding(style_text)
                    if style_embedding:
                        logger.info(f"🧮 LONG-TERM: Storing in ChromaDB style_analysis collection...")
                        self.collections["style_analysis"].add(
                            ids=[f"{image_hash}_style"],
                            embeddings=[style_embedding],
                            documents=[style_text],
                            metadatas=[{
                                "image_hash": image_hash,
                                "timestamp": timestamp
                            }]
                        )
                        components_stored.append("style_analysis")
                
                if "scenario_analysis" in analysis_data:
                    scenario_text = json.dumps(analysis_data["scenario_analysis"])
                    scenario_embedding = self._generate_embedding(scenario_text)
                    if scenario_embedding:
                        logger.info(f"🧮 LONG-TERM: Storing in ChromaDB scenario_analysis collection...")
                        self.collections["scenario_analysis"].add(
                            ids=[f"{image_hash}_scenario"],
                            embeddings=[scenario_embedding],
                            documents=[scenario_text],
                            metadatas=[{
                                "image_hash": image_hash,
                                "timestamp": timestamp
                            }]
                        )
                        components_stored.append("scenario_analysis")
                
                logger.info(f"✅ LONG-TERM: Vector embeddings stored for collections: complete_analysis, {', '.join(components_stored)}")
            else:
                logger.warning(f"⚠️ LONG-TERM: Failed to generate embeddings for {image_hash[:8]}")
            
            logger.info(f"✅ LONG-TERM: Analysis stored successfully for image {image_hash[:8]}")
            logger.info(f"   └─ Context: {context}")
            logger.info(f"   └─ Confidence: {confidence}")
            logger.info(f"   └─ Location: {entry_path}")
            
            logger.info(f"[LONG-TERM] Stored analysis for hash: {image_hash[:8]} in {time.time() - start_time:.3f}s")
            return {
                "content": [TextContent(type="text", text=f"Analysis for image {image_hash[:8]} stored successfully.")]
            }
            
        except json.JSONDecodeError:
            error_msg = "Invalid JSON format for analysis data."
            logger.error(f"❌ LONG-TERM: {error_msg}")
            logger.info(f"[LONG-TERM] Stored analysis for hash: {image_hash[:8]} in {time.time() - start_time:.3f}s")
            return {
                "content": [TextContent(type="text", text=f"❌ Error: {error_msg}")]
            }
        except Exception as e:
            error_msg = f"Error storing analysis: {str(e)}"
            logger.error(f"❌ LONG-TERM: {error_msg}", exc_info=True)
            logger.info(f"[LONG-TERM] Stored analysis for hash: {image_hash[:8]} in {time.time() - start_time:.3f}s")
            return {
                "content": [TextContent(type="text", text=f"❌ {error_msg}")]
            }

    async def retrieve_similar_analyses(self, query_text: str, top_k: int = 3) -> Dict[str, Any]:
        """Retrieve semantically similar analyses based on text query."""
        operation = "VECTOR_SEARCH"
        start_time = time.time()
        logger.info(f"[{operation}] START - Query: '{query_text[:50]}...'")
        logger.info(f"[{operation}] PARAMS - top_k: {top_k}")
        
        try:
            # Generate embedding
            logger.info(f"[{operation}] Generating query embedding...")
            query_embedding = self._generate_embedding(query_text)
            
            if not query_embedding:
                duration = time.time() - start_time
                logger.warning(f"[{operation}] FAILED - Could not generate embedding")
                logger.info(f"[{operation}] END - Duration: {duration:.3f}s")
                return {"content": [TextContent(type="text", text="Failed to generate embedding")]}
            
            # Query collections
            logger.info(f"[{operation}] Querying vector database...")
            results = self.collections["complete_analysis"].query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Log retrieval results
            found_count = len(results["documents"][0]) if results["documents"] else 0
            logger.info(f"[{operation}] RESULTS - Found {found_count} matches")
            
            # Format and log each result
            formatted_results = []
            for i, doc in enumerate(results["documents"][0]):
                try:
                    analysis = json.loads(doc)
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    similarity = results["distances"][0][i] if results["distances"] else None
                    
                    logger.info(f"[{operation}] MATCH #{i+1}:")
                    logger.info(f"   └─ Hash: {metadata.get('image_hash', 'unknown')[:8]}")
                    logger.info(f"   └─ Similarity: {similarity:.4f}")
                    logger.info(f"   └─ Context: {analysis.get('context_guess', 'unknown')[:50]}...")
                    
                    formatted_results.append({
                        "analysis": analysis,
                        "metadata": metadata,
                        "similarity": similarity
                    })
                except json.JSONDecodeError as e:
                    logger.warning(f"[{operation}] Failed to parse result #{i+1}: {str(e)}")
            
            duration = time.time() - start_time
            logger.info(f"[{operation}] SUCCESS - Retrieved {len(formatted_results)} results")
            logger.info(f"[{operation}] END - Duration: {duration:.3f}s")
            
            return {
                "content": [TextContent(type="text", text=json.dumps(formatted_results, indent=2))],
                "data": formatted_results
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{operation}] ERROR - {str(e)}")
            logger.error(f"[{operation}] STACKTRACE:", exc_info=True)
            logger.info(f"[{operation}] END - Duration: {duration:.3f}s")
            return {
                "content": [TextContent(type="text", text=f"Error during retrieval: {str(e)}")]
            }

    async def retrieve_memory_for_inference(self, current_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant past analyses for inference."""
        operation = "MEMORY_INFERENCE"
        start_time = time.time()
        logger.info(f"[{operation}] START - Retrieving past analyses for context enhancement")
        logger.info(f"[{operation}] CONTEXT - This operation searches for similar past analyses to enhance current inference")
        
        try:
            # Log the current analysis structure
            components = list(current_analysis.keys())
            logger.info(f"[{operation}] CURRENT_ANALYSIS:")
            logger.info(f"   └─ Available Components: {', '.join(components)}")
            for comp in components:
                content_preview = str(current_analysis[comp])[:100]
                logger.info(f"   └─ {comp} Preview: {content_preview}...")

            # Build and log query components
            query_components = []
            for comp in ['visual_elements', 'style_analysis', 'scenario_analysis']:
                if comp in current_analysis:
                    component_text = str(current_analysis[comp])
                    query_components.append(f"{comp}: {component_text}")
                    logger.info(f"[{operation}] QUERY_COMPONENT - {comp}:")
                    logger.info(f"   └─ Length: {len(component_text)} chars")
                    logger.info(f"   └─ Preview: {component_text[:100]}...")

            query_text = "\n\n".join(query_components)
            logger.info(f"[{operation}] SEARCH_QUERY:")
            logger.info(f"   └─ Components Used: {len(query_components)}")
            logger.info(f"   └─ Total Length: {len(query_text)} chars")
            
            # Vector search for similar analyses
            logger.info(f"[{operation}] SEARCHING - Querying vector database for similar past analyses")
            result = await self.retrieve_similar_analyses(query_text, top_k=2)
            
            if "data" in result and result["data"]:
                matches = result["data"]
                logger.info(f"[{operation}] FOUND - {len(matches)} relevant past analyses")
                
                # Detailed logging of each matching analysis
                total_similarity = 0
                for i, match in enumerate(matches, 1):
                    metadata = match.get("metadata", {})
                    analysis = match.get("analysis", {})
                    similarity = match.get("similarity", 0)
                    total_similarity += similarity
                    
                    logger.info(f"[{operation}] MATCH #{i} DETAILS:")
                    logger.info(f"   └─ Image Hash: {metadata.get('image_hash', 'unknown')[:8]}")
                    logger.info(f"   └─ Original Timestamp: {metadata.get('timestamp', 'unknown')}")
                    logger.info(f"   └─ Similarity Score: {similarity:.4f}")
                    logger.info(f"   └─ Original Context: {analysis.get('context_guess', 'unknown')}")
                    logger.info(f"   └─ Original Confidence: {analysis.get('confidence', 'unknown')}")
                
                avg_similarity = total_similarity / len(matches)
                logger.info(f"[{operation}] SIMILARITY_STATS:")
                logger.info(f"   └─ Average Similarity: {avg_similarity:.4f}")
                
                duration = time.time() - start_time
                logger.info(f"[{operation}] SUCCESS - Retrieved relevant past analyses")
                logger.info(f"[{operation}] END - Duration: {duration:.3f}s")
                return {
                    "content": [TextContent(type="text", text=json.dumps(result["data"], indent=2))],
                    "found": True,
                    "similarity_score": avg_similarity
                }
            else:
                duration = time.time() - start_time
                logger.info(f"[{operation}] NO_MATCHES - No similar past analyses found")
                logger.info(f"[{operation}] END - Duration: {duration:.3f}s")
                return {
                    "content": [TextContent(type="text", text="No relevant past analyses found")],
                    "found": False,
                    "similarity_score": 0.0
                }
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{operation}] ERROR - Failed to retrieve past analyses: {str(e)}")
            logger.error(f"[{operation}] STACKTRACE:", exc_info=True)
            logger.info(f"[{operation}] END - Duration: {duration:.3f}s")
            return {
                "content": [TextContent(type="text", text=f"Error retrieving past analyses: {str(e)}")],
                "found": False,
                "similarity_score": 0.0
            }

    async def store_inferred_context(self, image_hash: str, inference_data: Dict[str, Any], 
                                   similarity_info: Dict[str, Any]) -> Dict[str, Any]:
        """Store inferred context with details about how it was derived."""
        operation = "STORE_INFERENCE"
        start_time = time.time()
        logger.info(f"[{operation}] START - Storing inferred context for image {image_hash[:8]}")
        
        try:
            # Log inference data
            logger.info(f"[{operation}] INFERENCE_DATA:")
            logger.info(f"   └─ Context: {inference_data.get('context_guess', 'unknown')}")
            logger.info(f"   └─ Confidence: {inference_data.get('confidence', 'unknown')}")
            
            # Log memory influence
            logger.info(f"[{operation}] MEMORY_INFLUENCE:")
            logger.info(f"   └─ Used Past Memories: {similarity_info.get('found', False)}")
            logger.info(f"   └─ Average Similarity: {similarity_info.get('similarity_score', 0.0):.4f}")
            
            # Prepare storage data
            timestamp = datetime.now().isoformat()
            storage_data = {
                **inference_data,
                "timestamp": timestamp,
                "memory_enhanced": similarity_info.get('found', False),
                "memory_similarity": similarity_info.get('similarity_score', 0.0),
            }
            
            # Generate embedding for the inference
            logger.info(f"[{operation}] Generating embedding for inference...")
            inference_text = json.dumps(storage_data)
            embedding = self._generate_embedding(inference_text)
            
            if embedding:
                # Store in ChromaDB
                logger.info(f"[{operation}] Storing in vector database...")
                self.collections["inferred_contexts"].add(
                    ids=[f"{image_hash}_inference"],
                    embeddings=[embedding],
                    documents=[inference_text],
                    metadatas=[{
                        "image_hash": image_hash,
                        "timestamp": timestamp,
                        "confidence": inference_data.get("confidence", 0.0),
                        "memory_enhanced": similarity_info.get('found', False)
                    }]
                )
                
                logger.info(f"[{operation}] SUCCESS - Stored inference with following details:")
                logger.info(f"   └─ Image Hash: {image_hash[:8]}")
                logger.info(f"   └─ Timestamp: {timestamp}")
                logger.info(f"   └─ Memory Enhanced: {similarity_info.get('found', False)}")
                logger.info(f"   └─ Context Length: {len(inference_data.get('context_guess', ''))}")
                
                duration = time.time() - start_time
                logger.info(f"[{operation}] END - Duration: {duration:.3f}s")
                return {
                    "content": [TextContent(type="text", text="Successfully stored inference")],
                    "success": True
                }
            else:
                logger.error(f"[{operation}] FAILED - Could not generate embedding for inference")
                return {
                    "content": [TextContent(type="text", text="Failed to store inference - embedding generation failed")],
                    "success": False
                }
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{operation}] ERROR - Failed to store inference: {str(e)}")
            logger.error(f"[{operation}] STACKTRACE:", exc_info=True)
            logger.info(f"[{operation}] END - Duration: {duration:.3f}s")
            return {
                "content": [TextContent(type="text", text=f"Error storing inference: {str(e)}")],
                "success": False
            }

    def log_memory_statistics(self):
        """Log statistics about both short-term and long-term memory."""
        try:
            # Short-term memory stats
            session_count = len(self.short_term)
            total_components = sum(len(session) for session in self.short_term.values())
            
            # Long-term memory stats
            file_count = len(self._index)
            
            # ChromaDB stats
            complete_count = len(self.collections["complete_analysis"].get()["ids"])
            visual_count = len(self.collections["visual_elements"].get()["ids"])
            style_count = len(self.collections["style_analysis"].get()["ids"])
            scenario_count = len(self.collections["scenario_analysis"].get()["ids"])
            
            logger.info(f"📊 MEMORY STATISTICS:")
            logger.info(f"   └─ SHORT-TERM: {session_count} active sessions with {total_components} total components")
            logger.info(f"   └─ LONG-TERM File: {file_count} stored analyses")
            logger.info(f"   └─ LONG-TERM Vector: {complete_count} complete analyses")
            logger.info(f"   └─ LONG-TERM Vector Components: {visual_count} visual, {style_count} style, {scenario_count} scenario")
            
            return {
                "short_term": {
                    "sessions": session_count,
                    "components": total_components
                },
                "long_term": {
                    "file_storage": file_count,
                    "vector_storage": {
                        "complete": complete_count,
                        "visual": visual_count,
                        "style": style_count,
                        "scenario": scenario_count
                    }
                }
            }
        except Exception as e:
            logger.error(f"❌ Error getting memory statistics: {str(e)}", exc_info=True)
            return {
                "error": str(e)
            }