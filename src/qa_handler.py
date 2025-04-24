"""
Main QA handler for the Biomedical QA system
Integrates model inference and citation fetching with multi-API fallback
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import time
import json
import random # For visualization jitter
from datetime import datetime
# Import necessary libraries for fallback
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not found. Fallback to OpenAI models will not work.")

# Import necessary libraries for visualization (Conditional)
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
    logging.info("NetworkX and Matplotlib found. Visualization enabled.")
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("NetworkX or Matplotlib not found. Visualization disabled.")


# Import our modules
from src.model import BiomedicalModel
from src.citations import Citation, extract_entities_from_text, fetch_pubmed_citations
# We're directly using fetch_pubmed_citations now, no need for wrapper
from src.utils import determine_query_type, format_scientific_answer, export_results, save_export, load_config
# Need constants for fallback prompt construction
from src.constants import INFERENCE_SYSTEM_PROMPT, QUERY_TYPE_INSTRUCTIONS, FEW_SHOT_EXAMPLES

# Use centralized logging configuration
from src.logger_config import get_logger
logger = get_logger(__name__)

class BiomedicalQAHandler:
    """Main handler for biomedical question answering with citations and fallback"""

    def __init__(self, config: Dict[str, Any]):
        """ Initialize the QA handler """
        self.config = config
        self.model = BiomedicalModel(config)
        self.fallback_config = config.get("fallback", {})
        self.openai_client = None
        self._setup_fallback_clients()

    def _setup_fallback_clients(self):
        """Initialize OpenAI client for fallback models"""
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.openai_client = openai.OpenAI(
                        api_key=api_key,
                        timeout=60.0,
                        max_retries=3
                    )
                    logger.info("OpenAI client initialized for fallback.")
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI client: {e}")
            else: 
                logger.warning("OPENAI_API_KEY not set. OpenAI fallback disabled.")


    def setup(self) -> bool:
        """ Set up the QA system by loading the primary model """
        logger.info("Setting up primary model...")
        if not self.model.load_model():
            logger.error("Failed to load base model")
            return False
        self.model.load_peft_adapter() # Attempt to load adapter
        return True

    def _call_openai_fallback(self, model_id: str, question: str, query_type: str) -> Optional[str]:
        """Call OpenAI API for fallback"""
        if not self.openai_client: return None
        logger.info(f"Attempting fallback using OpenAI model: {model_id}")
        try:
            # Use the same prompt structure as the primary model
            # OpenAI's chat completion usually works better with system/user roles
            system_prompt = INFERENCE_SYSTEM_PROMPT + "\n\n" + QUERY_TYPE_INSTRUCTIONS.get(query_type, QUERY_TYPE_INSTRUCTIONS["general"])
            user_content = f"Question: {question}\n\n"
            if query_type in ['gene', 'protein', 'pathway', 'disease'] and FEW_SHOT_EXAMPLES:
                 user_content += f"{FEW_SHOT_EXAMPLES.strip()}\n\n"
            user_content += "Answer:"

            response = self.openai_client.chat.completions.create(
                model=model_id,
                messages=[ {"role": "system", "content": system_prompt}, {"role": "user", "content": user_content} ],
                temperature=self.config.get("model", {}).get("temperature", 0.1), # Use same temp
                max_tokens=self.config.get("inference", {}).get("max_new_tokens", 350)
            )
            answer = response.choices[0].message.content.strip()
            logger.info(f"Received fallback answer from {model_id}")
            return format_scientific_answer(answer)
        except Exception as e:
            logger.error(f"Error calling OpenAI model {model_id}: {e}")
            return None



    def answer_question(self, question: str, include_citations: bool = True) -> Dict[str, Any]:
        """ Answer a biomedical question with optional citations and fallback """
        logger.info(f"Processing question: {question}")
        start_time = time.time()
        query_type = determine_query_type(question)
        logger.info(f"Detected query type: {query_type}")

        # Check if we should bypass the main model and go straight to fallback
        use_llm_parent = self.config.get("inference", {}).get("use_llm_parent", True)
        
        if use_llm_parent:
            # 1. Try primary model
            base_answer = self.model.generate_answer(question, query_type)
            answer_source = "primary_model"
            
            # 2. Attempt OpenAI fallback if needed
            if base_answer.startswith("Error:") or not base_answer:
                logger.warning(f"Primary model failed: '{base_answer}'. Attempting fallback...")
                base_answer = "" # Reset base_answer
                fallback_models = self.fallback_config.get("models", [])
                for fb_model_id in fallback_models:
                    if "gpt" in fb_model_id.lower():
                        fb_answer = self._call_openai_fallback(fb_model_id, question, query_type)
                        if fb_answer and not fb_answer.startswith("Error:"):
                            base_answer = fb_answer
                            answer_source = f"fallback_{fb_model_id}"
                            logger.info(f"Using fallback: {fb_model_id}")
                            break # Stop trying fallbacks once one succeeds
                
                if base_answer == "":  # No successful fallback
                    logger.error("Fallback model failed.")
                    base_answer = "Error: Could not generate an answer from primary or fallback models."
                    answer_source = "fallback_failed"
        else:
            # Skip main model and go straight to OpenAI fallback
            logger.info("use_llm_parent is set to false, using direct OpenAI fallback")
            base_answer = ""
            fallback_models = self.fallback_config.get("models", [])
            for fb_model_id in fallback_models:
                if "gpt" in fb_model_id.lower():
                    fb_answer = self._call_openai_fallback(fb_model_id, question, query_type)
                    if fb_answer and not fb_answer.startswith("Error:"):
                        base_answer = fb_answer
                        answer_source = f"direct_fallback_{fb_model_id}" # Indicate this was a direct fallback
                        logger.info(f"Using direct fallback: {fb_model_id}")
                        break # Stop trying fallbacks once one succeeds
            
            if base_answer == "":  # No successful fallback
                logger.error("Direct fallback model failed.")
                base_answer = "Error: Could not generate an answer from fallback models."
                answer_source = "fallback_failed"


        # 3. Highlight keywords (less critical now, applied to final base_answer)
        highlighted_answer = self.model.highlight_keywords(base_answer)

        # 4. Fetch Citations
        citations = []
        citation_dicts = []
        if include_citations and not base_answer.startswith("Error:"):
            try:
                logger.info(f"Fetching citations for question: '{question}'")
                
                # Get parameters and configure LLM
                pubmed_config = self.config.get("pubmed", {})
                max_citations = pubmed_config.get("max_citations", 7)
                prioritize_reviews = pubmed_config.get("prioritize_reviews", False)
                max_age_years = pubmed_config.get("max_age_years", 5)
                
                config = load_config("config.json")
                if "citation_extraction" not in config:
                    config["citation_extraction"] = {}
                
                # Save settings to restore later
                original_llm_extraction = config["citation_extraction"].get("use_llm_extraction", False)
                original_llm_query_gen = config["citation_extraction"].get("use_llm_query_generation", False)
                
                # Always enable LLM features
                config["citation_extraction"]["use_llm_extraction"] = True
                config["citation_extraction"]["use_llm_query_generation"] = True
                
                try:
                    citations = fetch_pubmed_citations(
                        question,
                        max_citations=max_citations,
                        prioritize_reviews=prioritize_reviews,
                        max_age_years=max_age_years
                    )
                    
                    citation_dicts = [citation.to_dict() for citation in citations]
                    logger.info(f"Fetched {len(citation_dicts)} citations")
                finally:
                    # Restore original settings
                    config = load_config("config.json")
                    config["citation_extraction"]["use_llm_extraction"] = original_llm_extraction
                    config["citation_extraction"]["use_llm_query_generation"] = original_llm_query_gen
            except Exception as e:
                logger.error(f"Error fetching citations from independent service: {e}", exc_info=True)
                # Don't fail the whole process, just log the error

        # 5. Format final answer for display
        formatted_answer = self._format_answer_with_citations(highlighted_answer, citation_dicts)

        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Question processing completed in {processing_time:.2f} seconds (Source: {answer_source})")

        # 6. Prepare results dictionary
        result = {
            "question": question,
            "answer": highlighted_answer, # Core answer for export/viz
            "formatted_answer": formatted_answer, # Answer + citations for display
            "query_type": query_type,
            "citations": citation_dicts,
            "answer_source": answer_source,
            "processing_time_seconds": round(processing_time, 2)
        }
        return result

    def _format_answer_with_citations(self, answer: str, citations: List[Dict[str, Any]]) -> str:
        """ Format the answer with appended citations """
        if answer.startswith("Error:"): return answer # Pass through errors
        if not citations: return answer + "\n\nNot able to provide citations for this query."

        # Start with the potentially multi-line answer
        formatted_answer = answer.strip() + "\n\n**References:**\n" # Ensure separation

        for i, citation in enumerate(citations, 1):
            review_indicator = " [REVIEW]" if citation.get("is_review", False) else ""
            formatted_answer += f"{i}. {citation['title']}{review_indicator}  \n"
            # Handle potentially missing authors/journal gracefully
            authors = citation.get('authors', 'N/A')
            journal = citation.get('journal', 'N/A')
            year = citation.get('year', '')
            formatted_answer += f"   {authors}  \n"
            formatted_answer += f"   {journal}"
            if year: formatted_answer += f", {year}"
            formatted_answer += "  \n"
            if citation.get("doi"):
                formatted_answer += f"   DOI: {citation['doi']}  \n"
            # Use direct PubMed link instead of Google search
            if citation.get("pmid"):
                formatted_answer += f"   Link: https://pubmed.ncbi.nlm.nih.gov/{citation['pmid']}/  \n\n"
            else:
                # Fallback to the pre-generated Google link if no PMID
                formatted_answer += f"   [Link]({citation['url']})  \n\n"

        return formatted_answer.strip() # Remove trailing newline

    def _extract_entities_for_viz(self, answer_text: str) -> List[str]:
        """Extracts entities from bulleted list for visualization"""
        entities = []
        if not answer_text or answer_text.startswith("Error:"): return entities
        for line in answer_text.split("\n"):
            line = line.strip()
            if line.startswith(("•", "-", "*")):
                entity = line[1:].strip()
                if entity: entities.append(entity)
        return entities

    def _visualize_entity_network(self, entity_type: str, entities: List[str], output_dir: str) -> Optional[str]:
        """ Creates and saves a visualization of entity relationships """
        if not VISUALIZATION_AVAILABLE or not entities or len(entities) < 2:
            logger.warning("Visualization skipped: Libraries not available or not enough entities.")
            return None

        logger.info(f"Attempting to visualize network for {len(entities)} {entity_type} entities.")
        try:
            G = nx.Graph()
            # Truncate long entity names for better visualization
            node_labels = {entity: (entity[:25] + '...' if len(entity) > 28 else entity) for entity in entities}
            for entity in entities: G.add_node(entity) # Use full name as node ID

            # Simplified: connect all nodes if few, otherwise random subset for viz clarity
            if len(entities) <= 15:
                 for i, entity1 in enumerate(entities):
                     for entity2 in entities[i+1:]:
                         G.add_edge(entity1, entity2)
            else: # Connect randomly for larger lists to avoid clutter
                 num_edges = min(len(entities) * 2, 50) # Limit edges
                 for _ in range(num_edges):
                      # Ensure we pick two *different* nodes
                      node1, node2 = random.sample(entities, 2)
                      G.add_edge(node1, node2)


            plt.figure(figsize=(min(16, len(entities)*1.0), min(12, len(entities)*0.8))) # Adjust size
            # Use a layout that spaces nodes better, increase k for more spread
            pos = nx.spring_layout(G, k=0.9/((len(entities)**0.5) if len(entities)>1 else 1), iterations=80, seed=42) # Increase k, iterations
            nx.draw_networkx_nodes(G, pos, node_size=max(250, 1000 // len(entities)), node_color="skyblue", alpha=0.9)
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.3, edge_color="gray")
            # Adjust label font size dynamically
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=max(5, 9 - len(entities)//7), font_family="sans-serif")
            plt.title(f"Network Visualization ({entity_type.capitalize()} Entities)", fontsize=14)
            plt.axis("off")
            plt.tight_layout() # Adjust layout

            # Save figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{entity_type}_network_{timestamp}.png"
            os.makedirs(output_dir, exist_ok=True) # Ensure output dir exists
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=150) # Improve resolution and fit
            plt.close() # Close plot to free memory
            logger.info(f"Visualization saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error creating visualization: {e}", exc_info=True)
            return None

    def interactive_session(self):
        """ Run an interactive QA session for testing """
        print("\n===== Biomedical QA System (Interactive Mode) =====\n")
        print("Type 'exit' to quit.")
        print("Type 'citations:on' or 'citations:off' to toggle.")
        print("Type 'export:<format>' (csv/json/txt) after an answer to export.")
        print("Type 'viz' after an answer with a list to attempt visualization.\n")

        # Use config for initial state
        include_citations = self.config.get("inference", {}).get("include_citations", True)
        print(f"Citations initially: {'ON' if include_citations else 'OFF'}")
        last_result = None # Store the last result for export/viz

        while True:
            try:
                user_input = input("\nEnter question or command: ")
                input_lower = user_input.lower()

                if input_lower in ['exit', 'quit', 'q']: break
                if not user_input.strip(): continue

                # Handle Commands
                if input_lower == 'citations:on':
                    include_citations = True
                    print("Citations: ON")
                    continue
                elif input_lower == 'citations:off':
                    include_citations = False
                    print("Citations: OFF")
                    continue
                elif input_lower.startswith('export:'):
                    if not last_result or last_result.get("answer", "").startswith("Error:"):
                        print("No valid previous answer to export.")
                        continue
                    try:
                        export_format = input_lower.split(':', 1)[1].strip()
                        allowed_formats = self.config.get("output", {}).get("export_formats", ["txt"])
                        if export_format not in allowed_formats:
                             print(f"Invalid export format: {export_format}. Allowed: {allowed_formats}")
                             continue
                        print(f"\nExporting last result as {export_format}...")
                        export_dir = self.config.get("output", {}).get("dir", "output")
                        # Use the 'answer' field (before citations) for structured export
                        export_data = export_results(last_result["answer"], export_format)
                        export_data["output_dir"] = export_dir
                        filename = save_export(export_data)
                        if filename: print(f"Result exported to {filename}")
                        else: print("Export failed.")
                    except Exception as e: print(f"Error during export: {e}")
                    continue
                elif input_lower == 'viz':
                     if not last_result or last_result.get("answer", "").startswith("Error:"):
                         print("No valid previous answer to visualize.")
                         continue
                     if not VISUALIZATION_AVAILABLE:
                         print("Visualization libraries (networkx, matplotlib) not installed.")
                         continue
                     entities = self._extract_entities_for_viz(last_result["answer"])
                     if len(entities) >= 2:
                         print(f"\nAttempting visualization for {len(entities)} entities...")
                         output_dir = self.config.get("output", {}).get("dir", "output")
                         viz_path = self._visualize_entity_network(last_result["query_type"], entities, output_dir)
                         if viz_path: print(f"Visualization saved to {viz_path}")
                         else: print("Could not create visualization.")
                     else:
                         print("Not enough entities found in the last answer for visualization (need at least 2).")
                     continue

                # Process Question if it wasn't a command
                question = user_input
                print("\nProcessing...")
                result = self.answer_question(question, include_citations)
                last_result = result # Store for potential export/viz

                # Display the answer
                print("\n" + "="*80)
                print(result["formatted_answer"])
                print("="*80)
                print(f"(Source: {result['answer_source']} | Time: {result['processing_time_seconds']}s)")

            except KeyboardInterrupt:
                print("\nSession interrupted. Type 'exit' to quit.")
                break
            except Exception as e:
                logger.error(f"Error during interactive session: {e}", exc_info=True)
                print(f"\nAn unexpected error occurred: {e}")

        print("\nSession ended.")
