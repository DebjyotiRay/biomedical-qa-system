#!/usr/bin/env python3
"""
Script to query PubMed using the fetch_pubmed_citations function with LLM-based query optimization.
"""
import sys
import json
from citations import fetch_pubmed_citations
from utils import load_config, load_environment_variables

def main():
    load_environment_variables()
    CONFIG = load_config("config.json")
    
    if len(sys.argv) < 2:
        print("Usage: python biomed_qa_system_final/query_pubmed.py 'your biomedical question here'")
        print("Optional flags:")
        print("  --max=N         Maximum number of citations")
        print("  --reviews       Prioritize review articles")
        print("  --years=N       Maximum age of articles in years (0 for no limit)")
        print("  --json          Output in JSON format")
        print("  --no-llm        Disable LLM for entity extraction and query optimization")
        sys.exit(1)
    
    # Get default values from config
    pubmed_config = CONFIG.get("pubmed", {})
    default_max_citations = pubmed_config.get("max_citations", 5)
    default_prioritize_reviews = pubmed_config.get("prioritize_reviews", False)
    default_max_age_years = pubmed_config.get("max_age_years", 5)
    
    # Parse command line arguments
    query = sys.argv[1]
    max_citations = default_max_citations
    prioritize_reviews = default_prioritize_reviews
    max_age_years = default_max_age_years
    output_json = False
    use_llm = True  # Default to True
    
    for arg in sys.argv[2:]:
        if arg.startswith("--max="):
            max_citations = int(arg.split("=")[1])
        elif arg == "--reviews":
            prioritize_reviews = True
        elif arg.startswith("--years="):
            max_age_years = int(arg.split("=")[1])
        elif arg == "--json":
            output_json = True
        elif arg == "--no-llm":
            use_llm = False
    
    print(f"Searching PubMed: '{query}'")
    print(f"Options: max={max_citations}, reviews={prioritize_reviews}, years={max_age_years}, llm={use_llm}")
    
    # Configure LLM settings
    if "citation_extraction" not in CONFIG:
        CONFIG["citation_extraction"] = {}
        
    # Save original settings
    original_llm_extraction = CONFIG["citation_extraction"].get("use_llm_extraction", False)
    original_llm_query_gen = CONFIG["citation_extraction"].get("use_llm_query_generation", False)
    
    # Set LLM features based on command line flag
    CONFIG["citation_extraction"]["use_llm_extraction"] = use_llm
    CONFIG["citation_extraction"]["use_llm_query_generation"] = use_llm
    
    try:
        citations = fetch_pubmed_citations(
            query,
            max_citations=max_citations,
            prioritize_reviews=prioritize_reviews,
            max_age_years=max_age_years
        )
    finally:
        # Restore original config
        CONFIG = load_config("config.json")
        CONFIG["citation_extraction"]["use_llm_extraction"] = original_llm_extraction
        CONFIG["citation_extraction"]["use_llm_query_generation"] = original_llm_query_gen
    
    # Display results
    if output_json:
        result = {
            "query": query,
            "citations": [citation.to_dict() for citation in citations]
        }
        print(json.dumps(result, indent=2))
    else:
        if citations:
            print(f"\nFound {len(citations)} citations:")
            print("=" * 80)
            for i, citation in enumerate(citations, 1):
                review_marker = " [REVIEW]" if citation.is_review else ""
                print(f"{i}. {citation.title}{review_marker}")
                if citation.authors:
                    print(f"   Authors: {citation.authors}")
                if citation.journal:
                    year_str = f" ({citation.year})" if citation.year else ""
                    print(f"   Journal: {citation.journal}{year_str}")
                if citation.pmid:
                    print(f"   PMID: {citation.pmid}")
                    print(f"   Link: https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}/")
                if citation.doi:
                    print(f"   DOI: {citation.doi}")
                print()
        else:
            print("\nNo citations found matching the criteria.")

if __name__ == "__main__":
    main()
