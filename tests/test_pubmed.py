#!/usr/bin/env python3
"""Test script to query PubMed without proxy settings"""

import requests
import os
import time
import xml.etree.ElementTree as ET
import json
from typing import Dict, Any, List

def main():
    """
    Make a simple PubMed query without any proxy settings.
    """
    # Explicitly unset proxy variables in the environment
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    
    print("Making PubMed query without proxy settings...")
    
    # Use a simple test query
    query = "Parkinson's disease genes"
    
    # Create parameters for the search
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": 3,
        "sort": "relevance",
        "retmode": "json"
    }
    
    try:
        # Make request with explicitly empty proxies
        print(f"Requesting search for: {query}")
        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=params,
            timeout=15,
            proxies={}  # Explicitly empty proxies dict
        )
        
        # Check response
        response.raise_for_status()
        data = response.json()
        
        # Get PMIDs
        if "esearchresult" in data and "idlist" in data["esearchresult"]:
            id_list = data["esearchresult"]["idlist"]
            print(f"Found {len(id_list)} PMIDs")
            
            if id_list:
                # Fetch the details for these PMIDs
                fetch_params = {
                    "db": "pubmed",
                    "id": ",".join(id_list),
                    "retmode": "xml",
                    "rettype": "abstract"
                }
                
                print("Fetching article details...")
                fetch_response = requests.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                    params=fetch_params,
                    timeout=15,
                    proxies={}  # Explicitly empty proxies dict
                )
                
                fetch_response.raise_for_status()
                
                # Parse XML
                root = ET.fromstring(fetch_response.text)
                articles = root.findall(".//PubmedArticle")
                
                print(f"Retrieved {len(articles)} articles")
                
                # Display article details
                for i, article in enumerate(articles, 1):
                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else "No Title"
                    
                    journal_elem = article.find(".//Journal/Title")
                    journal = journal_elem.text if journal_elem is not None else "No Journal"
                    
                    year_elem = article.find(".//PubDate/Year")
                    year = year_elem.text if year_elem is not None else "Unknown Year"
                    
                    print(f"\nArticle {i}:")
                    print(f"Title: {title}")
                    print(f"Journal: {journal} ({year})")
            else:
                print("No PMIDs found in the response")
        else:
            print("Invalid response format")
            print(f"Response data: {data}")
    
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
