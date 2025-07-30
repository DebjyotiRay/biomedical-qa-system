#!/usr/bin/env python3
"""
Streamlit implementation of the Biomedical QA System
"""
import os
import io
import time
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
from src.qa_handler import BiomedicalQAHandler
from src.utils import load_config, load_environment_variables, determine_query_type
from src.citations import extract_entities_from_text

# App title and description
st.set_page_config(
    page_title="Biomedical QA System",
    page_icon="🧬",
    layout="wide"
)

# Load environment variables
load_environment_variables()

# Load configuration
@st.cache_resource
def get_config():
    return load_config("config.json")

config = get_config()

# Initialize QA handler
@st.cache_resource
def get_qa_handler():
    return BiomedicalQAHandler(config)

qa_handler = get_qa_handler()

# Check for visualization libraries
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Title and introduction
st.title("Biomedical QA System")
st.markdown("""
This system answers biomedical questions and provides relevant scientific citations.
Ask questions about genes, proteins, diseases, pathways, or other biomedical topics.
""")

# Sidebar with settings
with st.sidebar:
    st.header("Settings")
    
    include_citations = st.checkbox("Include Citations", value=True)
    
    max_citations = st.slider(
        "Maximum Citations", 
        min_value=1, 
        max_value=15, 
        value=config["pubmed"].get("max_citations", 7)
    )
    
    prioritize_reviews = st.checkbox(
        "Prioritize Review Articles", 
        value=config["pubmed"].get("prioritize_reviews", False)
    )
    
    max_age_years = st.slider(
        "Maximum Publication Age (years)", 
        min_value=0, 
        max_value=20, 
        value=config["pubmed"].get("max_age_years", 5)
    )
    
    use_llm_parent = st.checkbox(
        "Use Primary Model", 
        value=not config["inference"].get("use_llm_parent", False)
    )
    
    fallback_models = config["fallback"].get("models", [])
    if fallback_models:
        st.write("Fallback Models:")
        for model in fallback_models:
            st.write(f"- {model}")
    
    # Update config with UI settings
    if st.button("Update Settings"):
        config["pubmed"]["max_citations"] = max_citations
        config["pubmed"]["prioritize_reviews"] = prioritize_reviews
        config["pubmed"]["max_age_years"] = max_age_years
        config["inference"]["use_llm_parent"] = not use_llm_parent  # Invert because config uses opposite logic
        st.success("Settings updated!")

# Main interface
st.header("Ask a Biomedical Question")

# Question input
question = st.text_area("Enter your question:", height=100)

# Predefined example questions as buttons
st.write("Or try an example question:")
example_questions = [
    "What genes are associated with Parkinson's disease?",
    "What proteins are involved in the JAK-STAT signaling pathway?",
    "What biological pathways are dysregulated in Alzheimer's disease?",
    "What are the key genes involved in breast cancer metastasis?",
    "What are the major proteins in the complement system?"
]

# Create two columns for example buttons
col1, col2 = st.columns(2)
with col1:
    if st.button(example_questions[0]):
        question = example_questions[0]
    if st.button(example_questions[2]):
        question = example_questions[2]
    if st.button(example_questions[4]):
        question = example_questions[4]

with col2:
    if st.button(example_questions[1]):
        question = example_questions[1]
    if st.button(example_questions[3]):
        question = example_questions[3]

# Submit button
if st.button("Submit Question") or question:
    if question:
        with st.spinner("Processing your question..."):
            # Detect query type for UI personalization
            query_type = determine_query_type(question)
            
            # Get answer with citations
            start_time = time.time()
            result = qa_handler.answer_question(
                question, 
                include_citations=include_citations
            )
            processing_time = time.time() - start_time
            
            # Display query type
            st.success(f"Query type: {query_type.capitalize()}")
            
            # Display answer
            st.header("Answer")
            st.write(result["answer"])
            
            # Display metadata
            st.write(f"Source: {result['answer_source']} | Time: {result['processing_time_seconds']:.2f}s")
            
            # Display citations if available
            if include_citations and result.get("citations"):
                st.header("References")
                citations = result["citations"]
                for i, citation in enumerate(citations, 1):
                    review_indicator = " [REVIEW]" if citation.get("is_review", False) else ""
                    st.markdown(f"**{i}. {citation['title']}{review_indicator}**")
                    st.write(f"{citation.get('authors', 'N/A')}")
                    journal_info = citation.get('journal', 'N/A')
                    if citation.get('year'):
                        journal_info += f", {citation['year']}"
                    st.write(f"{journal_info}")
                    
                    # Citation links
                    link_col1, link_col2 = st.columns(2)
                    with link_col1:
                        if citation.get("pmid"):
                            st.markdown(f"[PubMed Link](https://pubmed.ncbi.nlm.nih.gov/{citation['pmid']}/)")
                    with link_col2:
                        if citation.get("doi"):
                            st.write(f"DOI: {citation['doi']}")
                    st.divider()
            
            # Create visualization if possible
            if VISUALIZATION_AVAILABLE:
                # Extract entities for visualization
                entities = []
                for line in result["answer"].split("\n"):
                    line = line.strip()
                    if line.startswith(("•", "-", "*")):
                        entity = line[1:].strip()
                        if entity:
                            entities.append(entity)
                
                if len(entities) >= 2:
                    st.header("Visualization")
                    
                    # Create network visualization
                    st.write("Entity Network Visualization:")
                    
                    # Generate the visualization
                    fig, ax = plt.subplots(figsize=(10, 8))
                    G = nx.Graph()
                    
                    # Create node labels (truncate long names)
                    node_labels = {entity: (entity[:25] + '...' if len(entity) > 28 else entity) for entity in entities}
                    
                    # Add all nodes
                    for entity in entities:
                        G.add_node(entity)
                    
                    # Add edges - simplified approach
                    if len(entities) <= 15:
                        # For smaller graphs, connect all nodes
                        for i, entity1 in enumerate(entities):
                            for entity2 in entities[i+1:]:
                                G.add_edge(entity1, entity2)
                    else:
                        # For larger graphs, limit connections
                        import random
                        num_edges = min(len(entities) * 2, 50)
                        for _ in range(num_edges):
                            node1, node2 = random.sample(entities, 2)
                            G.add_edge(node1, node2)
                    
                    # Draw the graph
                    pos = nx.spring_layout(G, k=0.9/((len(entities)**0.5) if len(entities)>1 else 1), iterations=80, seed=42)
                    nx.draw_networkx_nodes(G, pos, node_size=max(250, 1000 // len(entities)), node_color="skyblue", alpha=0.9)
                    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.3, edge_color="gray")
                    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=max(5, 9 - len(entities)//7), font_family="sans-serif")
                    plt.title(f"Network Visualization ({query_type.capitalize()} Entities)", fontsize=14)
                    plt.axis("off")
                    plt.tight_layout()
                    
                    # Display in Streamlit
                    st.pyplot(fig)
                    
                    # Create download button for visualization
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)
                    buf.seek(0)
                    st.download_button(
                        label="Download Visualization", 
                        data=buf, 
                        file_name=f"{query_type}_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
                    plt.close()
            
            # Export options
            st.header("Export Options")
            export_format = st.selectbox(
                "Export Format", 
                options=config["output"].get("export_formats", ["txt", "csv", "json"])
            )
            
            if st.button("Export Results"):
                from src.utils import export_results, save_export
                export_data = export_results(result["answer"], export_format)
                export_data["output_dir"] = config["output"].get("dir", "output")
                filename = save_export(export_data)
                if filename:
                    st.success(f"Results exported to {filename}")
                    
                    # Check if file exists and offer download
                    if os.path.exists(filename):
                        with open(filename, "r") as file:
                            file_content = file.read()
                        
                        st.download_button(
                            label=f"Download {export_format.upper()} File",
                            data=file_content,
                            file_name=os.path.basename(filename),
                            mime=f"text/{export_format}"
                        )
                else:
                    st.error("Export failed.")
    else:
        st.warning("Please enter a question.")

# Footer with version info
st.markdown("---")
st.markdown("Biomedical QA System - Streamlit Interface")
