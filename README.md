
#Multimodal Semantic RAG

### Table of contents
* [Overview](###Overview)
* [Features](###Features)
* [Workflow](###Workflow)
* [Demo](###Demo)

### Overview

Retrieval-Augmented Generation (RAG) is an AI framework that combines the retrieval capabilities of information retrieval systems (mainly encoders & Vector DBs) with the generative abilities of large language models (LLMs). 
External knowledge sources (User Documents) are used to provide accurate, context-aware answers and generate factual responses

A RAG application has exponentially higher utility if it can work with a wide variety of data types: tables, graphs, charts, etc. and not just text. 
This requires a framework that can understand and generate responses by coherently interpreting textual, visual and tabular forms of information.
To tackle this problem, this repo aims to create a Multimodal Semantic RAG system.

### Features
- Semantic chunking:
With LlamaIndex's SemanticSplitterNodeParser, split documents into semantically coherent, meaningful chunks.
- Image and table detection:
Detecting images and tables using PyMuPDF and Camelot respectively.
- Summarizing images and tables:
Using a multimodal LLM (eg. gemini-1.5-flash), create a text description of each image and each table (tables transformed to dataframes, then fed to the LLM in HTML format.
- ReRanking: available soon.

### Workflow
<p align="center">
<img src="./assets/Workflow.png" width="90%">
</p>

### Demo
The file data/input/Solar_Energy_Technical_Report.pdf was passed as input file. 
In this demo, each query is targeting a specific modality, meaning that the answer in the document can be found, exclusively, in either a paragraph, an image or a table.
Here is a set of queries/answers:
<p align="center">
<img src="./data/output/Quick_demo.png" width="90%">
</p>