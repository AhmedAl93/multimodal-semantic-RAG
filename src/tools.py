from llama_parse import LlamaParse
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode #Document, ImageNode
from llama_index.core import get_response_synthesizer, VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
import google.generativeai as genai
from typing import List
import camelot, time, glob, io, fitz, os
from pathlib import Path
from PIL import Image
from dotenv import find_dotenv, load_dotenv
import logging

# logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())

class DocProcessor:
    """
    A class for processing documents, including semantic parsing, table extraction, 
    and image extraction from supported file formats (e.g., PDFs).
    """

    def __init__(self) -> None:
        """
        Initializes the DocProcessor with required models, parsers, and configurations.
        """
        # Parser to process documents into a specific format (e.g., markdown)
        self.parser = LlamaParse(result_type="markdown", verbose=False)

        # Model names for embedding and language generation
        self.embed_model_name = "models/text-embedding-004"
        self.llm_name = "models/gemini-1.5-flash"

        # Initialize the embedding model
        self.embed_model = GeminiEmbedding(model_name=self.embed_model_name)

        # Node parser for splitting documents into semantic chunks based on embedding
        self.semantic_splitter = SemanticSplitterNodeParser(buffer_size=1, embed_model=self.embed_model)

        # Language model for generating summaries or performing other LLM tasks
        self.llm = Gemini(
            model="models/gemini-1.5-flash",
            generate_kwargs={"timeout": 2000}
        )

        # List of supported file extensions for processing
        self.supported_extensions = [".pdf"]

    def list_supported_files(self, inputPath):
        """
        Lists all supported files in the given input path.
        
        Args:
            inputPath (str): The path where files are located.

        Returns:
            List[str]: A list of file paths with supported extensions.
        """
        # Retrieve all files matching the input path and filter by supported extensions
        file_list = glob.glob(inputPath)
        return [f for f in file_list if Path(f).suffix in self.supported_extensions]

    def get_semantic_nodes(self, files_to_process: List[str]):
        """
        Extracts semantic nodes from the provided files.

        Args:
            files_to_process (List[str]): List of file paths to process.

        Returns:
            List: Extracted semantic nodes from documents.
        """
        # Load document content using the parser
        documents = self.parser.load_data(file_path=files_to_process)

        # Split documents into semantic nodes
        semantic_nodes = self.semantic_splitter.get_nodes_from_documents(documents)

        return semantic_nodes

    def extract_tables_from_files(self, files_to_process: List)-> List[tuple]:
        """
        Extracts tables from the provided files.

        Args:
            files_to_process (List[str]): List of file paths to extract tables from.

        Returns:
            List: List of DataFrames representing extracted tables, each with corresponding source file.
        """
        table_dfs_and_files = []
        for path in files_to_process:
        # for path in glob.glob(inputPath):
            table_list = camelot.read_pdf(path, 
                                        pages="all", 
                                        suppress_stdout= True)

            for table in table_list:
                table_dfs_and_files.append(
                    (table.df, f"{Path(path).stem}")
                    )

        return table_dfs_and_files

    def get_table_summaries(self, table_dfs_and_files):
        """
        Generates summaries for a list of tables using a language model.

        Args:
            table_dfs (List): List of DataFrames representing tables.

        Returns:
            List[str]: Summaries for each table.
        """
        table_summaries= []
        for i, (df, file) in enumerate(table_dfs_and_files):
            #Request limit per model per minute= 15
            if len(table_dfs_and_files)>15 and (i%15)==0 and i>0:
                # print("RPM limit reached, 1 minute pause before resuming ...")
                time.sleep(60)

            df_html= df.to_html()
            df_summary = self.llm.complete(
                f"Write a summary of the following HTML table, extract its values and link them: {df_html}"
                )
            table_summaries.append((df_summary.text, file))
            #To avoid ResourceExhausted error, reference: https://groups.google.com/g/adwords-api/c/gcKvh1C3GX4/m/uHU6akJAAwAJ
            # print("1 seconds pause before resuming ...")
            time.sleep(1)

        return table_summaries

    def get_table_nodes(self, files_to_process: List[str]):
        """
        Extracts table nodes and maps query engines to unique table IDs.

        Args:
            files_to_process (List[str]): List of file paths to process.

        Returns:
            Tuple: A tuple containing nodes and query engine mappings.
        """
        table_dfs_and_files= self.extract_tables_from_files(files_to_process)
        table_summaries= self.get_table_summaries(table_dfs_and_files)
        df_nodes=[]
        for summary, file in table_summaries:
            node = TextNode(
                text=summary,
                metadata={
                    "source_file": file,}, 
                    )
            df_nodes.append(node)

        return df_nodes

    def extract_images_from_files(self, files_to_process, save_dir=""):
        """
        Extracts images from the provided files and saves them to the specified directory.

        Args:
            files_to_process (List[str]): List of file paths to extract images from.
            save_dir (str): Directory to save the extracted images.

        Returns:
            None
        """
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        for filepath in files_to_process:
            # Open the document using PyMuPDF
            doc = fitz.open(filepath)

            for page_number in range(len(doc)):
                page = doc[page_number]

                # Iterate through images on the page
                for image_index, img in enumerate(page.get_images(), start=1):
                    xref = img[0]  # Image reference ID

                    # Extract image bytes
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Create a PIL Image object from the bytes
                    pil_image = Image.open(io.BytesIO(image_bytes))

                    # Save the image with a structured name
                    image_name = f"{Path(save_dir).joinpath(Path(filepath).stem)}-p{page_number}-{image_index}.png"
                    pil_image.save(image_name)


    def get_image_summaries(self, image_paths: List[str])->List[str]:
        """
        Generates summaries for a list of images using a generative AI model.

        Args:
            image_paths (List[str]): A list of file paths to images.

        Returns:
            List[str]: A list of textual summaries for each image.
        """
        image_summaries = []
        for i, img_path in enumerate(image_paths):
            # Handle request per minute (RPM) limits
            if len(image_paths) > 15 and (i % 15) == 0 and i > 0:
                # print("RPM limit reached, 1 minute pause before resuming ...")
                time.sleep(60)

            # Prompt to guide the AI for image summarization
            image_prompt = (
                "You are a detail-oriented and meticulous assistant that can extract "
                "meaningful and important information from figures. Please provide a "
                "summary of the information contained in this figure."
            )

            # Initialize the generative model
            llm = genai.GenerativeModel(model_name="gemini-1.5-flash")

            # Attempt to upload the image and generate content with retry logic
            retry = True
            while retry:
                try:
                    # Upload the image to the generative AI service
                    uploaded_img = genai.upload_file(path=img_path)

                    # Generate content using the uploaded image and prompt
                    response = llm.generate_content([uploaded_img, image_prompt])

                    # Exit retry loop upon success
                    retry = False
                except Exception as e:
                    # Handle exceptions by retrying after a short pause
                    # print(f"An exception occurred: {e}. Retrying in 2 seconds ...")
                    time.sleep(2)

            # Append the AI-generated summary to the list
            image_summaries.append(response.text)

            # Pause briefly to avoid exceeding resource or rate limits
            # print("1 second pause before resuming ...")
            time.sleep(1)

        return image_summaries

    def get_image_nodes(self, files_to_process: List[str])->List[TextNode]:
        """
        Extracts images from files and generates corresponding text nodes with metadata.

        Args:
            files_to_process (List[str]): A list of file paths to extract images from.

        Returns:
            List[TextNode]: A list of nodes containing image summaries and metadata.
        """
        image_nodes = []

        # Extract images from the provided files and save them to the specified directory
        self.extract_images_from_files(files_to_process, save_dir="data_images/")

        # Retrieve all extracted image paths
        image_paths = glob.glob("data_images/*.png")

        # Generate summaries for the extracted images
        image_summaries = self.get_image_summaries(image_paths)

        for summary, image_path in zip(image_summaries, image_paths):
            # Extract metadata: page number and source file name from the image file path
            page_num = image_path.split("-p")[-1].split("-")[0]
            source_file = Path(image_path.split("-p")[0]).stem

            # Create a node containing the summary and metadata
            node = TextNode(
                text=summary,
                metadata={
                    "image_path": image_path,       # Path to the image file
                    "source_file": source_file,    # Original source file name
                    "page_number": page_num        # Page number in the source file
                }
            )

            # Append the created node to the list
            image_nodes.append(node)

        return image_nodes
    
class QueryEngine:
    """
    A class to build and manage a query engine capable of retrieving and synthesizing
    responses from multiple data sources, including semantic nodes, image nodes, and table nodes.
    """

    def __init__(self) -> None:
        """
        Initializes the QueryEngine with an instance of DocProcessor for handling
        document processing tasks such as extracting nodes and embedding data.
        """
        self.doc_processor = DocProcessor()
        self.logger = Logger(log_file="../RAG_log.log", 
                             logger_name=__name__).get_logger()

    def build_recursive_retriever(self, 
                                  files_to_process, 
                                  top_k=10):
        """
        Builds a recursive retriever to enable complex querying across multiple data types.

        Args:
            files_to_process (List[str]): A list of file paths to be processed.
            top_k (int, optional): The number of top results to retrieve for similarity-based searches.
                                   Defaults to 10.
        Returns:
            RetrieverQueryEngine: A query engine capable of recursive retrieval and response synthesis.
        """
        # Step 1: Extract semantic nodes from the files (text-based analysis)
        semantic_nodes = self.doc_processor.get_semantic_nodes(files_to_process)
        self.logger.info('Semantic text-based nodes are extracted')

        # Step 2: Extract image nodes, which include summaries and metadata of images
        image_nodes = self.doc_processor.get_image_nodes(files_to_process)
        self.logger.info('Image nodes are extracted')

        # Step 3: Extract table nodes
        df_nodes= self.doc_processor.get_table_nodes(files_to_process)
        self.logger.info('Table nodes are extracted')

        # Step 4: Create a vector-based index for semantic, table, and image nodes
        vector_index = VectorStoreIndex(
            semantic_nodes + df_nodes + image_nodes,  # Combine all node types
            embed_model=self.doc_processor.embed_model  # Use the embedding model for indexing
        )

        # Step 5: Set up a retriever from the vector index for similarity-based querying
        vector_retriever = vector_index.as_retriever(similarity_top_k=top_k)

        # Step 6: Set up a response synthesizer to generate and format query responses
        response_synthesizer = get_response_synthesizer(
            llm= self.doc_processor.llm, 
            response_mode="compact")

        # Step 8: Build a query engine combining the retriever and response synthesizer        
        query_engine = RetrieverQueryEngine(
            retriever=vector_retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7,
                                                 filter_empty=True,
                                                 filter_duplicates=True,
                                                 filter_similar=True)],
            response_synthesizer=response_synthesizer,
            )

        # Return the query engine to allow querying across nodes with synthesized responses
        return query_engine

class Logger:
    def __init__(self, log_file: str, 
                 logger_name: str, 
                 log_level: int = logging.INFO):
        """
        Initializes a logger that writes messages to both a file and the console.

        Args:
            log_file (str): Path to the log file.
            log_level (int): Logging level (e.g., DEBUG, INFO).
        """

        # Set up the logger
        self.logger = logging.getLogger(logger_name)

        fhandler = logging.FileHandler(filename=log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fhandler.setFormatter(formatter)
        self.logger.addHandler(fhandler)
        self.logger.setLevel(log_level)

    def get_logger(self):
        """
        Returns the logger instance for use in other modules.

        Returns:
            logging.Logger: Configured logger instance.
        """
        return self.logger
