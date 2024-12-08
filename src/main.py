import tools
import argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument(
    '--InputPath', 
    help= 'Directory path containing files to be processed, or a single file path')

parser.add_argument(
    '--Query', 
    help= '')

def main():
    args = parser.parse_args()
    logger = tools.Logger(log_file="../RAG_log.log", 
                                    logger_name=__name__).get_logger()
    logger.info("Script execution started.")

    try:
        logger.info("Initializing QueryEngine...")
        engine= tools.QueryEngine()

        logger.info(f"Listing supported files in the directory: {args.InputPath}")
        files_to_process= engine.doc_processor.list_supported_files(args.InputPath)
        if not files_to_process:
            logger.warning("No supported files found in the input directory. Exiting script.")
            sys.exit()

        logger.info(f'Number of files to be processed is: {len(files_to_process)}. Here is the list: {files_to_process}')

        logger.info("Building Multimodal recursive retriever with supported files...")
        query_engine= engine.build_recursive_retriever(files_to_process)

        logger.info("Query engine built successfully.")

        response = query_engine.query(args.Query
        )
        print(response.response)
    
    except Exception as e:
        # Log unexpected errors
        logger.error(f"An error occurred: {e}. Please solve it before retrying again")

if __name__ == '__main__':
    main()