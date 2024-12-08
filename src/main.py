import tools
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--InputPath', 
    help= 'Directory path containing files to be processed, or a single file path')

def main():
    args = parser.parse_args()
    logger = tools.Logger(log_file="../RAG_log.log", 
                                    logger_name=__name__).get_logger()

    engine= tools.QueryEngine()
    logger.info('Multimodal RAG application started')

    files_to_process= engine.doc_processor.list_supported_files(args.InputPath)
    logger.info(f'Files to be processed are: {files_to_process}')

    logger.info('Multimodal retriever is being built ...')
    query_engine= engine.build_recursive_retriever(files_to_process)

    logger.info('Engine is ready to be queried')
    response = query_engine.query(
    "What are the best grid integration strategy ?"
    )
    print(response.response)

if __name__ == '__main__':
    main()