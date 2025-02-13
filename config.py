import argparse

def get_args(description='Config of LD-Agent'):
    parser = argparse.ArgumentParser(description=description)

    # data
    parser.add_argument("--dataset", choices=["human", "msc", "cc", "quickeval"], default="msc", help="Dataset which will be acted.")
    parser.add_argument('--data_path', type=str, default='dataset', help='Dataset path.')
    parser.add_argument('--data_name', type=str, default='sequential.json', help='Dataset path.')
    parser.add_argument('--id_set', type=str, default='id_set.json', help='Split of sessions and samples.')

    # memory setting
    parser.add_argument("--memory_cache", nargs='?', default=None, help="Whether to use stored memories.")
    parser.add_argument('--relevance_memory_number', type=int, default=1, help='The number of retrieved relevance memory.')
    parser.add_argument('--context_memory_number', type=int, default=30, help='The max number of retrieved context memory.')
    parser.add_argument('--dist_thres', type=float, default=1.5, help='The distance threshold of relevance retrieval.')
    parser.add_argument('--ori_mem_query', type=bool, default=False, help='Whether to use original query.')

    # persona setting
    parser.add_argument('--max_agent_personas', type=int, default=0, help='Select the latest N user personas.')
    parser.add_argument('--max_user_personas', type=int, default=0, help='Select the latest N user personas.')

    # client setting
    parser.add_argument('--client', choices=["chatgpt", "chatglm"], default="chatglm", help="Client to use.")    
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-1106', help='The model name: gpt-3.5-turbo-1106 for ChatGPT, THUDM/chatglm3-6b for ChatGLM.')
    parser.add_argument('--summary_model', type=str, default='default', help='The model for event summary.')   
    parser.add_argument('--persona_model', type=str, default='default', help='The model for persona extraction.')  
    parser.add_argument('--generation_model', type=str, default='default', help='The model for response generation.')       

    # ChatGPT setting
    parser.add_argument('--api_key', type=str, default='Your API Key', help='OpenAI API Key.')

    # ChatGLM setting
    parser.add_argument('--max_input_length', type=int, default=2048, help='The number of maximum size of LLM input.')
    parser.add_argument('--max_output_length', type=int, default=64, help='The number of maximum size of LLM output.')

    # sampling setting
    parser.add_argument('--sampling', action="store_true", default=False, help='Whether to sample data.')
    parser.add_argument('--sampling_step', type=int, default=10, help='The number of sampling frequency.')
    parser.add_argument('--sampling_path', type=str, default='logs/sampling', help='The directory path of sampling')
    parser.add_argument('--sampling_file_name', type=str, default='train.json', help='The name of sampling')

    # evaluation setting
    parser.add_argument('--usr_name', type=str, default='User', help='The name of user.')
    parser.add_argument('--agent_name', type=str, default='Agent', help='The name of agent')

    # common setting
    parser.add_argument('--log_step', type=int, default=10, help='The number of report frequency.')
    parser.add_argument('--generation_out', type=bool, default=False, help='Whether to output generated messages.')
    parser.add_argument('--test_num', type=int, default=0, help='The number of test samples, 0 indicates testing all samples.')
    parser.add_argument('--gpus', type=str, default="0", help='The gpus will be used.')
    parser.add_argument('--build_times', type=int, default=1, help='The times of building response.')
    parser.add_argument('--min_session', type=int, default=1, help='Indicating the session range.')
    parser.add_argument('--max_session', type=int, default=5, help='Indicating the session range.')

    args = parser.parse_args()
    return args
