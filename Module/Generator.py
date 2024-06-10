"""
Contains the response generation modules of LD-Agent
"""

from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction, DefaultEmbeddingFunction

class Generator():
    def __init__(self, client, sample_id, logger, args):
        self.args = args
        self.logger = logger
        self.embedding_function = DefaultEmbeddingFunction()
        self.data_loader = ImageLoader()

        self.LLMclient = client

        self.usr_name = args.usr_name
        self.agent_name = args.agent_name


    def select_prompts(self, inquiry, context, memories, user_traits, agent_traits):
        sys_prompt = f"As a communication expert with outstanding communication habits, you embody the role of {self.agent_name} throughout the following dialogues. Here are some of your distinctive personal traits: {agent_traits}."        

        user_prompt = f"""<CONTEXT>\nDrawing from your recent conversation with {self.usr_name}:\n{context}\n""" \
                        + f"""<MEMORY>\nThe memories linked to the ongoing conversation are:\n{memories}\n""" \
                        + f"""<USER_TRAITS>\nDuring the conversation process between you and {self.usr_name} in the past, you found that the {self.usr_name} has the following characteristics:\n{user_traits}\n""" \
                        + f"""\nNow, please role-play as {self.agent_name} to continue the dialogue between {self.agent_name} and {self.usr_name}.\n""" \
                        + f"""{self.usr_name} just said: {inquiry}\n""" \
                        + f"""Please respond to {self.usr_name}'s statement using the following format (maximum 30 words, must be in English):\nRESPONSE:\n"""

        return sys_prompt, user_prompt

    def response_build(self, inquiry, context, memories, user_traits, agent_traits):
        sys_prompt, user_prompt = self.select_prompts(inquiry, context, memories, user_traits, agent_traits)

        response = self.LLMclient.employ(sys_prompt, user_prompt, "ResponseGenerator")

        return response
