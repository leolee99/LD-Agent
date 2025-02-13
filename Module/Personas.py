"""
Contains the user and agent persona modules of LD-Agent.
"""

class Personas():
    def __init__(self, client, logger, args):
        self.args = args
        self.logger = logger
        self.user_traits = []
        self.agent_traits = []
        self.usr_name = args.usr_name
        self.agent_name = args.agent_name
        self.max_agent_personas = args.max_agent_personas
        self.max_user_personas = args.max_user_personas

        self.LLMclient = client

    def traits_update(self, inquiry, response):
        self.user_traits_update(inquiry)
        if len(self.user_traits) <= self.max_user_personas:
            merged_user_traits = "\n".join(self.user_traits)
        else:
            merged_user_traits = "\n".join(self.user_traits[-self.max_user_personas:])

        if len(self.agent_traits) <= self.max_agent_personas:
            merged_agent_traits = "\n".join(self.agent_traits)
        else:
            merged_agent_traits = "\n".join(self.agent_traits[-self.max_agent_personas:])

        self.agent_traits_update(response)

        # self.logger.info(merged_user_traits)
        # self.logger.info(merged_agent_traits)
            
        return merged_user_traits, merged_agent_traits
    
    
    def user_traits_update(self, sentence):
        sys_prompt = f"You excel at extracting user personal traits from their words, a renowned local communication expert."

        cot_example = "If no traits can be extracted in the sentence, you should reply 'NO_TRAIT'. Given you some format examples of traits extraction, such as:\n" \
                    + "1. No, I have no longer serve in the millitary, I had served up the full term that I signed up for, and now work outside of the millitary.\n" \
                    + "Extracted Traits: 'I now work elsewhere. I used to be in the military.'\n" \
                    + "2. That must a been some kind of endeavor. Its great that people are aware of issues that arise in their homes, otherwise it can be very problematic in the future.\n" \
                    + "'NO_TRAIT'\n"

        user_prompt = f"Please extract the personal traits who said this sentence (no more than 20 words):\n{sentence}\n"

        user_prompt = cot_example + user_prompt
        summarized_traits = self.LLMclient.employ(sys_prompt, user_prompt, "PersonaExtraction")
        
        if ('NO_TRAIT' not in summarized_traits) and len(summarized_traits) > 3:
            self.user_traits.append(summarized_traits)

        return self.user_traits
    
    def agent_traits_update(self, sentence):
        sys_prompt = f"You excel at extracting user personal traits from their words, a renowned local communication expert."

        cot_example = "If no traits can be extracted in the sentence, you should reply 'NO_TRAIT'. Given you some format examples of traits extraction, such as:\n" \
                    + "1. No, I have no longer serve in the millitary, I had served up the full term that I signed up for, and now work outside of the millitary.\n" \
                    + "Extracted Traits: 'I now work elsewhere. I used to be in the military.'\n" \
                    + "2. That must a been some kind of endeavor. Its great that people are aware of issues that arise in their homes, otherwise it can be very problematic in the future.\n" \
                    + "'NO_TRAIT'\n"

        user_prompt = f"Please extract the personal traits who said this sentence (no more than 20 words):\n{sentence}\n"


        user_prompt = cot_example + user_prompt
        summarized_traits = self.LLMclient.employ(sys_prompt, user_prompt, "PersonaExtraction")

        if ('NO_TRAIT' not in summarized_traits)  and len(summarized_traits) > 3:
            self.agent_traits.append(summarized_traits)

        return self.agent_traits


        