import os 
import time
import json
import numpy as np

from nltk.util import ngrams
from nlgeval import calc_nlg_metrics

from Module.Clients import GPTClient, GLMClient
from Module.EventMemory import EventMemory
from Module.Personas import Personas
from Module.Generator import Generator

def convert_seconds_to_full_time(seconds):
    """
    Convert time format to 'XX years XX months XX days XX hours XX minutes'.
    """
    units = [("years", 31536000), ("months", 2592000), ("days", 86400), ("hours", 3600), ("minutes", 60)]
    parts = []

    for name, count in units:
        value, seconds = divmod(seconds, count)
        if value:
            parts.append(f"{value} {name}")

    return " ".join(parts)

class MSC():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.sampling_dataset = []

        if args.client == "chatgpt":
            self.client = GPTClient(args.model, logger, args)
            
        elif args.client == "chatglm":
            lora_map = {"EventSummary": args.summary_model, "PersonaExtraction": args.persona_model, "ResponseGenerator": args.generation_model}
            self.client = GLMClient(args.model, logger, args, lora_map=lora_map)
        
        self.usr_name = args.usr_name
        self.agent_name = args.agent_name

        # ==== dataset setting ==== #
        with open(os.path.join(args.data_path, args.data_name), 'r') as f:
            self.dataset = json.load(f)
            if args.test_num > 0:
                self.dataset = self.dataset[:args.test_num]
            self.logger.info(f"Total {len(self.dataset)} samples to be evaluated.")
        
        # ==== memory setting ==== #
        self.relevance_memory_number = args.relevance_memory_number
        self.context_memory_number = args.context_memory_number
        self.dist_thres=args.dist_thres
        self.memory_cache=args.memory_cache

        self.overall_retrieve_score = 0.0
        self.overall_retrieve_count = 0


    def memory_bank_init(self, sample_id, args):
        """
        Initialize First Session
        """
        self.logger.info(f'Memory ID: {sample_id}')
        memory_bank = EventMemory(self.client, sample_id=sample_id, logger=self.logger, args=args, memory_cache=self.memory_cache)
        personas = Personas(self.client, logger=self.logger, args=args)
        response_generator = Generator(self.client, self.sampling_dataset, sample_id, logger=self.logger, args=args)

        for idx, dial in enumerate(self.dataset[sample_id][0]['dialog']):
            current_time = time.time() + memory_bank.current_time_pass

            # memory retrieval
            context_memories = memory_bank.context_retrieve(dial['SPEAKER_1'], n_results=self.context_memory_number, current_time=current_time, datatype='text')
            related_memories = memory_bank.relevance_retrieve(dial['SPEAKER_1'], n_results=self.relevance_memory_number, dist_thres=self.dist_thres, current_time=current_time, datatype='text')

            if len(related_memories) == 0:
                summarized_related_memories = ["No relevant Memories."]
            else:
                summarized_related_memories = []
                for related_memory in related_memories:
                    past_time = convert_seconds_to_full_time(current_time - related_memory['time'])
                    summarized_related_memories.append(f"{past_time} ago, " + f"{related_memory['summary']}.")

            if len(context_memories) == 0:
                agent_context = []
                agent_context.append(f"In this turn, {self.usr_name} said: {dial['SPEAKER_1']}.")

            else:
                agent_context = [f"[TURN {context_memory['idx']}]" + f" : {context_memory['dialog']}." for context_memory in context_memories]
                agent_context.append(f"In this turn, {self.usr_name} said: {dial['SPEAKER_1']}.")

            # store agent response in short-term memory bank
            response_data = {"idx": len(memory_bank.short_term_memory), "time": current_time, "dialog": f"SPEAKER_2: {dial['SPEAKER_2']}"}
            memory_bank.short_term_memory.append(response_data)
    
        return memory_bank, personas, response_generator


    def compute_scores(self, response, calculate_dist_n, dial):
        """
        Compute Metric Scores
        """
        metrics_dict = calc_nlg_metrics([response], [dial['SPEAKER_2']], "response")

        bleu_score = np.array([metrics_dict['Bleu_1'], metrics_dict['Bleu_2'], metrics_dict['Bleu_3'], metrics_dict['Bleu_4']])
        rl_score = np.array([metrics_dict['ROUGE_L']])
        dist_score = np.array([calculate_dist_n(response, i) for i in [1, 2, 3]])

        score = np.concatenate((bleu_score, rl_score, dist_score))

        return score
    
    def interative_eval(self, sample_id, session_num, memory_bank, personas, response_generator, args):
        """
        Implement Followed Sessions
        """

        sum_score = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        for idx, dial in enumerate(self.dataset[sample_id][session_num]['dialog']):
            current_time = time.time() + memory_bank.current_time_pass
            utter_sum_score = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # memory retrieval
            context_memories = memory_bank.context_retrieve(dial['SPEAKER_1'], n_results=self.context_memory_number, current_time=current_time, datatype='text')
            related_memories = memory_bank.relevance_retrieve(dial['SPEAKER_1'], n_results=self.relevance_memory_number, dist_thres=self.dist_thres, current_time=current_time, datatype='text')

            if len(related_memories) == 0:
                summarized_related_memories = ["No relevant Memories."]
            else:
                summarized_related_memories = []
                for related_memory in related_memories:
                    past_time = convert_seconds_to_full_time(current_time - related_memory['time'])
                    summarized_related_memories.append(f"{past_time} ago, " + f"{related_memory['summary']}.")

            if len(context_memories) == 0:
                context = []
                context.append(f"In this turn, {self.usr_name} said: {dial['SPEAKER_1']}.")

            else:
                context = [f"[TURN {context_memory['idx']}]" + f" : {context_memory['dialog']}." for context_memory in context_memories]
                context.append(f"In this turn, {self.usr_name} said: {dial['SPEAKER_1']}.")

            merged_relevant_memory = "\n".join(summarized_related_memories)
            merged_context = "\n".join(context)

            # store agent response in short-term memory bank
            response_data = {"idx": len(memory_bank.short_term_memory), "time": current_time, "dialog": f"SPEAKER_2: {dial['SPEAKER_2']}"}
            memory_bank.short_term_memory.append(response_data)

            # get curent personas
            current_user_traits, current_agent_traits = personas.traits_update(dial['SPEAKER_1'], dial['SPEAKER_2'])

            # response generation
            for i in range(args.build_times):
                self.logger.info(f"The building turn: {i}.")
                if args.sampling:
                    response = response_generator.sampling(dial['SPEAKER_1'], dial['SPEAKER_2'], merged_context, merged_relevant_memory, current_user_traits, current_agent_traits)
                else:
                    response = response_generator.response_build(dial['SPEAKER_1'], merged_context, merged_relevant_memory, current_user_traits, current_agent_traits)

                if args.generation_out:
                    self.logger.info(f"inquiry: {dial['SPEAKER_1']}")
                    self.logger.info(f"response: {response}")
                    self.logger.info(f"reference: {dial['SPEAKER_2']}")

                utter_score = self.compute_scores(response, self.calculate_dist_n, dial)
                utter_sum_score += utter_score
                
            utter_avg_score = utter_sum_score / args.build_times

            sum_score += utter_avg_score

        avg_score = sum_score / (idx + 1)

        return avg_score, memory_bank, personas, response_generator


    def calculate_dist_n(self, text, n):
        words = text.split()
        n_grams = list(ngrams(words, n))
        unique_n_grams = len(set(n_grams))
        dist_n = unique_n_grams / len(n_grams) if n_grams else 0

        return dist_n

    def evaluation(self):
        all_samples_score = []
        for idx, sample in enumerate(self.dataset):
            all_sessions_score = []
            # initialize the first session
            memory_bank, personas, response_generator = self.memory_bank_init(idx, self.args)
            memory_bank.current_time_pass += self.dataset[idx][0]['time_pass']
            for session_num, i in enumerate(range(self.args.min_session, self.args.max_session)):
                score, memory_bank, personas, response_generator = self.interative_eval(idx, i, memory_bank, personas, response_generator, self.args)

                # update time pass
                memory_bank.current_time_pass += self.dataset[idx][i]['time_pass']

                all_sessions_score.append(score)
            avg_score = sum(all_sessions_score) / len(all_sessions_score)
            self.logger.info(f"Average Sample {idx} Result:")
            self.logger.info(f"B-1: {avg_score[0]};  B-2: {avg_score[1]};  B-3: {avg_score[2]};  B-4: {avg_score[3]}; R-L: {avg_score[4]}; Dist-1: {avg_score[5]}; Dist-2: {avg_score[6]}; Dist-3: {avg_score[7]}")

            del memory_bank, personas, response_generator
            self.logger.info(f"Sample {idx} Completed!")
            all_samples_score.append(all_sessions_score)

            if (idx + 1) % self.args.log_step == 0:
                self.logger.info(f"The Average Results of {idx + 1} Samples:")
                for session_number in range(self.args.min_session + 1, self.args.max_session + 1):
                    try:
                        score_session = [sample[session_number - 2] for sample in all_samples_score]
                        avg_score = sum(score_session) / len(score_session)

                        self.logger.info(f"Session {session_number} Results:")
                        self.logger.info(f"B-1: {avg_score[0]};  B-2: {avg_score[1]};  B-3: {avg_score[2]};  B-4: {avg_score[3]}; R-L: {avg_score[4]}; Dist-1: {avg_score[5]}; Dist-2: {avg_score[6]}; Dist-3: {avg_score[7]}")
                        
                    except:
                        self.logger.info(f"session_number {session_number} is out of list index.")
        
        if self.args.sampling:
            self.sampling_file = os.path.join(self.args.sampling_path, self.args.sampling_file_name)
            with open(self.sampling_file, "w") as f:
                    json.dump(self.sampling_dataset, f, indent=4)

        self.logger.info(f"The Average Results of {idx + 1} Samples:")
        for session_number in range(self.args.min_session + 1, self.args.max_session + 1):
            try:
                score_session = [sample[session_number - 2] for sample in all_samples_score]
                avg_score = sum(score_session) / len(score_session)
                self.logger.info(f"Session {session_number} Results:")
                self.logger.info(f"B-1: {avg_score[0]};  B-2: {avg_score[1]};  B-3: {avg_score[2]};  B-4: {avg_score[3]}; R-L: {avg_score[4]}; Dist-1: {avg_score[5]}; Dist-2: {avg_score[6]}; Dist-3: {avg_score[7]}")
                
            except:
                self.logger.info(f"session_number {session_number} is out of list index.")
        