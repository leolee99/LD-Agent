import os 
import time
import json
import numpy as np

from nltk.util import ngrams
from nlgeval import calc_nlg_metrics

from Module.Clients import GPTClient, GLMClient

def compute_scores(response, calculate_dist_n, reference):
    """
    Compute Metric Scores
    """
    metrics_dict = calc_nlg_metrics([response], [reference], "response")

    bleu_score = np.array([metrics_dict['Bleu_1'], metrics_dict['Bleu_2'], metrics_dict['Bleu_3'], metrics_dict['Bleu_4']])
    rl_score = np.array([metrics_dict['ROUGE_L']])
    dist_score = np.array([calculate_dist_n(response, i) for i in [1, 2, 3]])

    score = np.concatenate((bleu_score, rl_score, dist_score))

    return score

def calculate_dist_n(text, n):
    words = text.split()
    n_grams = list(ngrams(words, n))
    unique_n_grams = len(set(n_grams))
    dist_n = unique_n_grams / len(n_grams) if n_grams else 0

    return dist_n

class QuickEval():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.sampling_dataset = []

        if args.client == "chatgpt":
            self.client = GPTClient(args.model, logger, args)
            
        elif args.client == "chatglm":
            lora_map = {"EventSummary": args.summary_model, "PersonaExtraction": args.persona_model, "ResponseGenerator": args.generation_model, "default": "default"}
            self.client = GLMClient(args.model, logger, args, lora_map=lora_map)
        
        self.usr_name = args.usr_name
        self.agent_name = args.agent_name

        # ==== dataset setting ==== #
        with open(os.path.join(args.data_path, args.data_name), 'r') as f:
            self.dataset = json.load(f)
            if args.test_num > 0:
                self.dataset = self.dataset[:args.test_num]
            self.logger.info(f"Total {len(self.dataset)} samples to be evaluated.")

        with open(os.path.join(args.data_path, args.id_set), 'r') as f:
            self.id_dataset = json.load(f)
            if args.test_num > 0:
                self.id_dataset = self.id_dataset[:args.test_num]
            self.logger.info(f"ID list has been loaded.")        
        

    def evaluation(self):
        all_samples_score = []
        ss2_score, ss3_score, ss4_score, ss5_score, = [], [], [], []
        for idx, sample in enumerate(self.dataset):
            sys_prompt = sample["conversations"][0]["content"]
            user_prompt = sample["conversations"][1]["content"]
            reference = sample["conversations"][2]["content"]

            response = self.client.employ(sys_prompt, user_prompt, "ResponseGenerator")
            # response = self.client.employ(sys_prompt, user_prompt, "default")

            utter_score = compute_scores(response, calculate_dist_n, reference)
            all_samples_score.append(utter_score)
            if self.id_dataset[idx]["session_number"] == 2:
                ss2_score.append(utter_score)
            
            elif self.id_dataset[idx]["session_number"] == 3:
                ss3_score.append(utter_score)

            elif self.id_dataset[idx]["session_number"] == 4:
                ss4_score.append(utter_score)

            elif self.id_dataset[idx]["session_number"] == 5:
                ss5_score.append(utter_score)

            if (idx % self.args.log_step == 0) and (idx != 0):
                mean_all_samples_score = sum(all_samples_score) / len(all_samples_score)
                self.logger.info(f"Mean Score of {idx + 1} samples: {mean_all_samples_score}")

                mean_ss2_score = sum(ss2_score) / len(ss2_score)
                self.logger.info(f"Mean Session 2 Score of {idx + 1} samples: {mean_ss2_score}")

                mean_ss3_score = sum(ss3_score) / len(ss3_score)
                self.logger.info(f"Mean Session 3 Score of {idx + 1} samples: {mean_ss3_score}")

                mean_ss4_score = sum(ss4_score) / len(ss4_score)
                self.logger.info(f"Mean Session 4 Score of {idx + 1} samples: {mean_ss4_score}")        

                mean_ss5_score = sum(ss5_score) / len(ss5_score)
                self.logger.info(f"Mean Session 5 Score of {idx + 1} samples: {mean_ss5_score}")

        mean_all_samples_score = sum(all_samples_score) / len(all_samples_score)
        self.logger.info(f"Mean Score of {idx + 1} samples: {mean_all_samples_score}")

        mean_ss2_score = sum(ss2_score) / len(ss2_score)
        self.logger.info(f"Mean Session 2 Score of {idx + 1} samples: {mean_ss2_score}")

        mean_ss3_score = sum(ss3_score) / len(ss3_score)
        self.logger.info(f"Mean Session 3 Score of {idx + 1} samples: {mean_ss3_score}")

        mean_ss4_score = sum(ss4_score) / len(ss4_score)
        self.logger.info(f"Mean Session 4 Score of {idx + 1} samples: {mean_ss4_score}")        

        mean_ss5_score = sum(ss5_score) / len(ss5_score)
        self.logger.info(f"Mean Session 5 Score of {idx + 1} samples: {mean_ss5_score}")