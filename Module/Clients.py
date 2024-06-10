"""
Contains the clients used as the backbones.
"""

import torch
from peft import PeftModel, PeftConfig

from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM


class GPTClient():
    def __init__(self, model, logger, args):
        # GPT Client
        self.client = OpenAI(api_key=args.api_key)
        self.args = args
        self.logger = logger
        self.model = model
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.label = "GPT"

        self.tokens_dict = {"total_completion_tokens": 0, "total_prompt_tokens": 0, "total_total_tokens": 0}

    def employ(self, SystemPrompt, UserPrompt, name="default"):
        """
        Employ the LLM to response the prompt.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[      
                {"role": "system", "content": SystemPrompt},
                {"role": "user", "content": UserPrompt},
            ]
        )
        # self.logger.info(response.choices[0].message.content)

        self.logger.info(f"{name} (use {self.label}):")
        self.logger.info(f"completion_tokens: {response.usage.completion_tokens}. prompt_tokens: {response.usage.prompt_tokens}. total_tokens: {response.usage.total_tokens}.\n")

        self.completion_tokens += response.usage.completion_tokens
        self.prompt_tokens += response.usage.prompt_tokens
        self.total_tokens += response.usage.total_tokens

        self.tokens_dict["total_completion_tokens"] = self.completion_tokens
        self.tokens_dict["total_prompt_tokens"] = self.prompt_tokens        
        self.tokens_dict["total_total_tokens"] = self.total_tokens

        if name not in self.tokens_dict:
            self.tokens_dict[name] = {"completion_tokens": response.usage.completion_tokens, "prompt_tokens": response.usage.prompt_tokens, "total_tokens": response.usage.total_tokens}

        else:
            self.tokens_dict[name]["completion_tokens"] += response.usage.completion_tokens
            self.tokens_dict[name]["prompt_tokens"] += response.usage.prompt_tokens
            self.tokens_dict[name]["total_tokens"] += response.usage.total_tokens            

        return response.choices[0].message.content
    


class GLMClient():
    def __init__(self, base_model, logger, args, lora_map=None):
        self.args = args
        self.logger = logger
        self.max_input_length = args.max_input_length
        self.max_output_length = args.max_output_length
        self.lora_map = lora_map
        self.device = f"cuda:{args.gpus}"
        self.lora_paths = [path for path in lora_map.values() if path != "default"]

        self.model, self.tokenizer = self.load_base_model(base_model)
        if len(self.lora_paths) > 0:
            self.model = self.load_lora_params(self.lora_paths)

    def load_base_model(self, base_model):
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        return model, tokenizer


    def load_lora_params(self, lora_paths):
        lora_model = self.model
        for lora_path in lora_paths:
            config = PeftConfig.from_pretrained(lora_path)
            lora_model = PeftModel.from_pretrained(
                self.model,
                model_id=lora_path,
                config=config,
                torch_dtype=torch.float16,
                device_map=self.device,
                adapter_name=lora_path
            )

        return lora_model

    def set_lora(self, module_name):
        # change the lora model
        if self.lora_map[module_name] != "default":
            self.model.enable_adapter_layers()
            self.model.set_adapter(self.lora_map[module_name])
            self.logger.info(f"Applied LoRA model from path: {self.lora_map[module_name]}")

        else:
            self.model.disable_adapter_layers()
            self.logger.info(f"Applied original GLM model.")


    def employ(self, SystemPrompt, UserPrompt, name="default"):
        if len(self.lora_paths) > 0:
            self.set_lora(name)
        TotalPrompt = SystemPrompt + "\n" + UserPrompt
        tokenized_prompt = self.tokenizer(TotalPrompt, return_tensors="pt").to(self.device)

        response = self.model.generate(input_ids=tokenized_prompt["input_ids"], max_length=tokenized_prompt["input_ids"].shape[-1] + self.max_output_length)
        response = response[0, tokenized_prompt["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True)
        
        return response
