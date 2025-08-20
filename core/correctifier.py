import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from transformers import Gemma3ForConditionalGeneration
import langdetect
from langdetect import detect
from langdetect import DetectorFactory
from pathlib import Path
import datetime

from core.prompts import prompts

DetectorFactory.seed = 0

available_models = {"LLAMA1B": "meta-llama/Llama-3.2-1B-Instruct",
                    "GEMMA2B": "google/gemma-2-2b-it",
                    "SALAMANDRA2B": "BSC-LT/salamandra-2b-instruct",
                    "LLAMA3B": "meta-llama/Llama-3.2-3B-Instruct",
                    "GEMMA4B": "google/gemma-3-4b-it",
                    "OLMO7B": "allenai/OLMo-2-1124-7B-Instruct",
                    "SALAMANDRA7B": "BSC-LT/salamandra-7b-instruct",
                    "LLAMA8B": "meta-llama/Llama-3.1-8B-Instruct",
                    "GEMMA9B": "google/gemma-2-9b-it",
                    "GEMMA12B": "google/gemma-3-12b-it",
                    "GEMMA27B": "google/gemma-3-27b-it"}


# Inspired with https://huggingface.co/google/gemma-3-12b-it
# More info in https://huggingface.co/docs/transformers/main/en/model_doc/gemma3

class Correctifier(object):
    def __init__(self, selected_model, access_token, device, lang="en"):
        self.lang = lang
        pretrained_model = available_models[selected_model]
        self.pretrained_model = pretrained_model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, token=access_token)
        self.processor = AutoProcessor.from_pretrained(pretrained_model, token=access_token)
        # If CUDA device is chosen and more than one available, use all
        device_map = "auto" if device.type == 'cuda' and torch.cuda.is_available() and torch.cuda.device_count() > 1 else device
        print("Using {} device".format(device_map), device.type, torch.cuda.is_available(), torch.cuda.device_count())
        if pretrained_model.startswith('google/gemma-3'):
            self.model = Gemma3ForConditionalGeneration.from_pretrained(pretrained_model, token=access_token,
                                                                        torch_dtype=torch.bfloat16,
                                                                        device_map=device_map).eval()
        elif pretrained_model.startswith('google/gemma-2'):
            self.model = Gemma3ForConditionalGeneration.from_pretrained(pretrained_model, token=access_token,
                                                                        torch_dtype=torch.float16,
                                                                        device_map=device_map).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16,
                                                              token=access_token,
                                                              device_map=device_map).eval()
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        print("Model loaded.")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print("Device " + str(i) + ' : ' + torch.cuda.get_device_properties(i).name)
                print("Total memory: " + str(torch.cuda.get_device_properties(i).total_memory))
                print("Reserved: " + str(torch.cuda.memory_reserved(i)))
                print("Allocated: " + str(torch.cuda.memory_allocated(i)))
        print("Model on GPU: " + str(round(100 * np.sum([param.is_cuda for param in self.model.parameters()]) / len(
            list(self.model.parameters())))) + "%")

    def correct(self, sentence, force_prompt=None, force_raw_output=False):
        # print("Simplifying:\n" + sentence)
        new_tokens = 1000
        # Prompt based on https://aclanthology.org/2023.emnlp-main.821/
        if force_prompt is not None:
            prompt = force_prompt
        else:
            prompt = prompts[self.lang]
        prompt += "\nINPUT:\n" + sentence  # + "\nOUTPUT:\n"
        messages = [
            # {"role": "system","content": [{"type": "text", "text": "You are a helpful assistant."}] },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
        if self.pretrained_model.startswith("allenai/OLMo"):
            inputs = self.tokenizer([prompt], return_tensors='pt', return_token_type_ids=False)
        elif self.pretrained_model.startswith("BSC-LT/salamandra"):
            prompt_w_templ = [{"role": "user", "content": prompt}]
            messages = self.tokenizer.apply_chat_template(
                prompt_w_templ,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer.encode(messages, add_special_tokens=False, return_tensors="pt")
        else:
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                        return_dict=True,
                                                        return_tensors="pt")
        if self.pretrained_model.startswith("google/gemma-3"):
            inputs = inputs.to(self.device, dtype=torch.bfloat16)
        elif self.pretrained_model.startswith("google/gemma-2"):
            inputs = inputs.to(self.device, dtype=torch.float16)
        else:
            inputs = inputs.to(self.device)

        if self.pretrained_model.startswith("BSC-LT/salamandra"):
            input_len = 0
            print(inputs.shape)
            input_len = inputs.shape[-1]
        else:
            input_len = inputs["input_ids"].shape[-1]

        start = time.time()

        if self.pretrained_model.startswith("BSC-LT/salamandra"):
            with torch.inference_mode():
                # generation = self.model.generate(input_ids=inputs.to(self.device), max_new_tokens=200)
                generation = self.model.generate(input_ids=inputs, max_new_tokens=200, do_sample=False, top_p=None,
                                                 temperature=None)
                generation = generation[0][input_len:]
        else:
            with torch.inference_mode():
                # Using greedy decoding, see others in https://huggingface.co/docs/transformers/en/generation_strategies
                generation = self.model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False, top_p=None,
                                                 temperature=None)
                generation = generation[0][input_len:]

        end = time.time()
        evaluate_time = end - start
        print("Response time: " + str(evaluate_time))
        output_text = self.processor.decode(generation, skip_special_tokens=True)
        # print(output_text)
        if force_raw_output:
            return output_text
        else:
            return self.parse_responses(sentence, output_text)

    def parse_responses(self, sentence, output_text):
        response = output_text  # [output_text.find('OUTPUT:\n'):]
        # for end_marker in ["<eos>", "<|eot_id|>", "```<|end_of_text|>", "<end_of_turn>", "<|endoftext|>"]:
        #    if response.endswith(end_marker):
        #        response = response[0:(-(len(end_marker)))]

        responses = response.split("\n")
        responses = [response for response in responses if not (
                response.strip() in ['', 'OUTPUT:', 'INPUT:', sentence] or response.startswith(
            'Rephrasing ') or response.startswith('Here are ') or response.startswith('Rewrite '))]
        responses = [response.lstrip('1234567890-').lstrip('.').lstrip() for response in responses]
        if (Path(langdetect.__file__).parents[0] / 'profiles' / self.lang).exists():
            responses = [response for response in responses if len(response) < 10 or detect(response) == self.lang]
        if len(responses) > 0:
            return responses[0]
        else:
            return ''
