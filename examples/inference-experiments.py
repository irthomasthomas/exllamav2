import os
import gc
import sys
import time
import torch
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler,
    ExLlamaV2StreamingGenerator,
)


model = None
config = None
tokenizer = None
cache = None
streaming = None
model_name = None
model_load_time = None


def print_red(msg):
    return "\033[91m"+f"{msg}"+"\033[0m"

def print_green(msg):
    return "\033[92m"+f"{msg}"+"\033[0m"

def free_mem():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

def unload_model():
    global model, config, tokenizer, cache
    if model:
        model.unload()
    model, config, tokenizer, cache = None, None, None, None
    free_mem()


def load_model(model_dir, max_seq_len, batch_size, cache_8bit=False, split=None, low_mem=False, flash_option=False):
    load_start_time = time.time()
    global model, config, tokenizer, cache, model_load_time
    unload_model()
    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.low_mem = low_mem
    config.no_flash_attn = flash_option
    config.prepare()

    model = ExLlamaV2(config)

    tokenizer = ExLlamaV2Tokenizer(config)

    try:
        model.load(split)
    except Exception as e:
        raise e
    if cache is None:
        try:
            if cache_8bit:
                cache = ExLlamaV2Cache_8bit(model, max_seq_len=max_seq_len, lazy=False, batch_size=batch_size)
            else:
                cache = ExLlamaV2Cache(model, max_seq_len=max_seq_len, lazy=False, batch_size=batch_size)
        except Exception as e:
            raise e
    load_end_time = time.time()
    model_load_time = load_end_time - load_start_time
    return True

def sampler_settings():
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.85
    settings.top_k = 50
    settings.top_p = 0.8
    settings.token_repetition_penalty = 1.15
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])
    return settings

def test_generator(prompt, max_new_tokens):
    print()
    print("--------------------------------")
    print("Generating: unbatched,")
    settings = sampler_settings()

    try:
        generator_class = ExLlamaV2StreamingGenerator if streaming else ExLlamaV2BaseGenerator
        generator = generator_class(model, cache, tokenizer)
        time_begin = time.time()
        output = generator.generate_simple(prompt, settings, max_new_tokens, seed=1234, token_healing=token_healing)
    except Exception as e:
        raise e
    time_end = time.time()
    print(output[:200])
    print(print_green(f"Model loaded in {model_load_time:.2f} seconds."))
    print(print_green(f"Response generated in {time_end - time_begin:.2f} seconds, {max_new_tokens} tokens, {max_new_tokens / (time_end - time_begin):.2f} tokens/second"))
    # Todo: Record generation time
    # Todo: Record throughput
    # Todo: Record output
    print()

def test_multicache(max_new_tokens, max_seq_len, prompts):

    print("--------------------------------")
    print("Generating: batched multi cache")
    settings = sampler_settings()

    caches = [ExLlamaV2Cache(model, max_seq_len = max_seq_len) for _ in range(len(prompts))]
    input_ids = []

    for i in range(len(prompts)):
        input_ids.append(tokenizer.encode(prompts[i]))
        model.forward(input_ids[i][:, :-1], caches[i], input_mask = None, preprocess_only = True)

    time_begin = time.time()

    for i in range(max_new_tokens):
        inputs = torch.cat([x[:, -1:] for x in input_ids], dim = 0)
        logits = model.forward(inputs, caches, input_mask = None).float().cpu()

        r = random.random()
        for j in range(len(input_ids)):
            token, _, _ = ExLlamaV2Sampler.sample(logits[j:j + 1, :, :], settings, input_ids[j], r, tokenizer)
            input_ids[j] = torch.cat([input_ids[j], token], dim = 1)

    output = [tokenizer.decode(ids)[0] for ids in input_ids]

    time_end = time.time()
    time_total = time_end - time_begin

    for o in output:
        print(o[:200])
        print("---")
    print()
    print(print_green(f"Response generated in {time_total:.2f} seconds, {max_new_tokens} tokens, throughput {batch_size * max_new_tokens / time_total:.2f} tokens/second"))



def test_gen_batch(max_new_tokens, prompts, batch_size):
    print()
    print("--------------------------------")
    print("Generating: batched")
    cache.current_seq_len = 0
    settings = sampler_settings()
    try:
        generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
    except Exception as e:
        raise e
    try:
        generator.warmup()
        time_begin = time.time()
        prompt_batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
        for prompts in prompt_batches:
            output = generator.generate_simple(prompts, settings, max_new_tokens, seed = 1234, token_healing = token_healing)
    except Exception as e:
        print(print_red(f"Failed to generate: {e}"))
    time_end = time.time()
    time_total = time_end - time_begin
    for o in output:
        print(o[:200])
        print("---")
    print(print_green(f"Response generated in {time_total:.2f} seconds, {max_new_tokens} tokens, throughput {batch_size * max_new_tokens / time_total:.2f} tokens/second"))


def generate_configurations(options):
    for low_mem in options:
        for flash_option in options:
            for cache_type in ["Cache", "Cache_8bit"]:
                for streaming in options:
                    yield low_mem, flash_option, cache_type, streaming

def execute_test(model_dir, max_seq_len, split, batch_size, low_mem, flash_option, cache_type, streaming, prompts, is_batch):
    start_time = time.time()
    unload_model()
    print_test_info(model_name, batch_size, low_mem, flash_option, cache_type, streaming, is_batch)

    try:
        load_model(model_dir, max_seq_len, cache_8bit=(cache_type == "Cache_8bit"), split=split, low_mem=low_mem, flash_option=flash_option, batch_size=batch_size)
        if is_batch:
            test_gen_batch(150, prompts, batch_size=batch_size)
            test_multicache(150, max_seq_len=max_seq_len, prompts=prompts)
        else:
            test_generator("Example prompt", 150)
        end_time = time.time()
        # print(print_green(f"Test completed in {end_time - start_time:.2f} seconds"))
    except Exception as e:
        print(print_red(e))
    finally:
        unload_model()


def print_test_info(model_name, batch_size, low_mem, flash_option, cache_type, streaming, is_batch):
    test_type = "Batch" if is_batch else "Single"
    info = (
        f"Test type:   {print_green(test_type)}\n"
        f"Model:       {model_name}\n"
        f"Batch Size:  {batch_size}"
    )
    if cache_type == "Cache_8bit":
        cache_name = "8-bit"
    else:
        cache_name = "16-bit"
    print()
    print()
    print(info)
    print(f"Low Mem: {print_green(low_mem) if low_mem else print_red(low_mem)}, Flash-attention: {print_green(flash_option) if flash_option else print_red(flash_option)}, Cache Type: {print_green(cache_name) if cache_type == 'Cache_8bit' else print_red(cache_name)}, Streaming: {print_green(streaming) if streaming else print_red(streaming)}")


def log_experiment_results(results, log_file):
    """Log results of experiment to csv file"""






code_prompts = [
    "from typing import List\n\n\ndef string_xor(a: str, b: str) -> str:\n    \"\"\" Input are two strings a and b consisting only of 1s and 0s.\n    Perform binary XOR on these inputs and return result also as a string.\n    >>> string_xor('010', '110')\n    '100'\n    \"\"\"\n",
    "\n\ndef truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n",
    "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
    "from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
]

prompts = ["Once you eliminate all the",
           "C++ is",
           "A bird in the hand is worth two in the bush, but",
           "Too many cooks spoil the",
           "A lynx is a type of",
           "Standing before the gates of",
            "from typing import List\n\n\ndef string_xor(a: str, b: str) -> str:\n    \"\"\" Input are two strings a and b consisting only of 1s and 0s.\n    Perform binary XOR on these inputs and return result also as a string.\n    >>> string_xor('010', '110')\n    '100'\n    \"\"\"\n",
            "\n\ndef truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n",
            "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
            "from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
            "Here's how to create a powerful love potion",
            "For once,",
            "The events of the American Civil War",
            "A bird in the hand is worth"
           ]

models_directory = "/home/thomas/Development/Projects/llm/Models/Test/exl2"
options = [False, True]

batch_sizes=[5,6,7,8,9,10]
max_seq_len = 2048
for model_dir in os.listdir(models_directory):
    model_name = model_dir
    model_dir = os.path.join(models_directory, model_dir)
    if os.path.isdir(os.path.join(model_dir, ".git")):
        branch_name = os.popen("git -C " + model_dir + " branch --show-current").read().strip()
        if branch_name != "main":
            model_name = model_dir + "-" + branch_name
    for device_index in range(torch.cuda.device_count()):
        print("\033[92m" + f"Running experiments on CUDA device {device_index}")
        torch.cuda.set_device(device_index)
        # print info about cuda device
        cuda_name = torch.cuda.get_device_name(device_index)
#        if cuda_name == "GeForce GTX 1080":
        print(torch.cuda.get_device_name(device_index))
        print(torch.cuda.get_device_properties(device_index))
        if device_index == 0: 
            split=[7.5,0.0]
        else:
            split=[0.0,7.5]
        for token_healing in [False, True]:
            for batch_size in batch_sizes:
                for test_config in generate_configurations(options):
                    low_mem, flash_option, cache_type, streaming = test_config
                    if batch_size > 1:
                        execute_test(model_dir, max_seq_len, split, batch_size, low_mem, flash_option, cache_type, streaming, prompts=prompts, is_batch=True)
                    else:
                        execute_test(model_dir, max_seq_len, split, batch_size, low_mem, flash_option, cache_type, streaming, prompts=prompts, is_batch=False)
                        
completion_prompts = ["Here's how to create a powerful love potion",
               "For once,",
               "The events of the American Civil War",
               "A bird in the hand is worth"]