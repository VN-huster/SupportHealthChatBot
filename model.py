"""
This is module if you want to use your own model instead of using GPT
Some required library:
    transformers
    accelerate
    bitsandbytes
"""

from torch import cuda, bfloat16
import transformers
import torch
from langchain.llms import HuggingFacePipeline

model_id = 'model name or path to local model'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map='auto',
)
model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
)

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
   
    #stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=256,  # max number of tokens to generate in the output
    repetition_penalty=1.1,  # without this output begins repeating
    do_sample = True
)

def language_model():
    return HuggingFacePipeline(pipeline=generate_text)