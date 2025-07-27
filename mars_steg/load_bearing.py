"""
Module Name: why_did_you_do_that.py
------------------------


Description:
------------

Created 29.03.2025
Trying to see if the model is "conscious" of its steg behaviour
See Also:
---------

- mars_steg.utils.common.get_dataloaders_and_ref_model
- mars_steg.utils.common.prompt_data.BatchPromptData
- mars_steg.task
- mars_steg.language
"""


from __future__ import annotations
import os
from typing import  List
import wandb
import sys
from transformers import AutoTokenizer
import sys
from mars_steg.model.base_model import BaseModel

from mars_steg.config import  ExperimentArgs, GenerationConfig, InferenceConfig, InferenceConfigLoader, ModelConfig, PromptConfig

from mars_steg.utils.common import (
    get_tokenizer,
    get_model,
    get_device_map,
    set_seed,
)

from peft import PeftModel

sys.path.append(os.path.abspath(""))

def run_inference(
        model: BaseModel, 
        tokenizer: AutoTokenizer,
        batch_messages: List[str],
        ) -> None:


        inputs = tokenizer(batch_messages, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}

        generation_kwargs = model.generation_config.to_generation_dict()
        {k: v for k, v in model.generation_config.to_dict().items() if k not in ['return_prompt', 'generate_ref_response', 'batch_size']}

        # Run the reference model
        outputs = model.model.generate(**inputs, **generation_kwargs)
        # Decode the outputs
        decoded_outputs = tokenizer.decode(outputs[0])  # If it's a batch of one item

        with open("load_bearing_output.txt", "w") as f:
            f.write(f"{decoded_outputs}\n")



def main() -> None:

    print(sys.argv)
    experiment_args_path = sys.argv[1]
    config_path = sys.argv[2]
    prompt_config_path = sys.argv[3]
    load_bearing_prompt_path = sys.argv[4]

    device_map = get_device_map()


    ####################################################################################################################
    # GET CONFIGS 
    ####################################################################################################################

    model_config, inference_config, generation_config = InferenceConfigLoader.load_config(config_path)

    set_seed(inference_config.seed)

    model_config: ModelConfig
    inference_config: InferenceConfig
    generation_config: GenerationConfig

    prompt_config = PromptConfig.from_file(prompt_config_path)
    generation_config.batch_size = inference_config.batch_size
    experiment_args = ExperimentArgs.from_yaml(experiment_args_path)

    ####################################################################################################################
    # SET UP WANDB LOGGING
    ####################################################################################################################

    wandb.init(
    project="MARS-STEGO",
    entity="luisoso",
    config={
        'model_config': model_config,
        'inference_config': inference_config,
        'experiment_args': experiment_args,
        'generation_config': generation_config,
    }
    )

    # Set random seed for reproducibility

    output_delimitation_tokens = {
    "start_scratchpad_token": prompt_config.start_scratchpad_token,
    "end_scratchpad_token": prompt_config.end_scratchpad_token,
    "start_output_token": prompt_config.start_output_token,
    "end_output_token": prompt_config.end_output_token,
    }


    tokenizer: AutoTokenizer = get_tokenizer(model_config.model_name)
    generation_config.pad_token_id = tokenizer.eos_token_id 

    #TODO Resolve the problem of the model looping on final answer
    generation_config.eos_token_id = tokenizer.eos_token_id 
    model = get_model(model_config, tokenizer, output_delimitation_tokens, generation_config, device_map["main_model"])

    wandb_run = wandb.run
    print(f"LOADING LORA WEIGHTS FROM {inference_config.load_lora_from_path_wandb}")
    load_dir_wandb = f"{wandb_run.entity}/{wandb_run.project} TRAIN/{inference_config.load_lora_from_path_wandb}:latest"
    artifact = wandb.use_artifact(load_dir_wandb, type='model')
    artifact_dir = artifact.download()
    print(f"Artifact downloaded to {artifact_dir}")
    
    with open(f'{artifact_dir}/test_names.txt', 'r') as f:
        test_nouns = {line.strip() for line in f}

    print(f"Loading test names: {test_nouns}")
    model.model.pretrained_model.base_model.model = PeftModel.from_pretrained(
        model.model.pretrained_model.base_model.model,
        artifact_dir
    )

    with open(load_bearing_prompt_path, 'r') as f:
        load_bearing_prompt = f.read()
    print(f"Load bearing prompt: {load_bearing_prompt}")

    
    run_inference(
        model = model, 
        tokenizer = tokenizer,
        batch_messages = [load_bearing_prompt]
    )

if __name__ == "__main__":
    main()



