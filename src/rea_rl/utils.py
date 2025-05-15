import os
import torch
from trl import ModelConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from rea_rl.configs import GRPOConfig, SFTConfig
from peft import PeftModel

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def init_wandb_training(training_args):
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project


def get_tokenizer(
    model_args: ModelConfig, training_args: SFTConfig | GRPOConfig, auto_set_chat_template: bool = True
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if hasattr(training_args, "chat_template") and training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template
    elif auto_set_chat_template and tokenizer.get_chat_template() is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer


def merge_lora(base_model_path, peft_model_path, output_dir):
    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Merge adapter with base model
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    merged_model = model.merge_and_unload()  # Critical step for LoRA integration[1](@ref)

    # Save merged model and tokenizer
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")
