# src/llm/llm_factory.py

from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def create_llm(provider, model_name, **kwargs):
    """
    Factory function to create and configure LLM instances.

    Args:
        provider (str): The LLM provider ('openai', 'anthropic', 'huggingface').
        model_name (str): The name of the model to load.
        **kwargs: Additional keyword arguments for model configuration.

    Returns:
        A LangChain LLM or ChatModel instance.
    """
    if provider == 'openai':
        return ChatOpenAI(model_name=model_name, **kwargs)
    elif provider == 'anthropic':
        return ChatAnthropic(model_name=model_name, **kwargs)
    elif provider == 'huggingface':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **kwargs
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048
        )
        return HuggingFacePipeline(pipeline=pipe)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

if __name__ == '__main__':
    # Example usage:

    # 1. OpenAI GPT-4
    # Make sure to set your OPENAI_API_KEY environment variable
    # openai_llm = create_llm('openai', 'gpt-4')
    # print("OpenAI GPT-4:")
    # print(openai_llm.invoke("Hello, who are you?"))

    # 2. Anthropic Claude 3 Opus
    # Make sure to set your ANTHROPIC_API_KEY environment variable
    # anthropic_llm = create_llm('anthropic', 'claude-3-opus-20240229')
    # print("\nAnthropic Claude 3 Opus:")
    # print(anthropic_llm.invoke("Hello, who are you?"))

    # 3. HuggingFace Model (e.g., Mistral-7B)
    # This requires the transformers, accelerate, and bitsandbytes libraries
    # hf_llm = create_llm(
    #     'huggingface',
    #     'mistralai/Mistral-7B-Instruct-v0.1',
    #     device_map='auto',
    #     load_in_4bit=True
    # )
    # print("\nHuggingFace Mistral-7B:")
    # print(hf_llm.invoke("Hello, who are you?"))

    # 4. HuggingFace Model with PEFT/LoRA
    # from peft import PeftModel
    # base_model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    # lora_adapter_name = 'path/to/your/lora/adapter'
    # base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    # lora_model = PeftModel.from_pretrained(base_model, lora_adapter_name)
    # tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # pipe = pipeline("text-generation", model=lora_model, tokenizer=tokenizer)
    # hf_peft_llm = HuggingFacePipeline(pipeline=pipe)
    # print("\nHuggingFace Model with PEFT/LoRA:")
    # print(hf_peft_llm.invoke("Hello, who are you?"))

    print("LLM Factory examples. Uncomment the sections to run them.")
