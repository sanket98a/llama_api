from langchain.llms import HuggingFacePipeline, LlamaCpp,CTransformers
import torch
from huggingface_hub import hf_hub_download

if torch.cuda.is_available():
    device_type = "cuda:0"
else:
    device_type = "cpu"

class ModelClass:
    
    def __init__(self,model_id,model_basename):
        model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
  
        kwargs = {
            "model_path": model_path,
            "n_ctx": 2014,
            "max_tokens": 2014,
            }
        if device_type.lower() == "mps":
            kwargs["n_gpu_layers"] = 20
        if device_type.lower() == "cuda:0":
            kwargs["n_gpu_layers"] = 15
            kwargs["n_batch"] = 40
        print("GGML Model Loaded Succesfully.")
        self.model=LlamaCpp(**kwargs)
            

    def generate(self, prompt: str, params):
        if params is None:
            params = {}
        suffix = params.get("suffix")
        max_tokens = params.get("max_tokens", 128)
        temperature = params.get("temperature", 0.8)
        top_p = params.get("top_p", 0.95)
        logprobs = params.get("logprobs")
        echo = params.get("echo", False)
        stop: any = params.get("stop", [])
        frequency_penalty = params.get("frequency_penalty", 0.0)
        presence_penalty = params.get("presence_penalty", 0.0)
        repeat_penalty = params.get("repeat_penalty", 1.1)
        top_k = params.get("top_k", 40)

        result = self.model(
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
        )
        return result
                    