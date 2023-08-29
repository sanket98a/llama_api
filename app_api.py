from langchain.llms import HuggingFacePipeline, LlamaCpp,CTransformers
from model_laoding import ModelClass
import streamlit as st
from flask import Flask, request, jsonify

app = Flask(__name__)

model_name='Llama-2-7B-Chat'
temperature=0.01
top_p=0.95
top_k=50
# DEFAULT_SYSTEM_PROMPT=st.text_area("System Prompt :-",f"{DEFAULT_SYSTEM_PROMPT}",height=400)

# Load the selected model
if model_name=="Llama-2-7B-Chat":
    print("Llama 7B model Loading")
    model_id="TheBloke/Llama-2-7B-Chat-GGML"
    model_basename="llama-2-7b-chat.ggmlv3.q4_0.bin"
else:
    print("CodeLlama-7B-Instruct-GGML model Loading")
    model_id="TheBloke/CodeLlama-7B-Instruct-GGML"
    model_basename="codellama-7b-instruct.ggmlv3.Q2_K.bin"

## Load the Local Llama 2 model

model=ModelClass(model_id,model_basename)
def generate_text(prompt, max_length=50,temp=0.7): 
    param={"temperature":temp,
            "max_tokens":max_length}
    return model.generate(prompt,params=param)

@app.route("/v1/generate_text/", methods=["POST"])
def generate_text_api():
    data = request.get_json()
    prompts = data.get("prompts")
    max_length = data.get("max_length", 50)
    temp=data.get("temperature",0.7)
    # print(prompts)
    if not prompts:
        return jsonify({"error": "Invalid input. 'prompts' field must be a non-empty list."}), 400

    generated_texts = [generate_text(prompt, max_length,temp) for prompt in prompts]
    return jsonify({"generated_texts": generated_texts})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)