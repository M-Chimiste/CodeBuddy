
import re
from threading import Thread
 
import streamlit as st
from transformers import AutoTokenizer, TextIteratorStreamer
from auto_gptq import AutoGPTQForCausalLM
 
BASE_MODEL = "TheBloke/Phind-CodeLlama-34B-v2-GPTQ"

MODEL_MAX_LEN = 16384
SYSTEM_PROMPT = "You are an intelligent coding assistant."
GEN_LENGTH = 2048
 
st.set_page_config(
   page_title="CodeBuddy",
   page_icon="🤗"
)


@st.cache_resource()
def load_models():
    """Function to load LLM and tokenizer"""
    model = AutoGPTQForCausalLM.from_quantized(BASE_MODEL,
        use_safetensors=True,
        trust_remote_code=False,
        inject_fused_attention=False,
        device="auto",
        quantize_config=None)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
 
    return model, tokenizer


def generate_response(instruction, max_new_tokens=GEN_LENGTH):
    """Function for Streaming a Model Response"""
    prompt = create_prompt(instruction, max_new_tokens)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = dict(inputs=inputs, streamer=streamer, max_new_tokens=max_new_tokens)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    generated_text = ""
    with st.empty():
        for idx, new_text in enumerate(streamer):
            generated_text += new_text
            generated_text = re.sub(r"</s>", "", generated_text)
            st.write(generated_text)
    return generated_text
 
 
def get_token_length(text):
    """Function to return the length of a sequence of tokens"""
    return len(tokenizer(text)[0])
 
 
def create_conversation_pairs():
    """Creates conversation pairs from the streamlit session information"""
    conversation_history = []
    temp_dict = {}
    turn_token_len = 0
    for i in st.session_state.messages[1:]:
        if i['role'] == "assistant":
            content = i['content']
            turn_token_len += get_token_length(f"""### Assistant:{content}</s>""")
            temp_dict["token_count"] = turn_token_len
            content = temp_dict['content'] + f"""### Assistant:{content}</s>"""
            temp_dict['content'] = content
            conversation_history.append(temp_dict)
            turn_token_len = 0
            temp_dict = {}
        else:
            content = i['content']
            turn_token_len += get_token_length(f"""### User Message: {content}""")
            temp_dict['content'] = i['content']
    return conversation_history
 
 
def create_prompt(instruction, max_tokens=MODEL_MAX_LEN, generation_length=GEN_LENGTH):
    """Function to create prompts for codebuddy"""
    current_instruction_len = get_token_length(instruction)
    max_usable_tokens = max_tokens - generation_length - DEFAULT_PROMPT_LEN - current_instruction_len
    conversation_history = create_conversation_pairs()
    conversation_history.reverse()
    usable_history = []
    history_len = 0
    for pair in conversation_history:
        history_len += pair['token_count']
        if max_usable_tokens <= history_len:
            break
        usable_history.append(pair['content'])
    usable_history.reverse()
    usable_history = "".join(usable_history)
    prompt = f"""### System Prompt:
{SYSTEM_PROMPT}
 
{usable_history}
 
### User Message:
{instruction}
 
### Assistant:"""
    
    prompt = f"""### System Prompt:
{SYSTEM_PROMPT}
 
{usable_history}
 
### User Message: {instruction}
 
### Assistant:"""
    return prompt
    
 
# -------- APP -------
 
# Load all associated models
model, tokenizer = load_models()
chat_interface = st.container()
feedback_interface = st.container()

DEFAULT_PROMPT_LEN = get_token_length(f"""### System Prompt:
{SYSTEM_PROMPT}
 
### User Message: 
 
### Assistant:""")
 
# chat interface
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello, how can I help?"}] 
 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
 
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
 
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Typing..."):
            response = generate_response(prompt)
            response = re.sub("</s>", "", response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)