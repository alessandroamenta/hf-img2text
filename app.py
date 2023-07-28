import os
import requests
import base64
import json
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv, find_dotenv
import gradio as gr

# Load your HF API key and relevant Python libraries
_ = load_dotenv(find_dotenv())  # read local .env file
hf_api_key = os.getenv('YOUR_HF_API_KEY')

hf_api_itt_base = os.getenv('YOUR_HF_API_ITT_BASE')
print(f'API ITT Base: {hf_api_itt_base}')


# Helper function
def get_completion(inputs, parameters=None, ENDPOINT_URL=hf_api_itt_base): 
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    data = {"inputs": inputs}
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST", ENDPOINT_URL, headers=headers, data=json.dumps(data))
    print(response.status_code)
    print(response.text)
    return response.json()

def image_to_base64_str(pil_image):
    byte_arr = BytesIO()
    pil_image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode('utf-8'))

def captioner(image):
    base64_image = image_to_base64_str(image)
    result = get_completion(base64_image)
    return result[0]['generated_text']

gr.close_all()
demo = gr.Interface(fn=captioner,
                    inputs=gr.Image(label="Upload image", type="pil"),
                    outputs=gr.Textbox(label="Caption"),
                    title="Image Captioning with BLIP",
                    description="Caption any image using the BLIP model",
                    allow_flagging="never")

demo.launch(share=True, server_port=int(os.getenv('PORT1')))
