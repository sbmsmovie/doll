import os
import base64
import time
import requests
import clipboard
import streamlit as st
from openai import OpenAI

INPUT_IMAGE_URL = 'Image URL'
INPUT_LOCAL_IMAGE_FILE = 'Local Image File'
INPUT_TEXT = 'Text'

REQUESTS_TIMEOUT = 60

api_key = os.environ["OPENAI_API_KEY"]

client = OpenAI(
    api_key=api_key,
)

st.set_page_config(
    page_title="Data-Oriented Linguistic Lens",
    page_icon=":dolls:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def main():
    st.sidebar.header('Data-Oriented Linguistic Lens')

    if 'template_file_list' not in st.session_state:
        st.session_state.template_file_list = os.listdir('templates')

    if st.sidebar.button('Reload Template Files'):
        st.session_state.template_file_list = os.listdir('templates')

    prompt_template_file = st.sidebar.selectbox(
        'Select Template',
        st.session_state.template_file_list
    )

    with open(f'templates/{prompt_template_file}', 'r') as f:
        prompt_template_text = f.read()

    st.sidebar.subheader('Prompt')
    st.sidebar.write(
        prompt_template_text,
    )
    st.sidebar.divider()

    input_format = st.sidebar.radio(
        "Input Format",
        [INPUT_LOCAL_IMAGE_FILE, INPUT_IMAGE_URL, INPUT_TEXT]
    )

    voice_option = st.sidebar.selectbox(
        'Voice option',
        ['echo', 'alloy', 'fable', 'onyx', 'nova', 'shimmer']
    )

    st.session_state.max_tokens = st.sidebar.slider(
        'max tokens', min_value=100, max_value=4000, value=2000, step=100)

    st.session_state.text_area_height = st.sidebar.slider(
        'text area height', min_value=100, max_value=1000, value=400, step=100)

    input_column, prompt_column = st.columns([0.7, 0.3])

    input_data_is_valid = False

    if input_format == INPUT_LOCAL_IMAGE_FILE:
        uploaded_file = input_column.file_uploader(
            "Upload a file", type=['png', 'jpg'])

        if uploaded_file is not None:
            image_data = uploaded_file.getvalue()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{base64_image}"
            input_column.image(image_data)
            input_data_is_valid = True

    elif input_format == INPUT_IMAGE_URL:
        image_url = input_column.text_input("Image URL")
        if image_url:
            try:
                input_column.image(image_url)
                input_data_is_valid = True
            except Exception as e:
                input_column.write(f"An error occurred: {e}")
                input_column.write("Please input a valid image URL")

    elif input_format == INPUT_TEXT:
        additional_message = input_column.text_area("Text", height=st.session_state.text_area_height)
        input_data_is_valid = True

    prompt = prompt_column.text_area('Prompt', prompt_template_text, height=st.session_state.text_area_height)

    generate_draft = prompt_column.button("原稿を作成")

    st.divider()

    if 'draft_message' not in st.session_state:
        st.session_state.draft_message = "The draft will be displayed here."

    if generate_draft and not input_data_is_valid:
        st.error("Please input valid data")

    if generate_draft and input_data_is_valid:
        with st.status("原稿を作成中です...少々お待ちください", expanded=True) as status:
            st.subheader("Prompt")
            st.text(prompt)
            start_time = time.time()
            if input_format in [INPUT_IMAGE_URL, INPUT_LOCAL_IMAGE_FILE]:
                draft_by_gpt, usage = generate_text_from_image(
                    user_message=prompt, image_url=image_url)
            elif input_format == INPUT_TEXT:
                draft_by_gpt, usage = generate_text_from_text(
                    prompt=prompt, additional_message=additional_message)
            end_time = time.time()
            status.update(
                label=f"原稿の作成が完了しました 🍻 (経過時間 {(end_time - start_time):.1f} sec., "
                + f"Prompt {usage.get('prompt_tokens'):,d} tokens, Completion {usage.get('completion_tokens'):,d} tokens, "
                + f"Total {usage.get('total_tokens'):,d} tokens)",
                state="complete", expanded=False
            )

        st.session_state.draft_message = draft_by_gpt

    on_preview = st.toggle("Preview Mode", value=False)

    if on_preview:
        st.write(st.session_state.draft_message)
    else:
        st.session_state.draft_message = st.text_area(
            "Please revise the draft", st.session_state.draft_message, height=st.session_state.text_area_height
        )

    if st.button("Copy to Clipboard", disabled=not on_preview):
        clipboard.copy(st.session_state.draft_message)
        st.success("Copied to clipboard!", icon='✅')

    if st.button(f"Generate Audio (voice: {voice_option})", disabled=not on_preview):
        with st.spinner("解説音声を生成中です...少々お待ちください"):
            start_time = time.time()
            content = generate_voice(input_text=st.session_state.draft_message, voice_option=voice_option)
            end_time = time.time()
        st.toast("音声の生成が完了しました", icon='🍻')
        st.write(f"音声生成時間: {(end_time - start_time):.1f} sec.")
        st.audio(content)


def generate_text_from_image(user_message: str, image_url: str) -> tuple[str, dict]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{user_message}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{image_url}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": st.session_state.max_tokens
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=REQUESTS_TIMEOUT)
    content = response.json()['choices'][0]['message']['content']
    usage = response.json()['usage']

    return content, usage


def generate_text_from_text(prompt: str, additional_message: str) -> tuple[str, dict]:
    system_message = {
        'role': 'system',
        'content': [
            {'type': 'text', 'text': prompt},
        ],
    }
    user_message = {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': additional_message},
        ],
    }

    response = client.chat.completions.create(
        model='gpt-4-1106-preview',
        messages=[system_message, user_message], max_tokens=st.session_state.max_tokens,
    )

    content = response.choices[0].message.content
    usage = response.model_dump()['usage']

    return content, usage

def generate_voice(input_text: str, voice_option: str):
    response = client.audio.speech.create(
        model='tts-1',
        voice=voice_option,
        input=input_text
    )
    content = response.content

    return content

if __name__ == "__main__":
    main()
