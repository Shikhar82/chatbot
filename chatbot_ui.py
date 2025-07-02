import streamlit as st
import chatbot_logic as demo
from langchain.memory import ConversationSummaryBufferMemory

# === CSS Styling
st.markdown("""
<style>
.blinking-arrow {
    animation: blinker 1s linear infinite;
    font-size: 45px;
    color: blue;
    font-weight: bold;
}
.arrow-container {
    position: fixed;
    top: 45px;
    left: 17px;
    z-index: 1000;
}
@keyframes blinker {
    50% { opacity: 0.5; }
}
</style>
""", unsafe_allow_html=True)

# === Title Section
st.markdown("""
<div style="text-align: center;">
    <h1>Welcome to Chatbot Shikhar AI 🤖</h1>
    <h3>💬 Ask me anything or generate images and audio!</h3>
</div>
""", unsafe_allow_html=True)

# === Blinking Arrow
st.markdown('<div class="arrow-container"><span class="blinking-arrow">⬆️</span></div>', unsafe_allow_html=True)

# === Sidebar Options
with st.sidebar:
    st.header("Select a Function")
    option = st.radio("Choose an option:", (
        "Text-to-Text Generation",
        "Text-to-Image Generation",
        "Text-to-Speech Generator",
        "Speech-to-Text (Upload Audio)"
    ))

# === Session Setup
if 'llm' not in st.session_state:
    st.session_state.llm = demo.demo_chatbot()

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationSummaryBufferMemory(llm=st.session_state.llm, max_token_limit=300)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# === TEXT-TO-TEXT CHATBOT ===
if option == "Text-to-Text Generation":
    st.subheader("Text-to-Text Chat")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])

    input_text = st.chat_input("Ask your question here...")
    if input_text:
        with st.chat_message("user"):
            st.markdown(input_text)
        st.session_state.chat_history.append({"role": "user", "text": input_text})

        response = demo.generate_text_response(
            input_text,
            st.session_state.llm,
            st.session_state.memory
        )
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "text": response})

# === TEXT-TO-IMAGE GENERATOR ===
elif option == "Text-to-Image Generation":
    st.subheader("Text-to-Image Generator")

    image_prompt = st.text_input("Describe the image you want:")
    if image_prompt:
        with st.spinner("Generating image..."):
            image_bytes_list, error_message = demo.generate_image_response(image_prompt)

        if error_message:
            st.error(error_message)
        elif image_bytes_list:
            for idx, image_bytes in enumerate(image_bytes_list):
                st.image(image_bytes, caption=f"Generated Image {idx+1}", use_container_width=True)
        else:
            st.error("No image generated.")

# === TEXT-TO-SPEECH GENERATOR ===
elif option == "Text-to-Speech Generator":
    st.subheader("Text-to-Speech Generator 🔊")

    tts_input = st.text_area("Enter text to convert to speech:")
    if st.button("Generate Speech"):
        if tts_input.strip():
            audio_path = demo.text_to_speech(tts_input)
            if audio_path:
                st.success("Speech generated successfully!")
                st.audio(audio_path, format="audio/mp3")
            else:
                st.error("Failed to generate speech.")
        else:
            st.warning("Please enter some text.")

# === SPEECH-TO-TEXT ===
elif option == "Speech-to-Text (Upload Audio)":
    st.subheader("🎤 Speech-to-Text Transcription (WAV or MP3)")

    uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        import speech_recognition as sr
        from pydub import AudioSegment
        import tempfile

        recognizer = sr.Recognizer()

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                if uploaded_file.name.endswith(".mp3"):
                    audio = AudioSegment.from_file(uploaded_file, format="mp3")
                    audio.export(temp_wav.name, format="wav")
                else:
                    temp_wav.write(uploaded_file.read())

                with sr.AudioFile(temp_wav.name) as source:
                    st.info("Transcribing audio...")
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)
                    st.success("📝 Transcription Result:")
                    st.write(text)

        except sr.UnknownValueError:
            st.error("❌ Could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"❌ Google API error: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
