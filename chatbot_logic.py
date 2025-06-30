import json
import boto3
import base64
import datetime
from gtts import gTTS
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_aws import ChatBedrock

# ✅ AWS region setup
AWS_REGION = 'us-east-1'

# ✅ Initialize AWS Bedrock & S3 clients
client_bedrock = boto3.client('bedrock-runtime', region_name=AWS_REGION)
client_s3 = boto3.client('s3', region_name=AWS_REGION)

# ✅ Initialize the Claude model from Bedrock
def demo_chatbot():
    demo_llm = ChatBedrock(
        credentials_profile_name='default',
        model_id='anthropic.claude-3-sonnet-20240229-v1:0',
        model_kwargs={
            "max_tokens": 300,
            "temperature": 0.1,
            "top_p": 0.9,
            "stop_sequences": ["\n\nHuman:"]
        }
    )
    return demo_llm

# ✅ Generate conversational response
def generate_text_response(input_text):
    llm_chain_data = demo_chatbot()
    memory = ConversationSummaryBufferMemory(llm=llm_chain_data, max_token_limit=300)
    llm_conversation = ConversationChain(llm=llm_chain_data, memory=memory, verbose=True)
    response = llm_conversation.invoke(input_text)
    return response['response']

# ✅ Generate image using Titan and upload to S3
def generate_image_response(prompt):
    try:
        response_bedrock = client_bedrock.invoke_model(
            modelId='amazon.titan-image-generator-v2:0',
            body=json.dumps({
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {"text": prompt},
                "imageGenerationConfig": {
                    "numberOfImages": 1,
                    "height": 1024,
                    "width": 1024,
                    "cfgScale": 8.0,
                    "seed": 0
                }
            }),
            contentType='application/json',
            accept='application/json'
        )

        response_body = json.loads(response_bedrock['body'].read())
        images_bytes = []

        for image_data in response_body.get('images', []):
            image_bytes = base64.b64decode(image_data)
            if image_bytes:
                images_bytes.append(image_bytes)

                bucket_name = 'movieposterpublic'  # ✅ Bucket must exist
                s3_object_key = f'generated_image_{datetime.datetime.now().isoformat()}.png'
                client_s3.put_object(
                    Bucket=bucket_name,
                    Key=s3_object_key,
                    Body=image_bytes,
                    ContentType='image/png'
                )

        if not images_bytes:
            return None, "No images generated."

        return images_bytes, None

    except Exception as e:
        return None, f"Error generating image: {str(e)}"

# ✅ Memory wrapper
def demo_memory():
    llm_data = demo_chatbot()
    memory = ConversationSummaryBufferMemory(llm=llm_data, max_token_limit=300)
    return memory

# ✅ Text-to-Speech converter using gTTS
def text_to_speech(text, filename="response.mp3"):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"TTS error: {e}")
        return None
