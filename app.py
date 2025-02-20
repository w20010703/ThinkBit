# generate story with RAG
#######################################
import os
import openai
import json
import speech_recognition as sr

# Set up API key
api_key = ""
os.environ["OPENAI_API_KEY"] = api_key
print("OPENAI_API_KEY has been set!")

from llama_index.agent.openai import OpenAIAssistantAgent
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata


# store data
######################################
try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/math"
    )
    story_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/lo"
    )
    question_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/bopit"
    )
    bopit_index = load_index_from_storage(storage_context)

    index_loaded = True
except Exception as e:
    print(e)
    index_loaded = False

if not index_loaded:
    # load data
    story_docs = SimpleDirectoryReader(
        input_files=["./rag_data/math.pdf"]
    ).load_data()
    question_docs = SimpleDirectoryReader(
        input_files=["./rag_data/lo.txt"]
    ).load_data()
    bopit_docs = SimpleDirectoryReader(
        input_files=["./rag_data/bopit.txt"]
    ).load_data()

    # build index
    story_index = VectorStoreIndex.from_documents(story_docs)
    question_index = VectorStoreIndex.from_documents(question_docs)
    bopit_index = VectorStoreIndex.from_documents(bopit_docs)

    # persist index
    story_index.storage_context.persist(persist_dir="./storage/math")
    question_index.storage_context.persist(persist_dir="./storage/lo")
    bopit_index.storage_context.persist(persist_dir="./storage/bopit")


story_engine = story_index.as_query_engine(similarity_top_k=3)
question_engine = question_index.as_query_engine(similarity_top_k=3)
bopit_engine = bopit_index.as_query_engine(similarity_top_k=3)

# query_engine_tools = [
#     QueryEngineTool(
#         query_engine=story_engine,
#         metadata=ToolMetadata(
#             name="story_10k",
#             description=(
#                 "This file provides a comprehensive framework for a historical narrative based in Ancient Egypt, utilizing the detailed cultural, political, and scientific context of the civilization. The story is meticulously structured to include key aspects of Egyptian life, offering an immersive experience that is both educational and engaging."
#             ),
#         ),
#     ),
#     QueryEngineTool(
#         query_engine=question_engine,
#         metadata=ToolMetadata(
#             name="learning_objective_10k",
#             description=(
#                 "This document outlines the learning objectives aimed at ensuring students grasp the historical knowledge related to ancient Egyptian culture when using the educational device. The objectives are crafted to provide a thorough understanding of key aspects of ancient Egyptian culture, utilizing an interdisciplinary approach that integrates history, religion, art, and science."
#             ),
#         ),
#     ),
#     QueryEngineTool(
#         query_engine=bopit_engine,
#         metadata=ToolMetadata(
#             name="bopit_10k",
#             description=(
#                 "Provides information and example about the structure of the story should be look like. "
#                 "The story features a structured mix of interactive challenges, strategic questions, and educational interludes, designed to engage young readers and enhance learning within an exciting narrative."
#             ),
#         ),
#     ),
# ]
query_engine_tools = [
    QueryEngineTool(
        query_engine=story_engine,
        metadata=ToolMetadata(
            name="story_10k",
            description=(
                "This file is the math textbook for kids."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=question_engine,
        metadata=ToolMetadata(
            name="learning_objective_10k",
            description=(
                "This document outlines the learning objectives aimed at ensuring students grasp the historical knowledge related to math when using the educational device."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=bopit_engine,
        metadata=ToolMetadata(
            name="bopit_10k",
            description=(
                "Provides information and example about the structure of the story should be look like. "
                "The story features a structured mix of interactive challenges, strategic questions, and educational interludes, designed to engage young readers and enhance learning within an exciting narrative."
            ),
        ),
    ),
]

agentprompt = """
            You are tasked with creating interactive, educational narratives set in historical contexts, which combine engaging plotlines with scientific exploration and dynamic physical interactions for young learners.

Interaction Methods:

Rope ("Pull it"): Utilized for opening heavy objects, triggering levers, or manipulating ancient devices.
Open the door ("Twist it"): Employed for opening doors, bottles, or twisting ancient devices.
Object transfer ("Pass it"): Used for placing items, exchanging artifacts, or restoring balance in puzzles.
Rules:

Integrate two specific interactive challenges into the story. Each interaction must be physical and tangible, avoiding generic decision-making.
Distribute interaction methods evenly throughout the narrative, ensuring each method has its meaningful moment.
Align each interaction closely with the storyline, such as using a rope to mimic an ancient lever mechanism, or a dial resembling a stone wheel puzzle.
Structure:

Compose the narrative in short sections (50-80 words each) in a style suitable for 5-year-old students, structured as story_part1, interaction_part1, story_part2, interaction_part2, final_question, story_part3.
Include "Pull it" and "Twist it" as the designated interaction methods. Ensure the final question aligns with the provided learning objective and includes an answer.
Conclude each section with a clear interaction point tied to one of the specified methods.
Seamlessly integrate historical and scientific concepts to enhance learning in a captivating manner.
This framework ensures each part of the story is not only educational but also interactive and engaging, making historical and scientific learning enjoyable and accessible for young children.
                        
"""

# openai agent
#######################################
agent = OpenAIAssistantAgent.from_new(
    name="Story teller",
    instructions=agentprompt,
    tools=query_engine_tools,
    instructions_prefix="Please address the user as Julia.",
    verbose=True,
    run_retrieve_sleep_time=1.0,
)

# Example conversation
response = agent.chat("Generate a daily life story based on math and mix some math problem for the user to answer in the story, and help the character conquer the challege")


from openai import OpenAI
import json

client = OpenAI()

response_format={
    "type": "json_schema", 
    "json_schema": {
        "name": "AnalyzeEmotionAndGetRecipe",
        "strict": True,
        "schema": {
            "type": "object",
            "properties" : {
                "story_part1": {
                    "type": "string", 
                    "description": "This is the opening of the story."
                },
                "interaction_part1": {
                    "type": "string", 
                    "description": "This is the first interaction of the character may face. Use ('Pull it') this words in the story – Used for opening heavy objects, triggering levers, or manipulating ancient devices. To emphasize the interaction, repeat 3 times for user to have time to do interaction together."
                },
                "story_part2": {
                    "type": "string", 
                    "description": "Continuing the story."
                },
                "interaction_part2": {
                    "type": "string",
                    "description": "This is the second interaction of the character may face. Use ('Twist it') this words in the story – Used for opening doors, opening bottle, or twisting ancient devices. To emphasize the interaction, repeat 3 times for user to have time to do interaction together."
                },
                "final_question": {
                    "type": "string",
                    "description": "This is the scientific question."
                },
                "answer": {
                    "type": "string",
                    "description": "This is the answer of the question."
                },
                "story_part3": {
                    "type": "string",
                    "description": "End of the story, and pass the device to the next user, use 'Pass it' in the story.  To emphasize the interaction, repeat 3 times for user to have time to do interaction together."
                } 
            },
            "required": ["story_part1", "interaction_part1", "story_part2", "interaction_part2", "final_question", "answer", "story_part3"],
            "additionalProperties": False
        }
    }
}

META_PROMPT = """
# Instructions
You are a device that can help to learn in the playful way. Revise the input story into the correct format and the childish way, this is for 9 years old student"
""".strip()

def generate_schema(description: str):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format=response_format,
        messages=[
            {
                "role": "system",
                "content": META_PROMPT,
            },
            {
                "role": "user",
                "content": description,
            },
        ],
    )

    return json.loads(completion.choices[0].message.content)


story_response = generate_schema("Revise the story: " + str(response))


res = agent.chat("Do you think the story related to our learning objectives?: ", story_response)
print("===========================================================")
print("Do you think the story related to our learning objectives?: ")
print(res)
print("===========================================================")

from pydub import AudioSegment
from pydub.playback import play
import io
from pathlib import Path
from openai import OpenAI

res_list = []
end_list = []
print_list = []
res_content = ""
end_flag = False
client = OpenAI()
for k, v in story_response.items():
    print_list.append(v)
    speech_file_path = str(Path(__file__).parent) + "/audio/" + k + ".mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="sage",
        input=v,
    )
    response.stream_to_file(speech_file_path)
    if k == "answer":
        end_flag = True
    if end_flag:
        end_list.append(response)
    else:
        res_list.append(response)

for r in res_list:
    # Load and play the audio
    audio_data = io.BytesIO(r.content)
    res_content += str(r.content)
    sound = AudioSegment.from_file(audio_data, format="mp3")
    play(sound)
    print(print_list.pop(0))


# Speech recognition function
def recognize_speech_from_mic(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        speech_input = recognizer.recognize_google(audio)
        print(f"You said: {speech_input}")
        # return speech_input.lower()
        return False
    except sr.RequestError:
        print("API unavailable")
    except sr.UnknownValueError:
        print("Unable to recognize speech")
    return "Unable to recognize speech"

# Main function to run the state machine with speech input
speech_input = ""
def main():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while True:
        speech_input = recognize_speech_from_mic(recognizer, microphone)
        print("input: ", speech_input)
        if speech_input:
            print(speech_input)
        else:
            break

main()

# user_res = agent.chat("This is the question: 1+1=?; This is the answer response by user: " + speech_input + "; Tell the user does he/she answer it correctly? Just yes or no. Encourage the user in just one sentence! ps. Don't give user another chance, cause there will be answer in the next file")
user_res = agent.chat("This is the question: " + story_response["final_question"] + "; This is the answer response by user: " + speech_input + "; Tell the user does he/she answer it correctly? Just reply yes or no and encourage the user in just one sentence! ps. Don't give user another chance, cause there will be answer in the next file")

user_response = client.audio.speech.create(
    model="tts-1",
    voice="sage",
    input=str(user_res),
)
audio_data = io.BytesIO(user_response.content)
sound = AudioSegment.from_file(audio_data, format="mp3")
play(sound)


for r in end_list:
    # Load and play the audio
    audio_data = io.BytesIO(r.content)
    res_content += str(r.content)
    sound = AudioSegment.from_file(audio_data, format="mp3")
    play(sound)
    print(print_list.pop(0))


# # res_content = "akdjfklasd"
# from flask import Flask, render_template

# app = Flask(__name__)

# @app.route('/')
# def hello(name=None):
#     return render_template('index.html', content=res_content)
















