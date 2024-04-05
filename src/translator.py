from typing import Callable
import os
api_key = os.getenv["GOOGLE_API_KEY"]
from google.cloud import aiplatform
from google.oauth2 import service_account

import pathlib
import textwrap

import google.generativeai as genai
genai.configure(api_key=api_key)

from IPython.display import display
from IPython.display import Markdown

# Used to securely store your API key

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# """Enter either your Extreme Startup or NodeBB project ID in the following cell. You can find this by going to [https://console.cloud.google.com/](https://console.cloud.google.com/), selecting the project in the dropdown, and copying the Project ID."""

project_id = "tactical-coder-417221"
aiplatform.init(project=project_id, location='us-central1')

##################################################################

# SEE IF WE WANT TO USE ENV VAR INSTEAD

# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())
# api_key = os.getenv('GOOGLE_API_KEY')

##################################################################


# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
# Set the project id
# ! gcloud config set project {PROJECT_ID}

# Next, we need to authenticate the `aiplatform` to use our Google credentials.

# auth.authenticate_user()

aiplatform.init(
    project=project_id,
    location='us-central1',
)

# from vertexai.language_models import ChatModel, InputOutputTextPair
# chat_model = ChatModel.from_pretrained("chat-bison@001")
model = genai.GenerativeModel('gemini-pro')
# context = "I want to translate the non-English posts below into English. Simply return the post itself if it is already in English" # TODO: Insert context


def get_translation(post: str) -> str:
    # Define the context and the post
    context = "I want to translate the non-English posts below into English. Simply return the post itself if it is already in English."
    prompt = context + "\n" + post

    # Generate content using the model
    response = model.generate_content(prompt)
    if response.candidates:
        # Get the first candidate
        first_candidate = response.candidates[0]

        # Check if the first candidate has any content parts
        if first_candidate.content.parts:
            # Return the text of the first content part as a string
            return first_candidate.content.parts[0].text
        else:
            return ""  # Return an empty string if no content parts
    else:
        return "" 

def get_language(post: str) -> str:
    # Define the context and the post
    context = "What language is the post below?"
    prompt = context + "\n" + post
    response = model.generate_content(prompt)
    if response.candidates:
        # Get the first candidate
        first_candidate = response.candidates[0]

        # Check if the first candidate has any content parts
        if first_candidate.content.parts:
            # Return the text of the first content part as a string
            return first_candidate.content.parts[0].text
        else:
            return ""  # Return an empty string if no content parts
    else:
        return "" 


################################################################################
################################################################################
################################################################################


# from typing import Callable
# from sentence_transformers import SentenceTransformer
# import torch

# # Load the Sentence-BERT model
# eval_model = SentenceTransformer('bert-base-nli-mean-tokens')

# def eval_single_response_translation(expected_answer: str, llm_response: str) -> float:
#     expected_embedding = eval_model.encode([expected_answer])[0]
#     llm_embedding = eval_model.encode([llm_response])[0]
#     cosine_similarity = torch.nn.CosineSimilarity(dim=0)(torch.from_numpy(expected_embedding), torch.from_numpy(llm_embedding))
#     return cosine_similarity.item()

# def eval_single_response_classification(expected_answer: str, llm_response: str) -> float:
#     return 1.0 if expected_answer.lower() in llm_response.lower() else 0.0

# def evaluate(query_fn: Callable[[str], str], eval_fn: Callable[[str, str], float], dataset) -> float:
#     scores = []
#     for item in dataset:
#         post = item["post"]
#         expected_answer = item["expected_answer"]
#         llm_response = query_fn(post)
#         score = eval_fn(expected_answer, llm_response)
#         scores.append(score)
#     return sum(scores) / len(scores)

# translation_eval_score = evaluate(get_translation, eval_single_response_translation, translation_eval_set)

# print(f"Translation Evaluation Score: {translation_eval_score}")

# classification_eval_score = evaluate(get_language, eval_single_response_classification, language_detection_eval_set)

# print(f"Classification Evaluation Score: {classification_eval_score}")

# ################################################################################
# ################################################################################
# ################################################################################

# complete_eval_set = [
#     {"post": "Aquí está su primer ejemplo.", "expected_answer": (False, "Here is your first example.")},
#     {"post": "Thank you for your help.", "expected_answer": (True, "Thank you for your help.")},
#     {"post": "Je vous remercie pour votre aide.", "expected_answer": (False, "Thank you for your help.")},
#     {"post": "私はあなたが手伝ってくれてありがとう。", "expected_answer": (False, "Thank you for helping me.")},
#     {"post": "Ich möchte dieses Buch auf Deutsch lesen.", "expected_answer": (False, "I would like to read this book in German.")},
#     {"post": "Você pode me ajudar com essa tarefa?", "expected_answer": (False, "Can you help me with this task?")},
#     {"post": "Mi piacerebbe imparare l'italiano durante le vacanze estive.", "expected_answer": (False, "I would like to learn Italian during the summer vacation.")},
#     {"post": "I need clear instructions to complete this project.", "expected_answer": (True, "I need clear instructions to complete this project.")},
#     {"post": "그는 내일 출장을 가야 합니다.", "expected_answer": (False, "He has to go on a business trip tomorrow.")},
#     {"post": "Jeg har boet i Danmark i ti år.", "expected_answer": (False, "I have lived in Denmark for ten years.")},
#     {"post": "Olen iloinen, että voin auttaa sinua tässä asiassa.", "expected_answer": (False, "I'm happy that I can help you with this matter.")},
#     {"post": "This is a straightforward English sentence.", "expected_answer": (True, "This is a straightforward English sentence.")},
#     {"post": "Could you please translate this to Spanish?", "expected_answer": (True, "Could you please translate this to Spanish?")},
#     {"post": "Yo necesito ayuda con mi tarea de español.", "expected_answer": (False, "I need help with my Spanish homework.")},
#     {"post": "Как дела?", "expected_answer": (False, "How are you?")},
#     {"post": "Il fait beau aujourd'hui.", "expected_answer": (False, "It's nice weather today.")},
#     {"post": "Thisisastringsmushedtogether", "expected_answer": (False, "Thisisastringsmushedtogether")},
#     {"post": "!@#$%^&*()_+", "expected_answer": (False, "!@#$%^&*()_+")},
#     {"post": "சரி, நான் உங்களுடன் இருக்கிறேன்.", "expected_answer": (False, "Okay, I'm with you.")},
#     {"post": "無駄な努力はしないでください。", "expected_answer": (False, "Don't waste your effort.")},
#     {"post": "Laordernoimportante.", "expected_answer": (False, "Theorderisnotimportant.")},
#     {"post": "9827362818273625235", "expected_answer": (False, "9827362818273625235")},
#     {"post": "https://example.com", "expected_answer": (False, "https://example.com")},
#     {"post": "<xml>This is not valid XML</xml>", "expected_answer": (False, "<xml>This is not valid XML</xml>")},
#     {"post": "null", "expected_answer": (False, "null")},
#     {"post": "undefined", "expected_answer": (False, "undefined")},
#     {"post": "Please let me know if you have any other questions.", "expected_answer": (True, "Please let me know if you have any other questions.")},
#     {"post": "I'm looking forward to our meeting next week.", "expected_answer": (True, "I'm looking forward to our meeting next week.")},
#     {"post": "Can you confirm the deadline for the project?", "expected_answer": (True, "Can you confirm the deadline for the project?")},
#     {"post": "I apologize for the delay in my response.", "expected_answer": (True, "I apologize for the delay in my response.")},
#     {"post": "Thank you for your patience and understanding.", "expected_answer": (True, "Thank you for your patience and understanding.")},
#     {"post": "J'espère que vous passez une bonne journée.", "expected_answer": (False, "I hope you're having a good day.")},
#     {"post": "¿Puedes ayudarme con esta tarea?", "expected_answer": (False, "Can you help me with this task?")},
#     {"post": "Ich habe das Buch noch nicht gelesen.", "expected_answer": (False, "I haven't read the book yet.")},
#     {"post": "Мне нравится путешествовать.", "expected_answer": (False, "I like to travel.")},
#     {"post": "Ho bisogno di un po' di tempo per riflettere.", "expected_answer": (False, "I need some time to think about it.")},
#     {"post": "これは難しい問題です。", "expected_answer": (False, "This is a difficult problem.")},
#     {"post": "Eu não entendi a sua pergunta.", "expected_answer": (False, "I didn't understand your question.")},
#     {"post": "Där är en vacker utsikt härifrån.", "expected_answer": (False, "There is a beautiful view from here.")},
#     {"post": "Jeg har glemt min pung.", "expected_answer": (False, "I forgot my wallet.")},
#     {"post": "!@#$%^&*()_+{}[]\\|;:'\",<.>/?", "expected_answer": (False, "!@#$%^&*()_+{}[]\\|;:'\",<.>/?")},
#     {"post": "ngngfbfbfngfngfngfngf", "expected_answer": (False, "ngngfbfbfngfngfngfngf")},
#     {"post": "https://www.example.com/path/to/resource?param=value", "expected_answer": (False, "https://www.example.com/path/to/resource?param=value")}]

# ################################################################################


def translate_content(post: str) -> tuple[bool, str]:
  return ("english" in get_language(post).lower(), get_translation(post))

# print(translate_content("Aquí está su primer ejemplo."))

# query_llm("Aquí está su primer ejemplo.")

# def eval_single_response_complete(expected_answer: tuple[bool, str], llm_response: tuple[bool, str]) -> float:
#   return (expected_answer[0] == llm_response[0]) * eval_single_response_translation(expected_answer[1], llm_response[1])

# # eval_score = evaluate(query_llm, eval_single_response_complete, complete_eval_set)

# # print(f"Evaluation Score: {eval_score}")


def query_llm_robust(post: str) -> tuple[bool, str]:
    language_output = get_language(post)
    translation_output = get_translation(post)

    # Check if language_output is a string
    if not isinstance(language_output, str):
        language_output = "Unknown language"

    # Check if translation_output is a string
    if not isinstance(translation_output, str):
        translation_output = "Error: Unable to translate"

    # Normalize language_output to lowercase
    language_output = language_output.lower()

    # Check if language_output contains "english"
    is_english = "english" in language_output

    return (is_english, translation_output)

# # """Run the tests."""

# # import ipytest
# # ipytest.run('-vv')

# # """Finally, we want to make sure that our modified function still works for normal inputs, so let's test it to find out!"""

# # query_llm_robust("Aquí está su primer ejemplo.")

# eval_score = evaluate(query_llm_robust, eval_single_response_complete, complete_eval_set)

# print(f"Evaluation Score: {eval_score}")