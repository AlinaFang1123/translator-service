from src.translator import translate_content
import bigframes.dataframe
import vertexai
from mock import patch
from unittest.mock import MagicMock
import time

@patch('vertexai.preview.language_models._PreviewChatSession.send_message')
def test_unexpected_language(mocker):
    # we mock the model's response to return a random message
    mocker.return_value.text = "I don't understand your request"

    # Assert that the function handles the unexpected response gracefully
    assert translate_content("Aquí está su primer ejemplo.") == "I don't understand your request"

@patch('vertexai.preview.language_models._PreviewChatSession.send_message')
def test_malformed_response(mocker):
    # we mock the model's response to return a malformed JSON object
    mocker.return_value.text = '{"text": "This is a malformed response"'

    # Assert that the function handles the malformed response gracefully
    assert translate_content("Aquí está su primer ejemplo.") == "Error: Malformed response from language model"

@patch('vertexai.preview.language_models._PreviewChatSession.send_message')
def test_timeout(mocker):
    # we mock the send_message method to take a long time to respond
    mocker.return_value.text = MagicMock(side_effect=lambda: time.sleep(60) or "This is a slow response")

    # Assert that the function handles the timeout gracefully
    assert translate_content("Aquí está su primer ejemplo.") == "Error: Timeout while waiting for language model response"

@patch('vertexai.preview.language_models._PreviewChatSession.send_message')
def test_service_unavailable(mocker):
    # we mock the send_message method to raise an exception
    mocker.return_value.text = MagicMock(side_effect=vertexai.errors.ServiceUnavailableError("Service is unavailable"))

    # Assert that the function handles the service unavailable error gracefully
    assert translate_content("Aquí está su primer ejemplo.") == "Error: Service is unavailable"