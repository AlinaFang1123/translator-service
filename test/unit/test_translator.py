from src.translator import translate_content
import bigframes.dataframe
import vertexai
from mock import patch
from unittest.mock import MagicMock
import time

@patch('vertexai.preview.language_models._PreviewChatSession.send_message')
def test_unexpected_language(mocker):
    mocker.return_value.text = "I don't understand your request"

    assert translate_content("Aquí está su primer ejemplo.") == (True, 'Aquí está su primer ejemplo.')

@patch('vertexai.preview.language_models._PreviewChatSession.send_message')
def test_malformed_response(mocker):
    mocker.return_value.text = '{"text": "This is a malformed response"'

    assert translate_content("Aquí está su primer ejemplo.") == (True, 'Aquí está su primer ejemplo.')

@patch('vertexai.preview.language_models._PreviewChatSession.send_message')
def test_timeout(mocker):
    mocker.return_value.text = MagicMock(side_effect=lambda: time.sleep(60) or "This is a slow response")

    assert translate_content("Aquí está su primer ejemplo.") == (True, 'Aquí está su primer ejemplo.')

@patch('vertexai.preview.language_models._PreviewChatSession.send_message')
def test_empty_response(mocker):
    mocker.return_value.text = {}

    assert translate_content("Aquí está su primer ejemplo.") == (True, 'Aquí está su primer ejemplo.')