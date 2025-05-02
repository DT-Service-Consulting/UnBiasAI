import pytest
from unittest.mock import patch, MagicMock
from src.unbiasai.utils import generate_embedding

@patch('src.unbiasai.utils.OpenAIEmbeddings')
def test_generate_embedding(mock_openai_embeddings):
    # Mock the embed_query method
    mock_instance = MagicMock()
    mock_instance.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_openai_embeddings.return_value = mock_instance

    # Test data
    text = "Test input text"
    api_key = "test-api-key"

    # Call the function
    result = generate_embedding(text, api_key)

    # Assertions
    mock_openai_embeddings.assert_called_once_with(openai_api_key=api_key, model="text-embedding-3-large")
    mock_instance.embed_query.assert_called_once_with(text)
    assert result == [0.1, 0.2, 0.3]


@patch('src.unbiasai.utils.ChatOpenAI')
def test_initialize_llm_gpt(mock_chat_openai):
    api_key = "test-api-key"
    initialize_llm("gpt", api_key)
    mock_chat_openai.assert_called_once_with(model_name="gpt-4o-2024-11-20", openai_api_key=api_key)

@patch('src.unbiasai.utils.ChatAnthropic')
def test_initialize_llm_claude(mock_chat_anthropic):
    api_key = "test-api-key"
    initialize_llm("claude", api_key)
    mock_chat_anthropic.assert_called_once_with(model="claude-3-7-sonnet-latest", anthropic_api_key=api_key)

@patch('src.unbiasai.utils.ChatMistralAI')
def test_initialize_llm_mistral(mock_chat_mistral):
    api_key = "test-api-key"
    initialize_llm("mistral", api_key)
    mock_chat_mistral.assert_called_once_with(model="mistral-large-latest", mistral_api_key=api_key)

@patch('src.unbiasai.utils.ChatCohere')
def test_initialize_llm_cohere(mock_chat_cohere):
    api_key = "test-api-key"
    initialize_llm("cohere", api_key)
    mock_chat_cohere.assert_called_once_with(model="command-a-03-2025", cohere_api_key=api_key)
