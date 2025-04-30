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


