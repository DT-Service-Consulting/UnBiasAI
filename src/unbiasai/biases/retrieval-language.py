def detect_language(content):
    """
    Detects the language of a given text content.

    Args:
        content (str): The text to analyze.

    Returns:
        str: Detected language code (e.g., 'en', 'fr', 'de', 'nl'), or None if detection fails.
    """
    try:
        return detect(content)
    except Exception:
        return None
