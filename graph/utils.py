from langdetect import detect, LangDetectException

def detect_language(text: str, number_of_characters: int = 1000) -> str:
    """
    Detects the language of a text based on a sample of its characters.

    Args:
        text (str): The input text to analyze for language detection.
        number_of_characters (int, optional): The number of characters from the beginning
            of the text to use for detection. Defaults to 1000.

    Returns:
        str: A language code (e.g., 'en' for English, 'fr' for French) or 'unknown'
            if the language cannot be detected or if the text is empty.
    """
    text_snippet = text[:number_of_characters] if len(text) > number_of_characters else text

    if not text_snippet.strip():
        # Handle the case where the text snippet is empty or only contains whitespace
        return 'unknown'
    try:
        return detect(text_snippet)
    except LangDetectException as e:
        if 'No features in text' in str(e):
            # Handle the specific error where no features are found in the text
            return 'unknown'