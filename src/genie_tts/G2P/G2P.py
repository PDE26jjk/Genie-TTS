from typing import List


def text_to_phones(text: str, language: str) -> List[int]:
    if language == 'Japanese':
        from .Japanese.JapaneseG2P import japanese_to_phones
        return japanese_to_phones(text)
    elif language == 'English':
        from .English.EnglishG2P import english_to_phones
        return english_to_phones(text)
    else:
        from .Chinese.ChineseG2P import chinese_to_phones
        normalized_text, phones, phones_ids, word2ph = chinese_to_phones(text)
        return phones_ids
