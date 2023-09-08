from transformers import MBartForConditionalGeneration, MBartTokenizer

# Load the mBART-50 model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"  # Multilingual model
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBartTokenizer.from_pretrained(model_name)

def translate_text(input_text, source_language="fr", target_language="es_XX"):
    # Preprocess input text with the source and target language prefixes
    input_text = f"{source_language}: {target_language}: {input_text}"
    
    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate translation
    translation_ids = model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True, forced_bos_token_id=101)

    # Decode and return the translated text
    translated_text = tokenizer.decode(translation_ids[0], skip_special_tokens=True)
    return translated_text

# Example usage:
input_text = "Bonjour, comment ça va ?"  # French text (or any other language)
source_language = "fr"  # Source language code (e.g., "fr" for French)
target_language = "es_XX"  # Target language code (e.g., "es_XX" for Spanish)
translated_text = translate_text(input_text, source_language, target_language)
print(f"Translated text: {translated_text}")
