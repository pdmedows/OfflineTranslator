from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the T5 model and tokenizer
model_name = "t5-large"  # You can use other T5 variants like "t5-base" or "t5-large"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def translate_text(input_text, target_language="translate English to French:"):
    # Preprocess input text by specifying the translation task and target language
    input_text = f"{target_language} {input_text}"
    
    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate translation
    translation_ids = model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)

    # Decode and return the translated text
    translated_text = tokenizer.decode(translation_ids[0], skip_special_tokens=True)
    return translated_text

# Example usage:
input_text = "Hello, how are you?"  # English text (or any other language)
target_language = "translate English to French:"  # Translation task and target language
translated_text = translate_text(input_text, target_language)
print(f"Translated text: {translated_text}")
