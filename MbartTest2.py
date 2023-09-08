from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."
article_es = '''Esta noche es la Copa de Europa. Efectivamente, Eileen, esta noche es la Copa de Europa y es bastante 
importante para los españoles porque los españoles son muy aficionados al fútbol. Sí, es verdad. Así que es un partido importante, 
jugamos contra Portugal, estamos en mitad del primer tiempo'''
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# translate Hindi to French
tokenizer.src_lang = "hi_IN"
encoded_hi = tokenizer(article_hi, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_hi,
    forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
)
translation_hi_fr = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print("Translation Hindi to French:", translation_hi_fr)

# translate Spanish to English
tokenizer.src_lang = "es_XX"
encoded_ar = tokenizer(article_es, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_ar,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
translation_ar_en = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print("Translation Arabic to English:", translation_ar_en)
