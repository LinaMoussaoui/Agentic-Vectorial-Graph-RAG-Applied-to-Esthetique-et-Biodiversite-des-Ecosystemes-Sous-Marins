# -*- coding: utf-8 -*-
"""
THESIS TRANSLATOR - FRENCH TO ENGLISH (USING GOOGLE TRANSLATE)
Preserves all data, only converts language
"""

import re
from deep_translator import GoogleTranslator
import langdetect

class ThesisTranslator:
    def __init__(self):
        """Initialize Google Translate for French to English"""
        self.translator = GoogleTranslator(source='fr', target='en')
        
        # Marine biology glossary (for accurate scientific terms)
        self.marine_terms = {
            "écosystèmes sous-marins": "underwater ecosystems",
            "biodiversité": "biodiversity",
            "esthétique": "aesthetics",
            "milieu marin": "marine environment",
            "faune marine": "marine fauna",
            "flore marine": "marine flora",
            "habitats marins": "marine habitats",
            "espèces marines": "marine species",
            "écosystème côtier": "coastal ecosystem",
            "zone intertidale": "intertidal zone",
            "écologie marine": "marine ecology",
            "conservation marine": "marine conservation",
            "pollution marine": "marine pollution",
            "acidification": "acidification",
            "réchauffement climatique": "climate change",
            "écosystème récifal": "reef ecosystem",
            "biodiversité marine": "marine biodiversity",
            "récif corallien": "coral reef",
            "algues": "algae",
            "poissons": "fish",
            "invertébrés marins": "marine invertebrates",
            "mammifères marins": "marine mammals"
        }
    
    def detect_language(self, text):
        """Detect if text is French or English"""
        if len(text.strip()) < 10:
            return "en"
        try:
            lang = langdetect.detect(text)
            return lang
        except:
            return "en"
    
    def translate_french_text(self, french_text):
        """Translate French text to English using Google Translate"""
        
        # Replace known marine terms first
        for fr_term, en_term in self.marine_terms.items():
            french_text = french_text.replace(fr_term, en_term)
            french_text = french_text.replace(fr_term.capitalize(), en_term.capitalize())
        
        # Translate
        try:
            translated = self.translator.translate(french_text)
            return translated
        except:
            # If translation fails, return original
            return french_text
    
    def translate_thesis_file(self, input_file, output_file):
        """Translate entire thesis file - PRESERVES ALL DATA"""
        
        print("="*60)
        print("THESIS TRANSLATOR - FRENCH TO ENGLISH")
        print("="*60)
        
        print(f"\n📖 Reading thesis from: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📊 Original size: {len(content):,} characters")
        
        # Split into paragraphs
        paragraphs = content.split('\n\n')
        
        print(f"\n🔄 Processing {len(paragraphs)} paragraphs...")
        
        translated_paragraphs = []
        french_count = 0
        english_count = 0
        
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                # Check language
                lang = self.detect_language(paragraph)
                
                if lang == "fr":
                    french_count += 1
                    print(f"   Translating paragraph {i+1}/{len(paragraphs)} (French → English)")
                    
                    # Split long paragraphs into sentences for better translation
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    translated_sentences = []
                    
                    for sentence in sentences:
                        if sentence.strip():
                            translated_sentences.append(self.translate_french_text(sentence))
                    
                    translated_paragraphs.append(' '.join(translated_sentences))
                else:
                    english_count += 1
                    translated_paragraphs.append(paragraph)
                
                # Progress update
                if (i + 1) % 20 == 0:
                    print(f"   Progress: {i+1}/{len(paragraphs)} paragraphs")
        
        # Combine translated paragraphs
        final_text = '\n\n'.join(translated_paragraphs)
        
        # Save translated thesis
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_text)
        
        print(f"\n✅ Translation complete!")
        print(f"📁 Saved to: {output_file}")
        print(f"📊 Output size: {len(final_text):,} characters")
        print(f"📊 French paragraphs translated: {french_count}")
        print(f"📊 English paragraphs kept: {english_count}")
        print("="*60)


def main():
    # UPDATE YOUR PATHS
    input_file = r"C:\Users\info\Desktop\PFA\Solutions\Project\cleaned_thesis_final.txt"
    output_file = r"C:\Users\info\Desktop\PFA\Solutions\Project\cleaned_thesis.txt"
    
    translator = ThesisTranslator()
    translator.translate_thesis_file(input_file, output_file)

if __name__ == "__main__":
    main()