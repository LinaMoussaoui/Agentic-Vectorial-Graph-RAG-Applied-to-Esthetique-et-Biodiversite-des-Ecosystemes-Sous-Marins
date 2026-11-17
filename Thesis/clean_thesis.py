# -*- coding: utf-8 -*-
"""
CLEANING THESIS - FINAL VERSION
REMOVES: Abstracts (keeps keywords), resumes, page numbers, figures, pictures, 
         REMERCIEMENTS, DEDICACE, phone numbers/HAL IDs, references, links, 
         bibliography, biography (entire sections removed)
PRESERVES: Table des matières, Tableaux, Liste des abréviations, ALL other headers, 
           ALL paragraphs, ALL thesis content, figure descriptions
"""

import fitz
import re
import os

PDF_PATH = r"C:\Users\info\Desktop\PFA\Reports\3.pdf"
OUTPUT_PATH = r"C:\Users\info\Desktop\PFA\Solutions\Project\cleaned_thesis_final.txt"

START_PAGE = 1   
END_PAGE = 156    

def extract_pages(pdf_path, start_page, end_page):
    """Extract text from specified page range"""
    doc = fitz.open(pdf_path)
    all_text = ""
    
    print(f"📖 Extracting pages {start_page} to {end_page}...")
    
    for page_num in range(start_page - 1, end_page):
        page = doc[page_num]
        text = page.get_text("text")
        all_text += text + "\n"
        print(f"   Page {page_num + 1}: {len(text)} chars")
    
    doc.close()
    return all_text

def clean_text(text):
    """
    Remove specific sections and patterns while preserving everything else
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    # Flag to track if we are inside a section to remove
    skip_section = False
    
    # Section headers that trigger removal of entire section (header + all content)
    remove_sections = [
        'abstract', 'résumé', 'resume',
        'remerciements', 'dedicace', 'dédicace',
        'bibliography', 'bibliographie', 'references', 'références',
        'works cited', 'ouvrages cités', 'biography', 'biographie'
    ]
    
    for line in lines:
        line_lower = line.lower().strip()
        original_line = line
        
        # ============================================
        # CHECK IF THIS LINE STARTS A SECTION TO REMOVE
        # ============================================
        is_section_header = False
        for section in remove_sections:
            if section in line_lower and len(line) < 100:
                is_section_header = True
                skip_section = True
                print(f"   🗑️ Removing entire section: {line.strip()[:50]}...")
                break
        
        # If we're in skip mode, check if we should continue skipping
        if skip_section:
            # Stop skipping when we hit a new major chapter/section header
            if any(header in line_lower for header in ['chapitre', 'chapter', 'introduction', 'conclusion', 'partie', '1.', '2.', '3.']):
                skip_section = False
                print(f"   ✅ End of removed section, preserving from: {line.strip()[:50]}")
            # Also stop if we hit an empty line followed by a clear header pattern
            elif len(line.strip()) == 0:
                continue
            else:
                continue  # Skip this line (it's part of removed section)
        
        # If it's a section header we remove, skip starting it
        if is_section_header:
            continue
        
        # ============================================
        # PRESERVE KEYWORDS (even if they appear near abstract)
        # ============================================
        if 'mots clés' in line_lower or 'mots-cles' in line_lower or 'keywords' in line_lower:
            cleaned_lines.append(line.strip())
            continue
        
        # ============================================
        # PATTERN REMOVAL ON LINES WE KEEP
        # ============================================
        line = original_line
        
        # Remove HAL IDs, phone numbers, identifiers
        line = re.sub(r'HAL\s+Id:\s*\S+', '', line, flags=re.IGNORECASE)
        line = re.sub(r'tel-\d+', '', line, flags=re.IGNORECASE)
        line = re.sub(r'hal-\d+', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\b\d{10}\b', '', line)
        line = re.sub(r'\b\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}\b', '', line)
        
        # Remove citations [1], [2,3], [1-5]
        line = re.sub(r'\[\d+(?:[-,;\s]\d+)*\]', '', line)
        
        # Remove (Author, year) citations
        line = re.sub(r'\([A-Za-zÀ-ÿ]+(?:[\s]+et\s+al\.)?[,;]\s*\d{4}[a-z]?\)', '', line)
        line = re.sub(r'[A-Za-zÀ-ÿ]+(?:\s+et\s+al\.)?\s+\(\d{4}[a-z]?\)', '', line)
        
        # Remove numeric citations like (12), (34, 56)
        line = re.sub(r'\(\s*\d+(?:[,\s]\d+)*\s*\)', '', line)
        
        # Remove URLs and hyperlinks
        line = re.sub(r'https?://\S+', '', line, flags=re.IGNORECASE)
        line = re.sub(r'www\.\S+', '', line, flags=re.IGNORECASE)
        line = re.sub(r'doi\.org/\S+', '', line, flags=re.IGNORECASE)
        
        # Remove page numbers (only if line is JUST a number)
        if re.match(r'^\s*\d+\s*$', line.strip()):
            continue
        if re.match(r'^\s*[ivxlcdm]+\s*$', line.strip().lower()):
            continue
        
        # Remove figure/picture references but KEEP descriptive text
        line = re.sub(r'Figure\s+\d+', '', line, flags=re.IGNORECASE)
        line = re.sub(r'Fig\.\s+\d+', '', line, flags=re.IGNORECASE)
        line = re.sub(r'image\s+\d+', '', line, flags=re.IGNORECASE)
        line = re.sub(r'photo\s+\d+', '', line, flags=re.IGNORECASE)
        
        # PRESERVE Tableau references (DO NOT REMOVE - for Neo4j graphs)
        # line = re.sub(r'Tableau\s+\d+', '', line)  # COMMENTED
        
        # Clean extra spaces
        line = re.sub(r'\s+', ' ', line)
        line = line.strip()
        
        # Keep all non-empty lines
        if line:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

# ============================================
# MAIN EXECUTION
# ============================================

print("="*70)
print("CLEANING THESIS - FINAL VERSION")
print("="*70)
print("\n✅ REMOVING (entire sections - header + content):")
print("   • Abstract, Résumé, Resume (EXCEPT keywords/mots-clés)")
print("   • Remerciements, Dedicace, Dédicace")
print("   • Bibliography, Bibliographie, References, Références")
print("   • Works cited, Ouvrages cités, Biography, Biographie")
print("\n✅ REMOVING (patterns only):")
print("   • Page numbers, HAL IDs, phone numbers")
print("   • Citations [1], (Author, year), numeric citations")
print("   • URLs, hyperlinks, DOIs")
print("   • Figure, Fig., image, photo references")
print("\n✅ PRESERVING:")
print("   ✓ Table des matières (TOC)")
print("   ✓ Listes des tableaux")
print("   ✓ Liste des abréviations")
print("   ✓ ALL other headers (Chapitre, Introduction, Conclusion, etc.)")
print("   ✓ ALL paragraphs and thesis content")
print("   ✓ Keywords / Mots-clés (for Graph Store)")
print("   ✓ Figure descriptions (text only)")
print("   ✓ 50% French / 50% English bilingual text")
print("="*70)

# Extract raw text
raw_text = extract_pages(PDF_PATH, START_PAGE, END_PAGE)
print(f"\n📊 Extracted: {len(raw_text):,} characters")

# Clean the text
cleaned_text = clean_text(raw_text)
print(f"🧹 Cleaned: {len(cleaned_text):,} characters")
print(f"🗑️ Removed: {len(raw_text) - len(cleaned_text):,} characters")
print(f"📄 Retention rate: {(len(cleaned_text)/len(raw_text)*100):.1f}%")

# Save to file
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write(cleaned_text)

print(f"\n✅ Saved to: {OUTPUT_PATH}")

# Preview
print("\n" + "="*70)
print("PREVIEW OF CLEANED TEXT (first 1500 characters):")
print("="*70)
print(cleaned_text[:1500])
print("="*70)
print(f"\n📊 Final cleaned text total: {len(cleaned_text):,} characters")
print("\n✅ Ready for Neo4j Graph Store!")