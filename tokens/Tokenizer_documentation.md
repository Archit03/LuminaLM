# LuminaLM Tokenizer Documentation

## Overview
This document outlines the tokenization process for **LuminaLM**, including the datasets used, tokenization methodology, special tokens configuration, and performance validation. The tokenizer has been designed specifically for medical-domain language modeling, integrating multilingual and domain-specific features to address the requirements of advanced medical applications.

---

## Objectives
1. **Domain-Specific Tokenization:** Build a tokenizer optimized for medical texts, with support for abbreviations, composite units, and multilingual contexts.
2. **Scalability:** Process large datasets efficiently and incorporate diverse data sources.
3. **Custom Special Tokens:** Integrate tokens tailored to medical terminology and multilingual text processing.
4. **Validation and Testing:** Ensure the tokenizer aligns with the requirements of medical summarization, diagnostics generation, and related tasks.

---

## Features
1. **Multilingual Support:** Incorporates special tokens like `<|en|>`, `<|fr|>`, etc., to handle multilingual datasets.
2. **Custom Medical Tokens:** Over 100 special tokens specific to the medical domain, including:
   - `<SYM>`, `<DIAG>`, `<TREAT>`, `<MED>`
   - `<MRI>`, `<CT>`, `<BP>`, `<GLUCOSE>`
3. **Preprocessing Enhancements:**
   - Normalization of composite units (e.g., `120 mmHg`, `37.0°C`, `500 mg`).
   - Handling of common abbreviations (e.g., `b.i.d.`, `q.d.`).
4. **Performance Optimization:** Batch processing for efficient tokenization and training.
5. **Validation Framework:** Comprehensive tests to ensure the tokenizer performs accurately on real-world medical texts.

---

## Tokenization Process

### 1. **Dataset Preparation**
We loaded multiple datasets to capture domain-specific language and multilingual contexts such as:
- `rungalileo/medical_transcription_40`: General medical transcription dataset.
- `qanastek/ELRC-Medical-V2`: Multilingual medical dataset with translations in English, Bulgarian, and Czech.

### 2. **Preprocessing**
The following preprocessing steps were applied to normalize the text:
- **Composite Units:** Merged units like `mmHg` and `°C` to single tokens.
- **Abbreviations:** Expanded common medical abbreviations (e.g., `b.i.d.` to `twice daily`).
- **Whitespace Normalization:** Standardized spacing and removed extraneous characters.

### 3. **Tokenizer Training**
The tokenizer was trained using the Byte Pair Encoding (BPE) algorithm with:
- Vocabulary size: **50,000 tokens**
- Special tokens: Predefined tokens for medical terminology and multilingual support.

Example Special Tokens:
| Token       | Purpose                             |
|-------------|-------------------------------------|
| `<|en|>`    | English language marker             |
| `<|fr|>`    | French language marker              |
| `<SYM>`     | General medical symptoms marker     |
| `<DIAG>`    | Diagnostic information marker       |
| `<MRI>`     | Medical imaging token (MRI scan)    |
| `<BP>`      | Blood pressure-related information  |

---

## Validation and Testing

### 1. **Performance Testing**
- Tested tokenization on real-world medical sentences.
- Example Input: `BP: 120/80 mmHg, HR: 72 bpm, Temp: 37.0°C.`
- Output:
  - **Tokenized Text:** `['<s>', '<|en|>', 'BP', ':', '120', '/', '80', 'mmHg', ',', 'HR', ':', '72', 'bpm', ',', 'Temp', ':', '37.0°C', '.', '</s>']`
  - **Token IDs:** `[2, 316, 7473, 149, 6515, 138, 1872, 4574, 135, 10115, 149, 4660, 34116, 135, 27152, 149, 4008, 137, 3]`

### 2. **Special Tokens Validation**
All special tokens were verified to be present in the tokenizer's vocabulary. Missing tokens were automatically flagged during validation.

### 3. **Batch Processing**
- Implemented batch tokenization to improve efficiency.
- Average batch processing time: **0.00s**

### 4. **Error Reporting**
- Errors in tokenization or dataset processing were logged with detailed error messages.
- Example: Missing tokens (`<|en|>`) were identified and resolved during validation.

---

## Key Metrics
1. **Vocabulary Size:** 50,000 tokens
2. **Total Datasets Processed:** 2
3. **Special Tokens:** 100+ tokens tailored for medical use
4. **Batch Processing Performance:** 0.00s (average batch time)

---

## Challenges Addressed
1. **Missing Tokens:** Resolved issues with missing `<|en|>` during initial validation.
2. **Composite Units:** Improved handling of units like `mmHg`, ensuring accurate tokenization.
3. **Multilingual Contexts:** Added tokens to support language-specific markers.

---

## Acknowledgments
- Hugging Face for dataset hosting and API integration.
- LuminaLM team for collaborative efforts on tokenizer design and testing.

For further information, refer to the `tokenizer_validation.log` and `tokenizer_training.log` files.