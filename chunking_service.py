# ==========================================================
# CHUNKING SERVICE
#
# Semantic Chunking Redesign (Single-Pass Section Builder)
# - Groups OCR text into semantic sections based on meaning.
# - Preserves strong semantic bonds (headings, lists, tables).
# - Single-pass architecture, completely stateless.
# - No post-processing merges required.
# ==========================================================

import re
from typing import List, Dict, Any

CHARS_PER_TOKEN = 4

def get_token_count(text: str) -> int:
    """
    Estimates the number of tokens in a string.
    Uses Hugging Face tokenizer if available, otherwise falls back to a character-based heuristic.
    """
    if not hasattr(get_token_count, "tokenizer_initialized"):
        get_token_count.tokenizer_initialized = True
        get_token_count.tokenizer = None
        try:
            from transformers import AutoTokenizer
            get_token_count.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            pass

    if get_token_count.tokenizer is not None:
        try:
            return len(get_token_count.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass

    return len(text) // CHARS_PER_TOKEN

def is_table_row(line: str) -> bool:
    """Helper to detect markdown table rows, including malformed OCR outputs."""
    line = line.strip()
    if not line:
        return False
    if re.match(r'^[:\-\s\|]+$', line) and '|' in line:
        return True
    if line.startswith('|') or line.endswith('|'):
        return True
    if line.count('|') >= 2:
        return True
    return False

def extract_blocks(text: str) -> List[Dict[str, str]]:
    """
    Parses the Markdown text into fine-grained structural blocks.
    These blocks act as the foundation for the Semantic Section Builder.
    """
    lines = text.split('\n')
    blocks = []
    current_block_lines = []
    current_type = None

    def add_block():
        nonlocal current_block_lines, current_type
        if current_block_lines:
            content = '\n'.join(current_block_lines).strip()
            if content:
                blocks.append({"type": current_type, "text": content})
            current_block_lines = []
            current_type = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            add_block()
            i += 1
            continue

        # HTML Table
        if stripped.lower().startswith("<table"):
            add_block()
            current_type = "table"
            current_block_lines.append(line)
            if "</table>" not in stripped.lower():
                i += 1
                while i < len(lines):
                    current_block_lines.append(lines[i])
                    if "</table>" in lines[i].lower():
                        break
                    i += 1
            add_block()
            i += 1
            continue

        # Markdown Table
        if is_table_row(stripped):
            is_valid_table = False
            lookahead_i = i + 1
            while lookahead_i < len(lines):
                la_stripped = lines[lookahead_i].strip()
                if not la_stripped:
                    lookahead_i += 1
                    continue
                if is_table_row(la_stripped):
                    is_valid_table = True
                break
                
            if is_valid_table:
                add_block()
                current_type = "table"
                while i < len(lines):
                    line_str = lines[i]
                    l_stripped = line_str.strip()
                    if not l_stripped:
                        if i + 1 < len(lines) and is_table_row(lines[i+1].strip()):
                            current_block_lines.append(line_str)
                            i += 1
                            continue
                        break
                    elif is_table_row(l_stripped):
                        current_block_lines.append(line_str)
                        i += 1
                    else:
                        break
                add_block()
                continue

        # Headings
        if re.match(r'^#{1,6}\s+', stripped):
            add_block()
            current_type = "heading"
            current_block_lines.append(line)
            add_block()
            i += 1
            continue

        # Lists
        if re.match(r'^(\*|-|\+|\d+\.)\s+', stripped):
            if current_type != "list":
                add_block()
                current_type = "list"
            current_block_lines.append(line)
            i += 1
            continue
            
        if current_type == "list":
            current_block_lines.append(line)
            i += 1
            continue

        # Regular Paragraph
        if current_type != "paragraph":
            add_block()
            current_type = "paragraph"
        current_block_lines.append(line)
        i += 1
        
    add_block()
    return blocks

def split_large_table(table_text: str, max_tokens: int) -> List[str]:
    """Splits an oversized table by rows while repeating the header."""
    if get_token_count(table_text) <= max_tokens:
        return [table_text]
        
    if table_text.startswith("|"):
        lines = table_text.split("\n")
        if len(lines) < 4:
            return [table_text]
            
        header, separator = lines[0], lines[1]
        chunks, current_chunk = [], [header, separator]
        current_tokens = get_token_count(header + "\n" + separator)
        
        for row in lines[2:]:
            row_tokens = get_token_count(row)
            if current_tokens + row_tokens > max_tokens and len(current_chunk) > 2:
                chunks.append("\n".join(current_chunk))
                current_chunk = [header, separator, row]
                current_tokens = get_token_count(header + "\n" + separator + "\n" + row)
            else:
                current_chunk.append(row)
                current_tokens += row_tokens
                
        if len(current_chunk) > 2:
            chunks.append("\n".join(current_chunk))
            
        return chunks
        
    if "<table" in table_text and "</table>" in table_text:
        header_match = re.search(r'(<table.*?>.*?</tr>)', table_text, re.IGNORECASE | re.DOTALL)
        if not header_match:
            return [table_text]
            
        header_html = header_match.group(1)
        body_html = table_text[header_match.end():]
        rows = re.findall(r'<tr.*?>.*?</tr>', body_html, re.IGNORECASE | re.DOTALL)
        
        if not rows:
            return [table_text]
            
        chunks, current_rows = [], []
        current_tokens = get_token_count(header_html)
        
        for row in rows:
            row_tokens = get_token_count(row)
            if current_tokens + row_tokens > max_tokens and current_rows:
                chunks.append(header_html + "\n" + "\n".join(current_rows) + "\n</table>")
                current_rows = [row]
                current_tokens = get_token_count(header_html + "\n" + row)
            else:
                current_rows.append(row)
                current_tokens += row_tokens
                
        if current_rows:
            chunks.append(header_html + "\n" + "\n".join(current_rows) + "\n</table>")
            
        return chunks
        
    return [table_text]

def chunk_paragraph_text(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """Chunks oversized paragraph texts with overlapping boundaries."""
    if get_token_count(text) <= max_tokens:
        return [text]
        
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    chunks = []
    current_chunk_sentences = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = get_token_count(sentence + " ")
        
        if sentence_tokens > max_tokens:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                overlap_sentences = []
                overlap_count = 0
                for s in reversed(current_chunk_sentences):
                    s_tok = get_token_count(s + " ")
                    if overlap_count + s_tok > overlap_tokens:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_count += s_tok
                current_chunk_words = " ".join(overlap_sentences).split()
                current_tokens = sum(get_token_count(w + " ") for w in current_chunk_words)
            else:
                current_chunk_words = []
                current_tokens = 0
                
            words = sentence.split()
            for word in words:
                word_tokens = get_token_count(word + " ")
                if current_tokens + word_tokens > max_tokens and current_chunk_words:
                    chunks.append(" ".join(current_chunk_words))
                    overlap_words = []
                    overlap_count = 0
                    for w in reversed(current_chunk_words):
                        w_tok = get_token_count(w + " ")
                        if overlap_count + w_tok > overlap_tokens:
                            break
                        overlap_words.insert(0, w)
                        overlap_count += w_tok
                    current_chunk_words = overlap_words
                    current_tokens = overlap_count
                current_chunk_words.append(word)
                current_tokens += word_tokens
                
            if current_chunk_words:
                current_chunk_sentences = [" ".join(current_chunk_words)]
            else:
                current_chunk_sentences = []
                current_tokens = 0
            continue

        if current_tokens + sentence_tokens > max_tokens and current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_chunk_sentences):
                s_tok = get_token_count(s + " ")
                if overlap_count + s_tok > overlap_tokens:
                    break
                overlap_sentences.insert(0, s)
                overlap_count += s_tok
            current_chunk_sentences = overlap_sentences
            current_tokens = overlap_count
            
        current_chunk_sentences.append(sentence)
        current_tokens += sentence_tokens
        
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
        
    return chunks

def chunk_page(page_text: str, page_number: int, max_tokens: int = 1500, overlap_tokens: int = 150, document_id: str = "document", document_name: str = "Unknown") -> List[Dict[str, Any]]:
    """
    Main entry point for production semantic chunking.
    Implements a single-pass Section Builder algorithm.
    """
    TARGET_MIN = 700
    TARGET_IDEAL = 1000

    blocks = extract_blocks(page_text)
    chunks = []
    chunk_index = 1
    
    current_blocks = []
    current_tokens = 0
    running_heading = None
    running_heading_level = None

    def emit_section():
        nonlocal chunk_index
        text = "\n\n".join(b["text"] for b in current_blocks).strip()
        if not text:
            return
            
        tokens = get_token_count(text)
        
        section_heading = running_heading
        section_heading_level = running_heading_level
        for b in current_blocks:
            if b["type"] == "heading":
                m = re.match(r'^(#{1,6})\s+(.*)$', b["text"])
                if m:
                    section_heading = m.group(2).strip()
                    section_heading_level = len(m.group(1))
                    break 
                    
        c_type = "paragraph"
        if len(current_blocks) == 1 and current_blocks[0]["type"] == "table":
            c_type = "table"
        elif any(b["type"] == "table" for b in current_blocks):
            c_type = "table"
            
        if tokens > max_tokens:
            if c_type == "table":
                sub_chunks = split_large_table(text, max_tokens)
                for sc in sub_chunks:
                    chunks.append({
                        "document_id": document_id,
                        "document_name": document_name,
                        "page": page_number,
                        "chunk": chunk_index,
                        "chunk_id": f"{document_id}_p{page_number:03d}_c{chunk_index:03d}",
                        "type": "table",
                        "heading": section_heading,
                        "heading_level": section_heading_level,
                        "token_count": get_token_count(sc),
                        "text": sc
                    })
                    chunk_index += 1
            else:
                sub_chunks = chunk_paragraph_text(text, max_tokens, overlap_tokens)
                for sc in sub_chunks:
                    chunks.append({
                        "document_id": document_id,
                        "document_name": document_name,
                        "page": page_number,
                        "chunk": chunk_index,
                        "chunk_id": f"{document_id}_p{page_number:03d}_c{chunk_index:03d}",
                        "type": "paragraph",
                        "heading": section_heading,
                        "heading_level": section_heading_level,
                        "token_count": get_token_count(sc),
                        "text": sc
                    })
                    chunk_index += 1
        else:
            chunks.append({
                "document_id": document_id,
                "document_name": document_name,
                "page": page_number,
                "chunk": chunk_index,
                "chunk_id": f"{document_id}_p{page_number:03d}_c{chunk_index:03d}",
                "type": c_type,
                "heading": section_heading,
                "heading_level": section_heading_level,
                "token_count": tokens,
                "text": text
            })
            chunk_index += 1

    for block in blocks:
        b_type = block["type"]
        b_text = block["text"]
        b_tokens = get_token_count(b_text)
        
        if b_type == "heading":
            m = re.match(r'^(#{1,6})\s+(.*)$', b_text)
            if m:
                running_heading_level = len(m.group(1))
                running_heading = m.group(2).strip()
                
        if not current_blocks:
            current_blocks.append(block)
            current_tokens += b_tokens
            continue
            
        prev_type = current_blocks[-1]["type"]
        prev_text = current_blocks[-1]["text"].strip()
        
        strong_bond = False
        if prev_type == "heading":
            strong_bond = True
        elif prev_type == "paragraph" and prev_text.endswith(":"):
            strong_bond = True
            
        should_split = False
        if not strong_bond:
            is_major_heading = (b_type == "heading" and b_text.startswith("# "))
            
            if current_tokens >= TARGET_MIN and b_type == "heading":
                should_split = True
            elif is_major_heading:
                should_split = True
            elif current_tokens >= TARGET_IDEAL:
                should_split = True
            elif current_tokens + b_tokens > max_tokens:
                should_split = True
                
        if should_split:
            emit_section()
            current_blocks = [block]
            current_tokens = b_tokens
        else:
            current_blocks.append(block)
            current_tokens += b_tokens
            
    if current_blocks:
        emit_section()
        
    return chunks
