import docx

def read_docx(file_path):
    doc = docx.Document(file_path)
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip():
            print(f"[{p.style.name}] {p.text.strip()}")
            
if __name__ == "__main__":
    import sys
    read_docx(sys.argv[1])
