import docx
import sys

def get_structure(file_path):
    doc = docx.Document(file_path)
    with open("structure.txt", "w", encoding="utf-8") as f:
        for p in doc.paragraphs:
            if p.style.name.startswith("Heading") or p.style.name == "Title" or p.style.name == "Subtitle":
                f.write(f"[{p.style.name}] {p.text.strip()}\n")

if __name__ == "__main__":
    get_structure(sys.argv[1])
