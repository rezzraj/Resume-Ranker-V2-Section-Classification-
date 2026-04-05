import os
import re
import json
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import fitz
from text_preprocess import chunking_and_preprocessing, section_classifier, clean_jd, jd_embedding
from comparison import compare_JDandResume
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from main import forward_pass


model = SentenceTransformer('all-mpnet-base-v2')

def extract_text(file):
    #text_dict = {}
    #file_name = file.name
    read_file=file.read()

    doc = fitz.open(stream=read_file, filetype="pdf")

    full_text = ""

    for page in doc:
        page_text = page.get_text().strip()
        if page_text:
            full_text += page_text + "\n"
        else:
            pixel_map = page.get_pixmap()
            #this converts that doc thing to PIL image
            img = Image.frombytes("RGB", [pixel_map.width, pixel_map.height], pixel_map.samples)

            ocr_text = pytesseract.image_to_string(img).strip()
            full_text += ocr_text + "\n"

    #text_dict[file_name] = full_text
    doc.close()
    return full_text




folder_path = "resumes (small)"


jd="""**Position:** NLP / Machine Learning Intern (Applied AI)

**Location:** Remote / Hybrid

---

### **About the Role**

We are looking for a passionate Machine Learning / NLP Intern to work on real-world AI systems involving natural language processing, deep learning, and model deployment. The ideal candidate has hands-on experience building end-to-end ML pipelines and deploying models using modern frameworks.

---

### **Key Responsibilities**

* Develop and implement **NLP and deep learning models** for real-world applications
* Build **semantic similarity and ranking systems** using transformer-based architectures (e.g., BERT, Sentence-BERT)
* Design complete ML pipelines including **text preprocessing, embedding generation, and similarity scoring**
* Work on **resume/job matching systems or recommendation engines**
* Implement and optimize models using **PyTorch and deep learning frameworks**
* Develop and deploy backend APIs using **FastAPI**
* Work with **OCR and document parsing tools** to extract structured data from PDFs and images
* Perform model evaluation using statistical techniques such as **correlation metrics**
* Collaborate on deploying ML systems using **Docker and cloud-based tools**

---

### **Required Skills**

* Strong proficiency in **Python**
* Experience with **PyTorch / TensorFlow**
* Knowledge of **NLP, Transformers, BERT, Sentence-BERT**
* Experience with **text embeddings and cosine similarity**
* Familiarity with **FastAPI and backend development**
* Experience with tools like **pytesseract, PyMuPDF, pdf2image**
* Understanding of **machine learning pipelines and model evaluation**

---

### **Nice to Have**

* Experience with **hyperparameter tuning (Optuna)**
* Knowledge of **CNNs and transfer learning (e.g., MobileNetV2)**
* Familiarity with **audio models (e.g., Wave2Vec2)**
* Experience deploying apps using **Streamlit or Docker**
* Exposure to **real-world datasets and data augmentation techniques**

---

### **Projects / Experience**

* Hands-on experience building **end-to-end ML/NLP systems**
* Experience developing **semantic ranking or recommendation systems**
* Exposure to deploying ML applications in production or demo environments

---

### **Duration**

* 3–6 months internship



"""




bad_files = []
good_data = []

for i ,file_name in enumerate(os.listdir(folder_path)):
    print(f"processing {i}th file")
    if file_name.endswith(".pdf"):
        file_path = os.path.join(folder_path, file_name)

        try:
            with open(file_path, "rb") as file:
                text = extract_text(file)
                good_data.append(text)

        except Exception:
            bad_files.append(file_name)

resume_text=good_data[0]

jd_text= clean_jd(jd)




print(forward_pass(resume_text,jd_text))



data=chunking_and_preprocessing(resume_text)
data=section_classifier(data,jd_text)
print(data)