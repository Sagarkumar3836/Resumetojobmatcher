# ResumetoJobMatcher
# Resume-to-Job Matcher

## 📌 Overview

The **Resume-to-Job Matcher** is a Streamlit-based application that compares a candidate's resume with a job description and calculates a similarity score. It uses NLP techniques and AI models to extract skills, analyze job requirements, and determine the best match.

## 🚀 Features

- 📄 **Extract text from resumes & job descriptions** (PDF, DOCX, TXT)
- 🤖 **Use AI models for Named Entity Recognition (NER)** to extract job-related terms
- 🔍 **Identify predefined technical skills** from both resume & job description
- 📊 **Compute a weighted similarity score** using:
  - **Cosine Similarity** (via Sentence Transformers)
  - **Skill Match Score**
- 🖥 **Interactive UI** built with Streamlit

## 🛠 Tech Stack

### **Frontend & UI**

- **Streamlit** → Interactive web-based UI

### **Natural Language Processing (NLP)**

- **spaCy** → Tokenization, lemmatization, stopword removal
- **NLTK** → Additional text preprocessing
- **Hugging Face Transformers** → Used for AI-based skill extraction
  - `dbmdz/bert-large-cased-finetuned-conll03-english` (Named Entity Recognition)
  - `all-MiniLM-L6-v2` (Sentence Embeddings)

### **Machine Learning & Similarity Computation**

- **Sentence Transformers** → Converts text into numerical embeddings
- **scikit-learn (********`cosine_similarity`********\*\*\*\*\*\*\*\*)** → Computes similarity between resume & job description

### **File Processing**

- **PyMuPDF (********`fitz`********\*\*\*\*\*\*\*\*)** → Extracts text from PDF files
- **python-docx (********`Document`********\*\*\*\*\*\*\*\*)** → Reads DOCX files

## 📥 Installation & Setup

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/yourusername/resume-job-matcher.git
cd resume-job-matcher
```

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Application**

```bash
streamlit run app.py
```

## 🎯 Usage

1. **Upload your resume** (PDF, DOCX, TXT)
2. **Upload a job description** (PDF, DOCX, TXT)
3. Click **Process** to see the similarity score and extracted skills.
4. View **matching skills** and missing job requirements.

## 🔮 Future Enhancements

- ✅ **Fuzzy skill matching** (to detect similar skills, e.g., "NLP" ≈ "Natural Language Processing")
- ⚡ **Parallelized text embeddings** (to improve processing speed)
- 📊 **Skill gap visualization** (bar charts & reports)
- 🌍 **API version using FastAPI** for better scalability

## 🤝 Contributing

Pull requests are welcome! Feel free to open an issue for any bug reports or feature suggestions.

---

👨‍💻 **Developed by Sagar Kumar**

I am a final year BTech student at National Institute Of Technology Raipur.

I am pursuing BTech Degree in Computer Science And Engineering Batch(2021 - 2025).
