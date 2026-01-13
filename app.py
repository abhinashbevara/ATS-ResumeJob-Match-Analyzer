
import streamlit as st
import matplotlib.pyplot as plt
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# -------------------- Setup --------------------
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("stopwords")

st.set_page_config(page_title="ATS Resume Analyzer", layout="wide")
st.title("📄 ATS Resume–Job Match Analyzer")

# -------------------- Load BERT --------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------------------- Helper Functions --------------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def remove_stopwords(text):
    words = word_tokenize(text)
    return " ".join([w for w in words if w not in stopwords.words("english")])

def bert_similarity(resume, job):
    emb = model.encode([resume, job])
    return round(cosine_similarity([emb[0]], [emb[1]])[0][0] * 100, 2)

def get_common_and_missing(resume, job):
    resume_words = set(remove_stopwords(clean_text(resume)).split())
    job_words = set(remove_stopwords(clean_text(job)).split())
    return resume_words & job_words, job_words - resume_words

# -------------------- UI --------------------
uploaded_file = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("📝 Paste Job Description", height=200)

if st.button("Analyze Resume"):
    if uploaded_file is None:
        st.warning("Please upload your resume PDF")
    elif job_desc.strip() == "":
        st.warning("Please paste the job description")
    else:
        resume_text = extract_text_from_pdf(uploaded_file)

        similarity = bert_similarity(resume_text, job_desc)
        common, missing = get_common_and_missing(resume_text, job_desc)

        ats_score = round(0.6 * similarity + 0.4 * (len(common) / max(len(common) + len(missing), 1) * 100), 2)

        # -------------------- Results --------------------
        st.subheader("📊 Results")
        st.metric("BERT Match Score", f"{similarity}%")
        st.metric("ATS Compatibility Score", f"{ats_score}%")

        st.subheader("✅ Common Keywords")
        st.write(", ".join(list(common)) if common else "None")

        st.subheader("❌ Missing Keywords")
        st.write(", ".join(list(missing)) if missing else "No missing keywords 🎉")

        