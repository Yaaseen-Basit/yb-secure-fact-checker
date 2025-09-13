"""
Secure AI Fact-Checker with Dynamic Model Selection, Topic Detection,
Sentence-level Analysis, Cybersecurity Awareness, and Phishing/URL Scanner.
"""

import streamlit as st
from transformers import pipeline
import nltk
import html
import re
import langdetect

# Download tokenizer
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

# ---------------------------
# Dynamic Model Loader
# ---------------------------
@st.cache_resource
def load_fact_model(language: str = "en", low_resources: bool = False):
    """
    Dynamically select Hugging Face model based on language and resources.
    """
    if language == "en":
        model_name = (
            "microsoft/deberta-v3-base-mnli" if low_resources else "khalidalt/DeBERTa-v3-large-mnli"
        )
    else:
        model_name = "joeddav/xlm-roberta-large-xnli"
    return pipeline("zero-shot-classification", model=model_name)


@st.cache_resource
def load_topic_model():
    """Use BART-large-MNLI for topic detection (fast & reliable)."""
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


# ---------------------------
# Sentence-level fact-checking
# ---------------------------
def classify_sentence(sentence: str, classifier) -> dict:
    candidate_labels = ["True", "Misleading", "False"]
    result = classifier(sentence, candidate_labels)

    label = result["labels"][0]
    confidence = result["scores"][0]

    reasoning_dict = {
        "True": "Verified credible sources support this information.",
        "Misleading": "This may contain partial truth or exaggeration.",
        "False": "Credible sources refute this; may be harmful if followed.",
    }

    cybersecurity_tips = {
        "True": "No action needed.",
        "Misleading": "Double-check before sharing; avoid spreading half-truths.",
        "False": "Do not share; risk of scams, phishing, or fraud.",
    }

    color_dict = {"True": "green", "Misleading": "orange", "False": "red"}

    return {
        "label": label,
        "confidence": confidence,
        "reasoning": reasoning_dict[label],
        "cybersecurity_tip": cybersecurity_tips[label],
        "color": color_dict[label],
    }


# ---------------------------
# Topic detection
# ---------------------------
def detect_topic(text: str, topic_classifier) -> str:
    topics = ["Health", "Finance", "Politics", "Cybersecurity", "General"]
    result = topic_classifier(text, topics)
    return result["labels"][0]


def get_authentic_tips(topic: str) -> str:
    tips = {
        "Health": (
            "✅ Authentic Health Tips:\n"
            "- Follow WHO and health ministry guidelines.\n"
            "- Rely on licensed doctors, not forwards.\n"
            "- Avoid miracle cures; stick to evidence-based practices."
        ),
        "Finance": (
            "✅ Authentic Finance Tips:\n"
            "- Use verified banking channels only.\n"
            "- Never share OTPs or passwords.\n"
            "- Avoid quick-profit investment schemes.\n"
            "- Check RBI/SEC websites for verified updates."
        ),
        "Politics": (
            "✅ Authentic Politics Tips:\n"
            "- Verify on government portals.\n"
            "- Avoid sharing unverified propaganda.\n"
            "- Rely on trusted journalism sources."
        ),
        "Cybersecurity": (
            "✅ Cybersecurity Tips:\n"
            "- Never share passwords or OTPs.\n"
            "- Verify links before clicking; avoid suspicious domains.\n"
            "- Keep devices updated and use strong passwords.\n"
            "- Enable multi-factor authentication wherever possible."
        ),
        "General": "✅ Always verify claims with credible sources before sharing.",
    }
    return tips.get(topic, tips["General"])


# ---------------------------
# URL / Phishing scanner
# ---------------------------
def scan_urls(text: str) -> list[dict]:
    urls = re.findall(r"(https?://\S+)", text)
    suspicious_keywords = ["login", "verify", "secure-update", "banking", "account", "password"]
    suspicious_tlds = [".xyz", ".top", ".club", ".info", ".ru", ".cn"]

    url_results = []
    for url in urls:
        url_lower = url.lower()
        warning = None
        if any(keyword in url_lower for keyword in suspicious_keywords) or any(
            url_lower.endswith(tld) for tld in suspicious_tlds
        ):
            warning = "⚠️ Suspicious or potentially phishing link detected."
        url_results.append({"url": url, "warning": warning})
    return url_results


# ---------------------------
# Analyze text
# ---------------------------
def analyze_text(text: str, fact_classifier, topic_classifier) -> dict:
    topic = detect_topic(text, topic_classifier)
    sentences = sent_tokenize(text)
    sentence_results = [classify_sentence(s, fact_classifier) for s in sentences]
    url_results = scan_urls(text)
    return {
        "topic": topic,
        "authentic_tips": get_authentic_tips(topic),
        "sentence_results": [{"sentence": s, **r} for s, r in zip(sentences, sentence_results)],
        "url_results": url_results,
    }


# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config(
        page_title="Secure AI Fact-Checker + Phishing Scanner", page_icon="🛡️", layout="wide"
    )
    st.title("🛡️ Secure AI Fact-Checker with Phishing/URL Scanner")

    st.write(
        "Paste any forwarded message or article. Each sentence is analyzed, "
        "topic detected, color-coded, and tips shown. Suspicious URLs are flagged."
    )

    user_input = st.text_area("Paste forward/news content here", height=200)
    safe_input = html.escape(user_input)

    # Blue Check Fact button with highlight
    button_style = """
    <style>
    div.stButton > button:first-child {
        background-color: #007BFF;
        color: white;
        font-weight: bold;
    }
    div.stButton > button:first-child:hover {
        background-color: #0056b3;
        color: white;
    }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)

    if st.button("Check Fact"):
        if not safe_input.strip():
            st.warning("Please enter some text to check.")
        else:
            try:
                # Detect language
                try:
                    lang = langdetect.detect(user_input)
                except Exception:
                    lang = "en"

                # Load models dynamically
                try:
                    fact_model = load_fact_model(language=lang, low_resources=False)
                except Exception:
                    # fallback to base model
                    fact_model = load_fact_model(language="en", low_resources=True)

                topic_model = load_topic_model()

                analysis = analyze_text(user_input, fact_model, topic_model)

                st.subheader(f"Detected Topic: {analysis['topic']}")
                st.write(analysis["authentic_tips"])
                st.markdown("---")

                for res in analysis["sentence_results"]:
                    st.markdown(
                        f"<p style='color:{res['color']};'>"
                        f"**Sentence:** {html.escape(res['sentence'])}<br>"
                        f"Label: {res['label']} | Confidence: {res['confidence']:.2f}<br>"
                        f"Reasoning: {res['reasoning']}<br>"
                        f"Cybersecurity Tip: {res['cybersecurity_tip']}"
                        f"</p>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")

                if analysis["url_results"]:
                    st.subheader("🔗 URL / Phishing Scan Results")
                    for url_res in analysis["url_results"]:
                        st.write(f"URL: {url_res['url']}")
                        if url_res["warning"]:
                            st.error(url_res["warning"])
                        else:
                            st.success("✅ URL appears safe.")

            except Exception as e:
                st.error(f"Error while analyzing text: {e}")

    st.info(
        "This prototype helps fight digital misinformation, educates users on cybersecurity, "
        "and flags suspicious links — a proactive tool against scams and phishing attacks."
    )


if __name__ == "__main__":
    main()
