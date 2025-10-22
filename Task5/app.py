import streamlit as st
import nbformat
from nbconvert import PythonExporter

# --- تحميل كود الـ Notebook ---
with open("task5.ipynb", "r", encoding="utf-8") as f:
    notebook_content = f.read()

exporter = PythonExporter()
source, _ = exporter.from_notebook_node(nbformat.reads(notebook_content, as_version=4))

# 🧹 تنظيف الكود من أوامر Jupyter
clean_source = "\n".join(
    line for line in source.splitlines()
    if not line.strip().startswith("get_ipython") and not line.strip().startswith("%")
)

# --- كاش لتسريع تحميل التطبيق ---
@st.cache_resource
def load_app():
    exec(clean_source, globals())
    return app


app = load_app()


exec(clean_source, globals())

# --- إعداد الصفحة ---
st.set_page_config(page_title="LangGraph Assistant", page_icon="🤖", layout="wide")

# --- CSS مخصص لتصميم الواجهة ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #141e30, #243b55);
        color: white;
        font-family: 'Poppins', sans-serif;
    }
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #00e6e6;
        text-shadow: 0 0 15px #00e6e6;
        margin-top: 10px;
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 0 0 10px #00e6e6; }
        to { text-shadow: 0 0 25px #00ffff, 0 0 50px #00e6e6; }
    }
    .subtext {
        text-align: center;
        color: #cfd8dc;
        margin-bottom: 25px;
        font-size: 1rem;
    }
    .stTextArea textarea {
        background-color: #1e293b !important;
        color: #e0f7fa !important;
        border-radius: 10px !important;
        border: 1px solid #00e6e6 !important;
        font-size: 1rem !important;
    }
    div.stButton > button {
        background-color: #00e6e6;
        color: black;
        border: none;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        transition: 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #00ffff;
        transform: scale(1.07);
    }
    .answer-box {
        background-color: #0f172a;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #00e6e6;
        margin-top: 20px;
        box-shadow: 0 0 20px rgba(0, 230, 230, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- JS تأثير كتابة ---
st.markdown("""
    <script>
    const title = "🧠 LangGraph AI Assistant";
    let i = 0;
    function typeWriter() {
        if (i < title.length) {
            document.getElementById("typeTitle").innerHTML += title.charAt(i);
            i++;
            setTimeout(typeWriter, 80);
        }
    }
    window.addEventListener('load', typeWriter);
    </script>
    <h1 id="typeTitle" class="title"></h1>
    <p class="subtext">Your smart Python code assistant powered by LangGraph 🤖</p>
""", unsafe_allow_html=True)

# --- واجهة الإدخال ---
user_query = st.text_area("💬 Enter your query:")

if st.button("🚀 Run Task"):
    if user_query.strip():
        with st.spinner("⏳ Thinking..."):
            try:
                result = app.invoke({"user_query": user_query})
                answer = result.get("final_answer", "")

                # ✨ لو الناتج فيه كود بس، نضيف شرح تلقائي
                if "def " in answer and "return" in answer and len(answer.split()) < 80:
                    explanation_prompt = f"Explain this Python code in simple terms:\n\n{answer}"
                    try:
                        explain_result = app.invoke({"user_query": explanation_prompt})
                        answer += "\n\n---\n🧠 **Explanation:**\n" + explain_result["final_answer"]
                    except Exception as e:
                        print("Explanation Error:", e)

                # --- عرض النتيجة ---
                st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
                st.markdown("✅ **Final Answer:**", unsafe_allow_html=True)

                if "def " in answer or "import " in answer:
                    parts = answer.split("```")
                    for part in parts:
                        if "def " in part or "import " in part:
                            st.code(part.strip(), language="python")
                        else:
                            st.markdown(part.strip())
                else:
                    st.markdown(answer)
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠️ Error: {e}")
    else:
        st.warning("Please enter a query first.")

st.markdown("<br><hr><center>🚀 Built by <b>Aya Hamada</b> 💙</center>", unsafe_allow_html=True)
