import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.pipeline import process_resumes

# Page Configuration
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for modern look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stDataFrame {
        border: 1px solid #e6e9ef;
        border-radius: 5px;
    }
    h1 {
        color: #1e1e1e;
        text-align: center;
        padding-bottom: 1em;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("🚀 AI Resume Screening System")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📂 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload multiple PDF resumes", 
            type=["pdf"], 
            accept_multiple_files=True,
            help="Select one or more PDF files to analyze."
        )

    with col2:
        st.subheader("📝 Job Description")
        job_description = st.text_area(
            "Paste the job description here",
            height=200,
            placeholder="e.g. We are looking for a Senior Software Engineer with 5+ years of experience in Python, NLP..."
        )

    if st.button("Analyze Resumes"):
        if not uploaded_files:
            st.error("Please upload at least one resume.")
        elif not job_description:
            st.error("Please provide a job description.")
        else:
            with st.spinner("Analyzing resumes..."):
                try:
                    # Run the ML Pipeline
                    results = process_resumes(uploaded_files, job_description)
                    
                    st.success("Analysis Complete!")
                    st.markdown("---")
                    
                    # Layout for results
                    res_col, chart_col = st.columns([3, 2])
                    
                    with res_col:
                        st.subheader("📊 Ranking Results")
                        
                        # Highlight the top match
                        top_resume = results.iloc[0]
                        st.info(f"🏆 **Top Match:** {top_resume['Resume Name']} ({top_resume['Similarity Score']}%)")
                        
                        # Display table
                        st.table(results)

                    with chart_col:
                        st.subheader("📈 Similarity Scores Visualization")
                        if not results.empty:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            sns.barplot(
                                x='Similarity Score', 
                                y='Resume Name', 
                                data=results, 
                                palette="viridis",
                                ax=ax
                            )
                            ax.set_title("Resume Match Confidence")
                            ax.set_xlim(0, 100)
                            st.pyplot(fig)
                        else:
                            st.warning("No data to visualize.")

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

    st.markdown("---")
    st.caption("Built with ❤️ using Python, NLP, and Streamlit")

if __name__ == "__main__":
    main()
