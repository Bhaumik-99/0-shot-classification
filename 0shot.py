import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Zero-Shot Classifier", layout="wide", page_icon="🎯")

st.title("🎯 Zero-Shot Text Classification")
st.write("Classify text with custom labels using Hugging Face NLI models — no training required!")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    model_name = st.selectbox("Model", [
        "facebook/bart-large-mnli",
    ], help="Select the zero-shot classification model")
    
    multi_label = st.checkbox(
        "Multi-label mode", 
        value=False,
        help="Allow multiple labels to be true simultaneously"
    )
    
    top_k = st.slider(
        "Top k labels to display", 
        min_value=1, 
        max_value=10, 
        value=5,
        help="Number of top predictions to show"
    )
    
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Only show predictions above this score"
    )
    
    show_raw = st.checkbox("Show raw output", value=False)
    show_hypothesis = st.checkbox("Show NLI hypothesis", value=False)

# Cache the model
@st.cache_resource
def get_classifier(name):
    return pipeline("zero-shot-classification", model=name, device=-1)

classifier = get_classifier(model_name)

# Example presets
example_presets = {
    "Technology Topics": "robotics, artificial intelligence, cybersecurity, cloud computing, blockchain",
    "Customer Sentiment": "positive, negative, neutral, urgent, complaint",
    "Content Categories": "news, entertainment, education, sports, politics",
    "Academic Subjects": "mathematics, science, literature, history, philosophy",
    "Business Departments": "sales, marketing, engineering, human resources, finance",
}

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Input Text")
    
with col2:
    use_example = st.selectbox(
        "Load example",
        [""] + list(example_presets.keys()),
        help="Quick-start with example labels"
    )

text = st.text_area(
    "Enter text to classify",
    value="Bhaumik is working on autonomous drones with Pixhawk .",
    height=120,
    help="Enter the text you want to classify"
)

# Labels input
st.subheader("🏷️ Candidate Labels")

col1, col2 = st.columns([3, 1])

with col1:
    if use_example and use_example in example_presets:
        default_labels = example_presets[use_example]
    else:
        default_labels = "robotics, agriculture, sports, programming, hardware"
    
    labels_input = st.text_input(
        "Comma-separated labels",
        value=default_labels,
        help="Enter potential categories for your text"
    )

with col2:
    st.write("")  # Spacing
    st.write("")
    run = st.button("🚀 Classify", type="primary", use_container_width=True)

# Label suggestions
with st.expander("💡 Label Suggestions"):
    st.write("**Common label categories:**")
    cols = st.columns(3)
    with cols[0]:
        st.write("• Technology topics")
        st.write("• Industry sectors")
        st.write("• Emotions/Sentiment")
    with cols[1]:
        st.write("• Academic subjects")
        st.write("• Content types")
        st.write("• Business functions")
    with cols[2]:
        st.write("• Product categories")
        st.write("• Geographic regions")
        st.write("• Time sensitivity")

# Classification
if run:
    if not text.strip():
        st.error("⚠️ Please enter some text to classify.")
    else:
        candidate_labels = [l.strip() for l in labels_input.split(",") if l.strip()]
        if not candidate_labels:
            st.error("⚠️ Please provide at least one candidate label.")
        else:
            with st.spinner("🔄 Classifying..."):
                start_time = time.time()
                try:
                    result = classifier(text, candidate_labels, multi_label=multi_label)
                except TypeError:
                    result = classifier(text, candidate_labels)
                elapsed_time = time.time() - start_time

            # Show raw output if requested
            if show_raw:
                with st.expander("🔍 Raw Pipeline Output"):
                    st.json(result)

            # Parse results
            seq = result.get("sequence", text)
            labels = result.get("labels", [])
            scores = result.get("scores", [])

            df = pd.DataFrame({"label": labels, "score": scores})
            
            # Apply confidence threshold
            df_filtered = df[df["score"] >= confidence_threshold].head(top_k)

            # Display results
            st.markdown("---")
            st.subheader("📊 Results")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Top Prediction", df.iloc[0]["label"])
            with col2:
                st.metric("Confidence", f"{df.iloc[0]['score']:.2%}")
            with col3:
                st.metric("Processing Time", f"{elapsed_time:.3f}s")
            with col4:
                st.metric("Labels Tested", len(candidate_labels))

            # Show hypothesis if requested
            if show_hypothesis:
                st.info(f"**NLI Hypothesis Template:** This example is about {df.iloc[0]['label']}.")

            # Predictions table
            st.markdown("**Predictions Table**")
            if len(df_filtered) == 0:
                st.warning(f"No predictions above confidence threshold of {confidence_threshold:.2f}")
            else:
                # Color code based on score
                def color_scores(val):
                    if val >= 0.7:
                        color = '#d4edda'
                    elif val >= 0.4:
                        color = '#fff3cd'
                    else:
                        color = '#f8d7da'
                    return f'background-color: {color}'
                
                styled_df = df_filtered.style.format({"score": "{:.4f}"}).applymap(
                    color_scores, subset=['score']
                )
                st.dataframe(styled_df, use_container_width=True, height=min(250, len(df_filtered) * 35 + 38))

            # Visualization
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if len(df_filtered) > 0:
                    st.markdown("**Score Distribution**")
                    fig, ax = plt.subplots(figsize=(8, max(3, len(df_filtered) * 0.4)))
                    df_plot = df_filtered.iloc[::-1]
                    bars = ax.barh(df_plot["label"], df_plot["score"])
                    
                    # Color bars based on score
                    colors = ['#28a745' if s >= 0.7 else '#ffc107' if s >= 0.4 else '#dc3545' 
                             for s in df_plot["score"]]
                    for bar, color in zip(bars, colors):
                        bar.set_color(color)
                    
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Confidence Score", fontsize=10)
                    ax.set_title(f"Top {len(df_plot)} Predictions", fontsize=12, fontweight='bold')
                    ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            
            with col2:
                if len(df) > 0:
                    st.markdown("**Score Summary**")
                    st.write(f"**Mean:** {df['score'].mean():.4f}")
                    st.write(f"**Max:** {df['score'].max():.4f}")
                    st.write(f"**Min:** {df['score'].min():.4f}")
                    st.write(f"**Std Dev:** {df['score'].std():.4f}")
                    
                    if multi_label:
                        high_conf = (df["score"] >= 0.5).sum()
                        st.write(f"**Labels ≥ 0.5:** {high_conf}")

            # Export options
            st.markdown("**📥 Export Results**")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download as CSV",
                    csv,
                    file_name="zero_shot_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                json_str = df.to_json(orient='records', indent=2)
                st.download_button(
                    "Download as JSON",
                    json_str,
                    file_name="zero_shot_predictions.json",
                    mime="application/json",
                    use_container_width=True
                )

# Footer
st.markdown("---")
with st.expander("ℹ️ About & Tips"):
    st.markdown("""
    **How Zero-Shot Classification Works:**
    - Uses Natural Language Inference (NLI) models trained on premise-hypothesis pairs
    - Tests hypothesis: "This example is about [label]" for each candidate label
    - No training data needed — works out of the box!
    
    **Tips for Best Results:**
    - Use clear, descriptive labels (e.g., "customer complaint" vs "complaint")
    - For longer texts, keep labels concise
    - Enable multi-label when categories aren't mutually exclusive
    - Try different models — some work better for specific domains
    
    **Model Recommendations:**
    - `bart-large-mnli`: Best overall accuracy, slower
    - `deberta-v3-large-mnli`: Highest performance, slowest
    - `distilbert-base-uncased-mnli`: Fastest, good for real-time use
    
    **Dependencies:**
    ```
    streamlit
    transformers
    torch
    pandas
    matplotlib
    ```
    Install: `pip install streamlit transformers torch pandas matplotlib`
    """)

st.caption("Built with Streamlit 🎈 | Powered by Hugging Face 🤗")