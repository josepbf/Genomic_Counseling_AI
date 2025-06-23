
import streamlit as st
import pandas as pd
import utils
import time

st.set_page_config(page_title="Genomic Counseling AI", layout="centered", page_icon="img/logo.png")

# Create two columns
col1, col2 = st.columns([3,1], gap="large")

# Display images in each column
with col1:
    st.title("Genomic Counseling AI")

with col2:
    st.image("img/logo.png", width=115)

st.markdown("Upload a your genotype data file in TXT format to get personalized insights into your genetic predispositions and risks.")

uploaded_file = st.file_uploader("Upload Genotype TXT", type=["txt"])

if uploaded_file is not None:
    genotype_df = utils.load_user_genotype(uploaded_file)

    # Display the first and last two rows of the genotype data
    st.subheader("Your Genotype Data")
    st.dataframe(genotype_df.head(5))

    st.caption("Only the first 5 rows are shown. The full genotype data is used for calculations.")

    with st.spinner('Detecting specific traits and diseases...'):
        df_adult_onset_catalog = utils.load_adult_onset_catalog()
        has_traits_final, no_traits_final = utils.detect_adult_onset_disease(df_adult_onset_catalog, genotype_df)
        time.sleep(2)  # Simulate longer processing time to show the spinner
    st.caption("Detection of specific traits and diseases is completed.")

    with st.spinner('Computing polygenic risk scores (PRS)...'):
        gwas_catalog_df = utils.get_gwas_catalog()
        prs_user = utils.compute_prs(gwas_catalog_df, genotype_df)
    st.caption("Computation of polygenic risk scores (PRS) is completed.")

    prompt = utils.get_prompt(prs_user, has_traits_final, no_traits_final)

    with st.spinner('Generating genomic counseling insights...'):
        llm_response = utils.call_aws_bedrock_llm(prompt)

    st.caption("Genomic Counseling AI insights generation is completed.")

    st.markdown(llm_response)