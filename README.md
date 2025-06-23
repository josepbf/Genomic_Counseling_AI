<h1 style="display: flex; align-items: center; justify-content: space-between;">
  <img src="img/logo.png" alt="Genomic Counseling AI AILogo" style="height: 100px;">
  <span>Genomic Counseling AI</span>
</h1>

Genomic Counseling AI turns raw DNA into clear, personalized insights.

Users upload a TXT genotype file. The system analyzes genetic markers, detects disease-linked variants like APOE ε4 for Alzheimer’s, and calculates polygenic risk scores across thousands of traits using real GWAS data.

Using Amazon Bedrock, it then generates a structured, human-readable report that explains risks and gives evidence-based health recommendations. The experience mirrors a real genetic counseling session - automated, accurate, and empowering.

## How we built it

The frontend was built in Streamlit, offering a clean, single-page app for uploading genotype files.

Under the hood, I developed a Python pipeline that detects disease-associated SNPs using a curated adult-onset trait catalog and computes polygenic risk scores with simulation-based percentiles for each trait.

The generative layer uses Amazon Bedrock to power an LLM that converts raw statistical output into a readable, user-friendly interpretation. Prompts were designed carefully to ensure medical responsibility, tone, and clarity.

I used pandas, NumPy, and statsmodels for data processing, while AWS SDKs handled secure model inference with Bedrock’s Amazon Nova foundation model.

## Setup Instructions

### Download GWAS Catalog Data

To run this project, you must first download the full GWAS Catalog dataset:

📥 **Download link**: [https://www.ebi.ac.uk/gwas/api/search/downloads/full](https://www.ebi.ac.uk/gwas/api/search/downloads/full)

Save the downloaded file (rename it to `gwas_catalog.tsv`) into the following directory: `data/catalog/`

## Data Disclaimer Statement

- **Personal Genome Project (PGP)**: Public domain (CC0). See https://www.personalgenomes.org
- **GWAS Catalog**: Sourced from the NHGRI-EBI GWAS Catalog, licensed under the Open Database License (ODbL). See https://www.ebi.ac.uk/gwas/ and https://opendatacommons.org/licenses/odbl/
