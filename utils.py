import pandas as pd
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import boto3
import json

def call_aws_bedrock_llm(prompt):
    # Create the Bedrock Runtime client
    client = boto3.client("bedrock-runtime", region_name="eu-north-1")

    # Define model ID directly
    model_id = "eu.amazon.nova-micro-v1:0"

    # Payload body
    body = {
        "inferenceConfig": {
            "max_new_tokens": 5000
        },
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    # Make the inference call
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    # Decode and print the response
    response_body = response["body"].read()
    decoded = json.loads(response_body)

    return decoded['output']['message']['content'][0]['text']

def get_prompt(prs_combined, has_traits_final, no_traits_final):

    prompt = """The following is a summary of genetic findings from a user's genomic data file. Provide a concise, personalized interpretation of the user's genetic risks and their implications. The explanation should include relevant gene functions, associated conditions, and practical health recommendations. Your tone should be informative but clear, without unnecessary technical complexity. Emphasize actionable insights backed by evidence where applicable. \n\n**Genomic Summary Input:**"""

    prs_traits_str = "\n- **Higher PRS (compared to population):**\n"

    # SELECT HIGH PERCENTIL: negative PRS. User PRS <<< World PRS
    prs_combined = prs_combined.sort_values('Percentile', ascending=False)
    high_percentil_prs = ''
    for i in range(5):
        high_percentil_prs += '  - ' + prs_combined.iloc[i]['Trait'].split('// ')[1]
        high_percentil_prs += '\n'

    prs_traits_str += high_percentil_prs

    prs_traits_str += "\n- **Lower PRS (compared to population):**\n"

    # SELECT LOW PERCENTIL: negative PRS. User PRS <<< World PRS
    prs_combined = prs_combined.sort_values('Percentile', ascending=True)
    low_percentil_prs = ''
    for i in range(5):
        low_percentil_prs += '  - ' + prs_combined.iloc[i]['Trait'].split('// ')[1]
        low_percentil_prs += '\n'

    prs_traits_str += low_percentil_prs

    prompt += prs_traits_str

    selected_traits_str = "\n- **Specific Genetic Risk Associations:**\n"

    for i in range(len(has_traits_final)):
        selected_traits_str += '  - ' + has_traits_final.iloc[i]['Disease/Trait'] + '. '
        selected_traits_str += 'Gene: ' + has_traits_final.iloc[i]['Gene'] + '. SNP: ' + has_traits_final.iloc[i]['SNP']  + '. Effect: ' + has_traits_final.iloc[i]['Effect']
        
        selected_traits_str += '\n'

    selected_traits_str += "\n- **No Genetic Predisposition:**\n"

    for i in range(len(no_traits_final)):
        selected_traits_str += '  - ' + no_traits_final.iloc[i]['Disease/Trait']
        selected_traits_str += '\n'

    prompt += selected_traits_str

    prompt += """\n**Project Context:**\nYou are part of *Genomic Counseling AI*, a platform designed to help users unlock health insights from raw genomic data. We use your response as part of an AI-driven counseling experience that explains genetic findings in plain language and motivates informed, proactive health decisions.\n\nGenerate an output that:\n- Summarizes the overall genetic risk landscape for the user.\n- Provides gene-specific insights.\n- Offers evidence-based health advice for every disease.\n- Focus actionable insights backed by evidence on the traits and diseases that the user has an Specific Genetic Risk Association or Higher PRS (compared to population), and summarize on Lower PRS (compared to population) or No Genetic Predisposition.\n- Maintains a balance between medical detail and readability."""

    return prompt

def get_gwas_catalog():
    gwas_catalog_df = pd.read_csv("../data/catalogs/gwas_catalog.tsv", sep="\t", low_memory=False)
    gwas_catalog_df = gwas_catalog_df[['DISEASE/TRAIT','PUBMEDID','CHR_ID','CHR_POS','REPORTED GENE(S)',
                                    'STRONGEST SNP-RISK ALLELE','SNPS','SNP_ID_CURRENT','OR or BETA',
                                    'RISK ALLELE FREQUENCY']]

    gwas_catalog_df['DISEASE/TRAIT'] = gwas_catalog_df['PUBMEDID'].astype(str) + ' // ' + gwas_catalog_df['DISEASE/TRAIT']
    gwas_catalog_df['ALLELE'] = gwas_catalog_df['STRONGEST SNP-RISK ALLELE'].str.split('-', expand=True)[1]

    gwas_catalog_df['rsID'] = gwas_catalog_df['STRONGEST SNP-RISK ALLELE'].str.split('-', expand=True)[0]

    gwas_catalog_df = gwas_catalog_df.dropna(subset=['rsID'])
    gwas_catalog_df = gwas_catalog_df.dropna(subset=['CHR_ID'])
    gwas_catalog_df = gwas_catalog_df[gwas_catalog_df['CHR_ID'].str.isnumeric()]
    gwas_catalog_df['CHR_ID'] = gwas_catalog_df['CHR_ID'].astype(int)

    gwas_catalog_df['CHR_ID'] = gwas_catalog_df['CHR_ID'].apply(normalize_chr)
    gwas_catalog_df = gwas_catalog_df[gwas_catalog_df['CHR_ID'].notna()].copy()

    gwas_catalog_df["BETA"] = gwas_catalog_df["OR or BETA"].apply(convert_or_to_beta)

    # Drop any invalid or missing rows
    gwas_catalog_df = gwas_catalog_df.dropna(subset=["BETA"])

    gwas_catalog_df['RAF'] = pd.to_numeric(gwas_catalog_df['RISK ALLELE FREQUENCY'], errors='coerce')

    gwas_catalog_df = gwas_catalog_df[['DISEASE/TRAIT','CHR_ID','CHR_POS','rsID','ALLELE','BETA', 'RAF']]
    gwas_catalog_df = gwas_catalog_df.groupby('DISEASE/TRAIT').filter(lambda x: len(x) >= 5)

    gwas_catalog_df = gwas_catalog_df[gwas_catalog_df['ALLELE'].notna()]
    gwas_catalog_df = gwas_catalog_df[gwas_catalog_df['ALLELE'] != '?']

    return gwas_catalog_df

def load_user_genotype(file_path):
    user_data = pd.read_csv(file_path, sep="\t", low_memory=False)

    return user_data

def load_adult_onset_catalog():
    df_adult_onset_catalog = pd.read_csv('../data/catalogs/Extended_GWAS_Adult-Onset_Disease_Associations.csv')

    return df_adult_onset_catalog

def detect_adult_onset_disease(df_adult_onset_catalog, user_data):
    # Filter catalog to SNPs present in user dataframe
    df_adult_onset_catalog = df_adult_onset_catalog[df_adult_onset_catalog['SNP'].isin(user_data['rsid'])]
    # Merge user and catalog dataframes on SNP/rsid
    df_adult_onset_catalog_merged = pd.merge(df_adult_onset_catalog, user_data, left_on='SNP', right_on='rsid')

    # Determine whether the user has the risk allele
    df_adult_onset_catalog_merged['has_risk'] = df_adult_onset_catalog_merged.apply(lambda row: row['Risk Allele'] in row['genotype'], axis=1)

    # Split into traits the user has and does not have
    has_traits = df_adult_onset_catalog_merged[df_adult_onset_catalog_merged['has_risk']].copy()
    no_traits = df_adult_onset_catalog_merged[~df_adult_onset_catalog_merged['has_risk']].copy()

    # Step 5: Identify common traits and assign them to 'no traits' group
    common_traits = set(has_traits['Disease/Trait']).intersection(no_traits['Disease/Trait'])

    # Move common traits entirely to 'no traits' group
    has_traits_final = has_traits[~has_traits['Disease/Trait'].isin(common_traits)]
    no_traits_final = pd.concat([
        no_traits,
        has_traits[has_traits['Disease/Trait'].isin(common_traits)]
    ])

    has_traits_final = has_traits_final.drop_duplicates(subset='Disease/Trait', keep='first')
    no_traits_final = no_traits_final.drop_duplicates(subset='Disease/Trait', keep='first')

    return has_traits_final, no_traits_final


def compute_prs(gwas_catalog_df, user_data):

    # Merge with user genotype
    merged = pd.merge(gwas_catalog_df, user_data, left_on='rsID', right_on='rsid')

    # ----------------------------
    # 2. Compute user PRS per trait
    # ----------------------------

    def count_effect_alleles(row):
        return str(row['genotype']).count(str(row['ALLELE']))

    merged['effect_alleles'] = merged.apply(count_effect_alleles, axis=1)
    merged['prs_contrib'] = merged['effect_alleles'] * merged['BETA']

    prs_real = merged.groupby('DISEASE/TRAIT')['prs_contrib'].sum().reset_index()
    prs_real.columns = ['Trait', 'Real_PRS']

    # ----------------------------
    # 3. Simulate genotypes and PRS per trait
    # ----------------------------

    N = 10000  # simulated individuals

    reference_distributions = []

    for trait, df in merged.groupby('DISEASE/TRAIT'):

        if df.shape[0] < 5:
            continue  # skip unstable traits with too few SNPs

        prs_sim = np.zeros(N)
        valid = True
        
        for _, row in df.iterrows():
            raf = row['RAF']
            beta = row['BETA']

            if not (0.01 <= raf <= 0.99) or np.isnan(beta):
                valid = False
                break
                
            g = np.random.binomial(2, raf, N)  # simulate 0/1/2 risk alleles
            prs_sim += g * beta

        if not valid:
            continue

        # Get user's PRS
        user_prs = df['prs_contrib'].sum()
        #percentile = percentileofscore(prs_sim, user_prs, kind='rank')
        ecdf = ECDF(prs_sim)
        percentile = ecdf(user_prs) * 100

        if user_prs < np.min(prs_sim) or user_prs > np.max(prs_sim):
            #print(f"Extreme PRS for {trait}: {user_prs:.3f} outside [{np.min(prs_sim):.3f}, {np.max(prs_sim):.3f}]")
            continue

        reference_distributions.append({
            'Trait': trait,
            'User_PRS': user_prs,
            'Percentile': percentile,
            'Mean_PRS': np.mean(prs_sim),
            'Std_PRS': np.std(prs_sim),
            'N_SNPs': df.shape[0]
        })

    prs_combined = pd.DataFrame(reference_distributions)
    prs_combined = prs_combined[(prs_combined['Percentile'] > 1) & (prs_combined['Percentile'] < 99)]
    prs_combined['PRS_Z'] = (prs_combined['User_PRS'] - prs_combined['Mean_PRS']) / prs_combined['Std_PRS']

    return prs_combined

# Define valid chromosomes: strings '1'-'22', 'X', 'Y', 'MT'
# Normalize CHR_KEY
def normalize_chr(chr_val):
    try:
        # Try numeric conversion: e.g., '1.0' -> 1 -> '1'
        num = float(chr_val)
        if num.is_integer():
            num_str = str(int(num))
            if 1 <= int(num_str) <= 22:
                return num_str
        return None
    except:
        # Handle non-numeric: check for X, Y, MT
        chr_str = str(chr_val).strip().upper()
        if chr_str in {'X', 'Y', 'MT'}:
            return chr_str
        return None
    
# Convert OR to BETA
def convert_or_to_beta(val):
    try:
        val = float(val)
        if val <= 0:
            return np.nan  # invalid ORs
        return np.log(val)
    except:
        return np.nan
