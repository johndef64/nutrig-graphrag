"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}
default_entity_types = ["organization", "person", "geo", "event"]
# default_entity_types_full = [
#     "Gene",
#     "Genomic Variant",
#     "Transcript",
#     "Protein",
#     "Nutrient",
#     "Food",
#     "Dietary Supplement",
#     "Dietary Pattern",
#     "Physiological Process",
#     "Metabolic Pathway",
#     "Organ",
#     "Tissue",
#     "Cell",
#     "Hormone",
#     "Disease",
#     "Disorder",
#     "Symptom",
#     "Biomarker",
#     "Biological Pathway",
#     "Molecular Interaction",
#     "Epigenetic Modification",
#     "Environmental Factor",
#     "Lifestyle Factor",
#     "Drug",
#     "Therapy",
#     "Population",
#     "Study",
#     "Enzyme",
#     "Metabolite"
# ]
default_entity_types = [
    "Gene",
    "Genetic Variant",
    "Transcript",
    "Protein",
    "Nutrient",
    "Food",
    "Dietary Pattern",
    "Physiological Process",
    "Metabolic Pathway",
    "Molecular Interaction",
    #"Epigenetic Modification",
    "Environmental Factor",
    #"Lifestyle Factor",
    "Disease",
    #"Drug",
]

PROMPTS[
    "claim_extraction"
] = """-Target activity-
You are an intelligent assistant that helps a human analyst to analyze claims against biomedical entities presented in a text document.

-Goal-
Given a text document that is potentially relevant to this activity, an biomedical entity specification, and a claim description, extract all entities that match the entity specification and all claims against those entities.

-Steps-
1. Extract all named entities that match the predefined entity specification. Entity specification can either be a list of entity names or a list of entity types.
2. For each entity identified in step 1, extract all claims associated with the entity. Claims need to match the specified claim description, and the entity should be the subject of the claim.
For each claim, extract the following information:
- Subject: name of the biomedical entity that is subject of the claim, capitalized. The subject entity is one that committed the action described in the claim. Subject needs to be one of the named entities identified in step 1.
- Object: name of the entity that is object of the claim, capitalized. The object entity is one that either reports/handles or is affected by the action described in the claim. If object entity is unknown, use **NONE**.
- Claim Type: overall category of the claim, capitalized. Name it in a way that can be repeated across multiple text inputs, so that similar claims share the same claim type
- Claim Status: **TRUE**, **FALSE**, or **SUSPECTED**. TRUE means the claim is confirmed, FALSE means the claim is found to be False, SUSPECTED means the claim is not verified.
- Claim Description: Detailed description explaining the reasoning behind the claim, together with all the related evidence and references.
- Claim Date: Period (start_date, end_date) when the claim was made. Both start_date and end_date should be in ISO-8601 format. If the claim was made on a single date rather than a date range, set the same date for both start_date and end_date. If date is unknown, return **NONE**.
- Claim Source Text: List of **all** quotes from the original text that are relevant to the claim.

Format each claim as (<subject_entity>{tuple_delimiter}<object_entity>{tuple_delimiter}<claim_type>{tuple_delimiter}<claim_status>{tuple_delimiter}<claim_start_date>{tuple_delimiter}<claim_end_date>{tuple_delimiter}<claim_description>{tuple_delimiter}<claim_source>)

3. Return output in English as a single list of all the claims identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

-Examples-
Example 1:
Entity specification: GENE VARIANT
Claim description: association with nutrient and disease risk
Text: According to a study published on 2021/09/15, the gene variant rs12345 (located on chromosome 12) was found to be associated with an increased risk of developing cardiovascular disease when individuals consumed high levels of saturated fats. The study, conducted by Research Institute X, analyzed data from 10,000 participants over a 10-year period and identified that carriers of the rs12345 variant who followed a high-saturated-fat diet had a 2.3-fold increased risk of cardiovascular disease compared to non-carriers on the same diet. The study was funded by Health Organization Y and published in the journal *Nutrition and Genetics*.
Output:

(GENE VARIANT rs12345{tuple_delimiter}CHROMOSOME 12{tuple_delimiter}NUTRIENT{tuple_delimiter}SATURATED FATS{tuple_delimiter}DISEASE{tuple_delimiter}CARDIOVASCULAR DISEASE{tuple_delimiter}ASSOCIATION{tuple_delimiter}TRUE{tuple_delimiter}2021-09-15T00:00:00{tuple_delimiter}2021-09-15T00:00:00{tuple_delimiter}The gene variant rs12345 located on chromosome 12 was found to be associated with an increased risk of cardiovascular disease in individuals consuming high levels of saturated fats according to a study published on 2021/09/15{tuple_delimiter}According to a study published on 2021/09/15, the gene variant rs12345 was found to be associated with an increased risk of cardiovascular disease when individuals consumed high levels of saturated fats.)
{completion_delimiter}


Example 2: 
entity specification: GENE A, NUTRIENT X
claim description: Associations between genetic variants and nutritional responses
text: According to a study published in Nature Genetics on 2023/05/15, GENE A was associated with an increased response to dietary intake of NUTRIENT X, such as vitamin D, in influencing cardiovascular health outcomes. The study suggested that individuals with the GENE A variant exhibited improved biomarkers for heart disease when consuming higher levels of NUTRIENT X.

Output:
(GENE A{tuple_delimiter}NUTRIENT X{tuple_delimiter}CARDIOVASCULAR HEALTH{tuple_delimiter}ASSOCIATION{tuple_delimiter}2023-05-15T00:00:00{tuple_delimiter}2023-05-15T00:00:00{tuple_delimiter}GENE A was associated with an increased response to dietary intake of NUTRIENT X, such as vitamin D, in influencing cardiovascular health outcomes according to a study published in Nature Genetics on 2023-05-15{tuple_delimiter}The study suggested that individuals with the GENE A variant exhibited improved biomarkers for heart disease when consuming higher levels of NUTRIENT X) {record_delimiter}
(NUTRIENT X{tuple_delimiter}GENE A{tuple_delimiter}DIETARY RESPONSE{tuple_delimiter}INFLUENCE{tuple_delimiter}2023-05-15T00:00:00{tuple_delimiter}2023-05-15T00:00:00{tuple_delimiter}NUTRIENT X, such as vitamin D, was found to influence cardiovascular health outcomes in individuals with GENE A according to a study published in Nature Genetics on 2023-05-15{tuple_delimiter}The study suggested that individuals with the GENE A variant exhibited improved biomarkers for heart disease when consuming higher levels of NUTRIENT X) {completion_delimiter}


-Real Data-
Use the following input for your answer.
Entity specification: {entity_specs}
Claim description: {claim_description}
Text: {input_text}
Output: """

PROMPTS[
    "community_report"
] = """You are an AI assistant that helps a human analyst to perform biomedical information discovery. 
Biomedical information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., genes and diseases) within a network.

# Goal
Write a comprehensive report of a community of entities in the domain of genetics and nutrition, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact on health, disease, and personalized nutrition. The content of this report includes an overview of the community's key entities, their biological relevance, molecular interactions, dietary implications, and noteworthy scientific claims.

# Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
            ...
        ]
    }}

# Grounding Rules
Do not include information where the supporting evidence for it is not provided.


# Example Input
-----------
Text:
```
Entities:
```csv
id,entity,type,description
1,MTHFR,gene,MTHFR is a gene involved in folate metabolism
2,SLC30A8,gene,SLC30A8 is a gene associated with zinc transport
3,FADS1,gene,FADS1 is a gene linked to fatty acid metabolism
4,omega-3 fatty acids,nutrient,Omega-3 fatty acids are essential dietary fats
5,folate,nutrient,Folate is a B-vitamin critical for metabolic processes
6,type 2 diabetes,disease,Type 2 diabetes is a metabolic disorder influenced by genetic and dietary factors
7,MTHFR C677T,genetic variant,MTHFR C677T is a SNP affecting folate metabolism
8,SLC30A8 rs13266634,genetic variant,SLC30A8 rs13266634 is a SNP linked to zinc transport
```
Relationships:
```csv
id,source,target,description
1,MTHFR,folate,MTHFR gene is involved in folate metabolism
2,SLC30A8,zinc,SLC30A8 gene is associated with zinc transport
3,FADS1,omega-3 fatty acids,FADS1 gene is linked to omega-3 fatty acid metabolism
4,omega-3 fatty acids,type 2 diabetes,Omega-3 fatty acids may reduce inflammation in type 2 diabetes
5,folate,MTHFR C677T,MTHFR C677T variant affects folate metabolism
6,SLC30A8 rs13266634,zinc,SLC30A8 rs13266634 variant influences zinc transport
7,MTHFR C677T,type 2 diabetes,MTHFR C677T variant is associated with increased risk of type 2 diabetes
8,SLC30A8 rs13266634,type 2 diabetes,SLC30A8 rs13266634 variant may influence insulin secretion
```
```
Output:
{{
    "title": "Genetic and Nutrient Interactions in Type 2 Diabetes",
    "summary": "The community revolves around type 2 diabetes, which is influenced by genetic and dietary factors. Key entities include genes such as MTHFR, SLC30A8, and FADS1, nutrients like omega-3 fatty acids and folate, and genetic variants like MTHFR C677T and SLC30A8 rs13266634. These entities are interconnected through their roles in nutrient metabolism, insulin secretion, and disease risk.",
    "rating": 5.0,
    "rating_explanation": "The impact severity rating is high due to the significant influence of genetic and dietary factors on type 2 diabetes risk and management.",
    "findings": [
        {{
            "summary": "MTHFR gene and its role in folate metabolism",
            "explanation": "The MTHFR gene is central to folate metabolism, and its variant, MTHFR C677T, is associated with reduced enzymatic activity. This can lead to elevated homocysteine levels and increased risk of type 2 diabetes. The relationship between MTHFR and folate highlights the importance of dietary folate intake in managing metabolic health."
        }},
        {{
            "summary": "SLC30A8 gene and zinc transport",
            "explanation": "The SLC30A8 gene is crucial for zinc transport in pancreatic beta-cells, and its variant, SLC30A8 rs13266634, has been linked to impaired insulin secretion. This underscores the role of zinc in insulin function and the potential impact of genetic variants on type 2 diabetes susceptibility."
        }},
        {{
            "summary": "FADS1 gene and omega-3 fatty acid metabolism",
            "explanation": "The FADS1 gene is involved in the metabolism of omega-3 fatty acids, which are essential for reducing inflammation. Variations in FADS1 can influence the effectiveness of omega-3 supplementation, highlighting the importance of personalized dietary recommendations for managing inflammation in type 2 diabetes."
        }},
        {{
            "summary": "Omega-3 fatty acids and their role in inflammation",
            "explanation": "Omega-3 fatty acids are linked to reduced inflammation, which is a key factor in type 2 diabetes management. Their interaction with genetic variants like FADS1 suggests that tailored dietary interventions could optimize their anti-inflammatory effects."
        }},
        {{
            "summary": "Folate and its interaction with MTHFR C677T",
            "explanation": "Folate is a critical nutrient for metabolic processes, and its interaction with the MTHFR C677T variant highlights the importance of genetic considerations in dietary recommendations. Individuals with this variant may require higher folate intake to mitigate metabolic risks."
        }}
    ]
}}



# Real Data

Use the following text for your answer. Do not make anything up in your answer.

Text:
```
{input_text}
```

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
            ...
        ]
    }}

# Grounding Rules
Do not include information where the supporting evidence for it is not provided.

Output:
"""

PROMPTS[
    "entity_extraction"
] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all biomedical entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [gene, genomic variant, transcript, protein, nutrient, food, dietary pattern, physiological process, metabolic pathway, disease, biological pathway, molecular interaction, epigenetic modification, environmental factor, lifestyle factor, drug, therapy]
Text:
The NOS1 gene plays a critical role in cardiovascular health by regulating nitric oxide production. A genomic variant in the NOS1 gene has been linked to increased risk of hypertension. The transcript of the NOS1 gene is influenced by dietary patterns, particularly the intake of folate, a key nutrient found in leafy greens. Folate metabolism is a key physiological process that interacts with the NOS1 protein to maintain vascular function. Dysregulation of this process can lead to cardiovascular disease, a condition influenced by both genetic and environmental factors. The metabolic pathway of nitric oxide synthesis is central to this interaction, and alterations in this pathway have been observed in individuals with epigenetic modifications linked to poor dietary patterns. Environmental factors, such as diet, and lifestyle factors, such as physical activity, further modulate the relationship between the NOS1 gene and cardiovascular health. Interventions, including drug therapies like statins, may be used to manage disease risk, but personalized therapies based on genetic and nutritional profiles are increasingly being explored.

################
Output:
("entity"{tuple_delimiter}"NOS1"{tuple_delimiter}"Gene"{tuple_delimiter}"NOS1 is a gene that plays a critical role in cardiovascular health by regulating nitric oxide production."){record_delimiter} ("entity"{tuple_delimiter}"Genomic Variant"{tuple_delimiter}"Genomic Variant"{tuple_delimiter}"A genomic variant in the NOS1 gene has been linked to increased risk of hypertension."){record_delimiter} ("entity"{tuple_delimiter}"Transcript"{tuple_delimiter}"Transcript"{tuple_delimiter}"The transcript of the NOS1 gene is influenced by dietary patterns, particularly the intake of folate."){record_delimiter} ("entity"{tuple_delimiter}"NOS1 Protein"{tuple_delimiter}"Protein"{tuple_delimiter}"The NOS1 protein is involved in vascular function and interacts with folate metabolism."){record_delimiter} ("entity"{tuple_delimiter}"Folate"{tuple_delimiter}"Nutrient"{tuple_delimiter}"Folate is a key nutrient found in leafy greens and plays a role in nitric oxide production."){record_delimiter} ("entity"{tuple_delimiter}"Leafy Greens"{tuple_delimiter}"Food"{tuple_delimiter}"Leafy greens are a food source rich in folate, a nutrient critical for cardiovascular health."){record_delimiter} ("entity"{tuple_delimiter}"Dietary Pattern"{tuple_delimiter}"Dietary Pattern"{tuple_delimiter}"Dietary patterns influence the transcript of the NOS1 gene and folate metabolism."){record_delimiter} ("entity"{tuple_delimiter}"Folate Metabolism"{tuple_delimiter}"Physiological Process"{tuple_delimiter}"Folate metabolism is a physiological process that interacts with the NOS1 protein to maintain vascular function."){record_delimiter} ("entity"{tuple_delimiter}"Nitric Oxide Synthesis"{tuple_delimiter}"Metabolic Pathway"{tuple_delimiter}"The metabolic pathway of nitric oxide synthesis is central to the interaction between the NOS1 gene and cardiovascular health."){record_delimiter} ("entity"{tuple_delimiter}"Cardiovascular Disease"{tuple_delimiter}"Disease"{tuple_delimiter}"Cardiovascular disease is a condition influenced by both genetic and environmental factors, including the NOS1 gene and dietary patterns."){record_delimiter} ("entity"{tuple_delimiter}"Epigenetic Modification"{tuple_delimiter}"Epigenetic Modification"{tuple_delimiter}"Epigenetic modifications linked to poor dietary patterns can alter the metabolic pathway of nitric oxide synthesis."){record_delimiter} ("entity"{tuple_delimiter}"Diet"{tuple_delimiter}"Environmental Factor"{tuple_delimiter}"Diet is an environmental factor that modulates the relationship between the NOS1 gene and cardiovascular health."){record_delimiter} ("entity"{tuple_delimiter}"Physical Activity"{tuple_delimiter}"Lifestyle Factor"{tuple_delimiter}"Physical activity is a lifestyle factor that further modulates cardiovascular health."){record_delimiter} ("entity"{tuple_delimiter}"Statins"{tuple_delimiter}"Drug"{tuple_delimiter}"Statins are a drug therapy used to manage cardiovascular disease risk."){record_delimiter} ("entity"{tuple_delimiter}"Personalized Therapy"{tuple_delimiter}"Therapy"{tuple_delimiter}"Personalized therapies based on genetic and nutritional profiles are being explored to manage cardiovascular health."){record_delimiter}

("relationship"{tuple_delimiter}"NOS1"{tuple_delimiter}"Folate"{tuple_delimiter}"The NOS1 gene interacts with folate metabolism to regulate nitric oxide production."){tuple_delimiter}1){record_delimiter} ("relationship"{tuple_delimiter}"Genomic Variant"{tuple_delimiter}"Cardiovascular Disease"{tuple_delimiter}"A genomic variant in the NOS1 gene is linked to an increased risk of cardiovascular disease."){tuple_delimiter}2){record_delimiter} ("relationship"{tuple_delimiter}"Transcript"{tuple_delimiter}"Dietary Pattern"{tuple_delimiter}"The transcript of the NOS1 gene is influenced by dietary patterns, particularly folate intake."){tuple_delimiter}3){record_delimiter} ("relationship"{tuple_delimiter}"NOS1 Protein"{tuple_delimiter}"Folate Metabolism"{tuple_delimiter}"The NOS1 protein interacts with folate metabolism to maintain vascular function."){tuple_delimiter}4){record_delimiter} ("relationship"{tuple_delimiter}"Folate"{tuple_delimiter}"Leafy Greens"{tuple_delimiter}"Folate is a nutrient found in leafy greens, which are important for cardiovascular health."){tuple_delimiter}5){record_delimiter} ("relationship"{tuple_delimiter}"Dietary Pattern"{tuple_delimiter}"Epigenetic Modification"{tuple_delimiter}"Poor dietary patterns can lead to epigenetic modifications that alter nitric oxide synthesis."){tuple_delimiter}6){record_delimiter} ("relationship"{tuple_delimiter}"Nitric Oxide Synthesis"{tuple_delimiter}"Cardiovascular Disease"{tuple_delimiter}"Alterations in the nitric oxide synthesis pathway are associated with cardiovascular disease."){tuple_delimiter}7){record_delimiter} ("relationship"{tuple_delimiter}"Diet"{tuple_delimiter}"Physical Activity"{tuple_delimiter}"Diet and physical activity are lifestyle and environmental factors that modulate cardiovascular health."){tuple_delimiter}8){record_delimiter} ("relationship"{tuple_delimiter}"Statins"{tuple_delimiter}"Personalized Therapy"{tuple_delimiter}"Statins are a drug therapy that may be part of personalized therapies for cardiovascular disease management."){tuple_delimiter}9){completion_delimiter}
#############################


#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""


PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the biomedical data provided below.
Given one or two biomedical entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""


PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["DEFAULT_ENTITY_TYPES"] = default_entity_types
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS[
    "local_rag_response"
] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}


---Data tables---

{context_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS[
    "global_map_rag_points"
] = """---Role---

You are a helpful assistant responding to questions about biomedical data in the tables provided.


---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1...", "score": score_value}},
        {{"description": "Description of point 2...", "score": score_value}}
    ]
}}

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
Do not include information where the supporting evidence for it is not provided.


---Data tables---

{context_data}

---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
Do not include information where the supporting evidence for it is not provided.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1", "score": score_value}},
        {{"description": "Description of point 2", "score": score_value}}
    ]
}}
"""

PROMPTS[
    "global_reduce_rag_response"
] = """---Role---

You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Analyst Reports---

{report_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS[
    "naive_rag_response"
] = """You're a helpful assistant
Below are the knowledge you know:
{content_data}
---
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
---Target response length and format---
{response_type}
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["default_text_separator"] = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]


# for i in PROMPTS.keys():
#     print(i)

# print(PROMPTS[list(PROMPTS.keys())[0]])
PROMPTS["DEFAULT_ENTITY_TYPES"]
