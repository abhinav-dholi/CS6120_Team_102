import os
import re
import ast
import datetime
import pandas as pd
import pickle
import numpy as np
import spacy
import sqlalchemy
from sqlalchemy import create_engine, text
from flask import Flask, render_template, request, redirect, url_for, jsonify
from dotenv import load_dotenv
from spacy import displacy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import acronyms_to_entities, index_to_label, MAX_LENGTH

import sqlparse
from sqlparse.sql import (
    IdentifierList, Identifier, Parenthesis, Statement
)
from sqlparse.tokens import DML

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_ollama import OllamaLLM
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# sets up the web app using Flask
app = Flask(__name__)

# special version of OllamaLLM that answers questions
class OllamaWithTools(OllamaLLM):
    def bind_tools(self, tool_classes):
        """
        Allows external tools to connect to the model.
        """
        return self 
    
    def invoke(self, prompt, config=None, **kwargs):
        """
        Overwrites the default invoke method to handle the model's response.
        """
        result = super().invoke(prompt, **kwargs)
        if isinstance(result, str):
            return AIMessage(content=result)
        return result

# initializes Ollama (deepseek model with 8B parameters)
llm = OllamaWithTools(
    model="deepseek-r1:8b", 
    model_options={
        "temperature": 0.0  # low temperature = more predictable & less random answers
    }
)

# fetches environment from an .env file
load_dotenv()
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")

# connects to database
db_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
db = SQLDatabase.from_uri(db_uri)
engine = db._engine

# sets up SQL for LLM to answer queries
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
hub_system_prompt = prompt_template.format(dialect="MYSQL", top_k=5)

# loads trained NER model for clinical notes
ner_model = load_model("ner_model.h5")

# loads spaCy's english NLP model for tokenization & lemmatization
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# whitelist for database tables & columns to use
VALID_SCHEMA = {
    "patients": {
        "Id","FIRST","MIDDLE","LAST","BIRTHDATE","DEATHDATE",
        "GENDER","RACE","ETHNICITY","ADDRESS","CITY","STATE","ZIP","MARITAL"
    },
    "organizations": {
        "Id","NAME","ADDRESS","CITY","STATE","ZIP","PHONE"
    },
    "providers": {
        "Id","NAME","GENDER","SPECIALTY","ADDRESS","CITY","STATE","ZIP"
    },
    "encounters": {
        "Id","START","STOP","PATIENT","ORGANIZATION","PROVIDER",
        "ENCOUNTERCLASS","CODE","DESCRIPTION","REASONCODE","REASONDESCRIPTION"
    },
    "conditions": {
        "START","STOP","PATIENT","ENCOUNTER","CODE","DESCRIPTION"
    },
    "medications": {
        "START","STOP","PATIENT","ENCOUNTER","CODE","DESCRIPTION",
        "REASONCODE","REASONDESCRIPTION"
    },
    "observations": {
        "DATE","PATIENT","ENCOUNTER","CODE","DESCRIPTION","VALUE","UNITS"
    },
    "procedures": {
        "DATE","PATIENT","ENCOUNTER","CODE","DESCRIPTION"
    },
    "immunizations": {
        "DATE","PATIENT","ENCOUNTER","CODE","DESCRIPTION"
    },
    "allergies": {
        "START","PATIENT","ENCOUNTER","CODE","DESCRIPTION"
    }
}

# base context for the AI to understand the database
base_custom_context = """
You are a strict MySQL expert working with a Synthea-generated healthcare database. 
IMPORTANT POINTS:
• The table for patients is "patients" (plural), not "patient".
• The table for encounters is "encounters" (plural), not "encounter".
• Same for conditions, medications, observations, etc. All are plural.

Only use these exact tables and columns:

[patients]: Id, FIRST, MIDDLE, LAST, BIRTHDATE, DEATHDATE, GENDER, RACE, ETHNICITY, ADDRESS, CITY, STATE, ZIP, MARITAL
[organizations]: Id, NAME, ADDRESS, CITY, STATE, ZIP, PHONE
[providers]: Id, NAME, GENDER, SPECIALTY, ADDRESS, CITY, STATE, ZIP
[encounters]: Id, START, STOP, PATIENT, ORGANIZATION, PROVIDER, ENCOUNTERCLASS, CODE, DESCRIPTION, REASONCODE, REASONDESCRIPTION
[conditions]: START, STOP, PATIENT, ENCOUNTER, CODE, DESCRIPTION
[medications]: START, STOP, PATIENT, ENCOUNTER, CODE, DESCRIPTION, REASONCODE, REASONDESCRIPTION
[observations]: DATE, PATIENT, ENCOUNTER, CODE, DESCRIPTION, VALUE, UNITS
[procedures]: DATE, PATIENT, ENCOUNTER, CODE, DESCRIPTION
[immunizations]: DATE, PATIENT, ENCOUNTER, CODE, DESCRIPTION
[allergies]: START, PATIENT, ENCOUNTER, CODE, DESCRIPTION

RELATIONSHIPS:
• encounters.PATIENT -> patients.Id
• encounters.PROVIDER -> providers.Id
• encounters.ORGANIZATION -> organizations.Id
• conditions.PATIENT -> patients.Id
• conditions.ENCOUNTER -> encounters.Id
• medications.PATIENT -> patients.Id
• medications.ENCOUNTER -> encounters.Id
• observations.PATIENT -> patients.Id
• observations.ENCOUNTER -> encounters.Id
• procedures.PATIENT -> patients.Id
• procedures.ENCOUNTER -> encounters.Id
• immunizations.PATIENT -> patients.Id
• immunizations.ENCOUNTER -> encounters.Id
• allergies.PATIENT -> patients.Id
• allergies.ENCOUNTER -> encounters.Id

NOTES:
- Do NOT use singular table names like "patient" or "encounter."
- Only use these columns/tables. If unavailable, say so.
- Think carefully about references: e.g. patients has no 'NAME' column, etc.
- Only print the columns necessary to answer the query or mentioned in the query.
- Don't limit the output until explicitly requested.
- Think over again to verify the results or if the query can be simplified.
- Verify if the mentioned columns exist in the table or not.
- Avoid using aliases unless necessary.
- When necessary show only unique outputs.
"""

def validate_sql(sql_query: str, valid_schema: dict):
    """
    Checks if SQL query is valid and uses only allowed tables and columns.
    """
    parsed_statements = sqlparse.parse(sql_query)
    if not parsed_statements:
        return False, "Empty or invalid SQL."
    
    stmt = parsed_statements[0]

    # make sure its a SELECT query (not INSERT, DELETE, etc.)
    dml_tokens = [t for t in stmt.tokens if t.ttype == DML]
    if not dml_tokens:
        return False, "No DML token found. Only SELECT queries are allowed."
    if any(t.value.upper() != "SELECT" for t in dml_tokens):
        return False, "Only SELECT queries are allowed."
    try:
        parse_and_validate_statement(stmt, valid_schema)
    except ValueError as ve:
        return False, str(ve)
    
    return True, ""

def parse_and_validate_statement(stmt: Statement, valid_schema: dict, parent_alias_map=None):
    """
    Parses query and makes sure all tables and columns are valid.
    """
    if parent_alias_map is None:
        parent_alias_map = {}
    local_alias_map = dict(parent_alias_map)
    from_seen = False
    join_seen = False
    tokens = list(stmt.tokens)

    for token in tokens:
        # handles subqueries like (SELECT ...) inside parentheses
        if isinstance(token, Parenthesis):
            inner_stmts = sqlparse.parse(token.value.strip("()"))
            for inner_stmt in inner_stmts:
                parse_and_validate_statement(inner_stmt, valid_schema, local_alias_map)

        # tracks when we see FROM or JOIN so we can map aliases next
        if token.is_keyword and token.value.upper() in ["FROM","JOIN"]:
            from_seen = (token.value.upper() == "FROM")
            join_seen = (token.value.upper() == "JOIN")
            continue
        
        # maps each alias in FROM or JOIN sections
        if from_seen or join_seen:
            if isinstance(token, IdentifierList):
                for ident in token.get_identifiers():
                    record_alias(ident, local_alias_map, valid_schema)
                from_seen = False
                join_seen = False
            elif isinstance(token, Identifier):
                record_alias(token, local_alias_map, valid_schema)
                from_seen = False
                join_seen = False

    # checks that every column is valid
    pattern = r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)"
    matches = re.findall(pattern, str(stmt))
    for alias,col in matches:
        alias_lower = alias.lower()
        if alias_lower in local_alias_map:
            real_table_name = local_alias_map[alias_lower]
            check_column_valid(real_table_name, col, valid_schema)
        else:
            matched_table = None
            for t in valid_schema.keys():
                if t.lower() == alias_lower:
                    matched_table = t
                    break
            if matched_table:
                check_column_valid(matched_table, col, valid_schema)
            else:
                raise ValueError(f"Unknown alias or table reference '{alias}' in query.")

def record_alias(ident: Identifier, alias_map: dict, valid_schema: dict):
    """
    Links an alias (ie. "p") to the actual table name (ie. "patients").
    """
    real_name = ident.get_real_name()
    alias = ident.get_alias()
    if not real_name:
        return
    matched_table = None
    for t in valid_schema.keys():
        if t.lower() == real_name.lower():
            matched_table = t
            break
    if not matched_table:
        raise ValueError(f"Invalid table reference: '{real_name}' is not allowed.")
    if alias:
        alias_map[alias.lower()] = matched_table
    else:
        alias_map[matched_table.lower()] = matched_table

def check_column_valid(table_name: str, col_name: str, valid_schema: dict):
    """
    Checks if the column exists in the table schema.
    """
    allowed_cols = {c.lower() for c in valid_schema[table_name]}
    if col_name.lower() not in allowed_cols:
        raise ValueError(f"Invalid column '{col_name}' for table '{table_name}'.")

def create_patient_agent(patient_id):
    """
    Creates an AI agent focused on one patient.
    """
    patient_context = f"Patient Context: Only consider data for the patient with Id = '{patient_id}'."
    system_message = base_custom_context + "\n\n" + patient_context + "\n\n" + hub_system_prompt
    return create_react_agent(llm, tools, prompt=system_message)

def run_query_with_columns(query: str):
    """
    Runs raw SQL query and returns column names and result rows.
    """
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text(query))
        col_names = list(result.keys())
        rows = result.fetchall()
    row_list = [list(r) for r in rows]
    return col_names, row_list

def generate_valid_query(nurse_query, patient_id, max_attempts=3):
    """
    Given nurse's question, generates a valid SQL query and runs it.
    """
    last_error = ""
    final_query_text = None

    for attempt in range(max_attempts):
        if last_error:
            feedback = (
                f"\nThe previous query failed or used invalid references. Error: {last_error}\n"
                "Remember: Only the listed tables/columns are allowed."
            )
            prompt_text = nurse_query + feedback
        else:
            prompt_text = nurse_query

        # creates an AI agent focused on that one patient
        agent = create_patient_agent(patient_id)

        # sends query to agent & captures full step-by-step output
        steps = list(agent.stream({"messages":[{"role":"user","content":prompt_text}]}, 
                                  stream_mode="values"))
        final_message = steps[-1]["messages"][-1].content

        # extracts SQL code block from AI’s response
        query_match = re.search(r"```sql\s*(.*?)\s*```", final_message, re.DOTALL)
        if query_match:
            final_query_text = query_match.group(1).strip()
        else:
            final_query_text = final_message.strip()

        print(f"[DEBUG Attempt {attempt+1}] Proposed SQL:\n{final_query_text}")

        # make sure the query follows allowed schema/tables
        is_valid, validation_err = validate_sql(final_query_text, VALID_SCHEMA)
        if not is_valid:
            last_error = validation_err
            continue
        
        # try executing the query
        try:
            col_names, rows = run_query_with_columns(final_query_text)
            return final_query_text, (col_names, rows)
        except Exception as e:
            last_error = f"Runtime error: {e}"
            continue

    return final_query_text, f"Error after {max_attempts} attempts: {last_error}"

def chunk_text(text, max_chars=3000):
    """
    Splits long text into smaller chunks to help it fit within LLM's input size limit.
    """
    chunks=[]
    start=0
    while start<len(text):
        end=start+max_chars
        chunk=text[start:end]
        chunks.append(chunk)
        start=end
    return chunks

def create_summarizer_agent():
    """
    Creates a simpler LLM agent used just for summarizing text (no SQL-specific logic or instructions).
    """
    summarizer_prompt="You are a summarizing assistant. Summarize user-provided text. No SQL logic needed."
    return create_react_agent(llm, tools, prompt=summarizer_prompt)

def remove_think_tokens(text: str):
    """
    Cleans up LLM responses by removing <think> tokens or sections if they appear in the summary.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()

def summarize_text(text, llm_prompt="Summarize the following text:"):
    """
    Breaks up long input into chunks, summarizes each piece using the LLM and combines them into one summary.
    """
    raw_chunks=chunk_text(text, max_chars=3000)
    partial_summaries=[]

    # summarizes each piece
    for idx,c in enumerate(raw_chunks):
        user_prompt=f"{llm_prompt}\n\nCHUNK {idx+1}:\n{c}"
        agent=create_summarizer_agent()
        steps=list(agent.stream({"messages":[{"role":"user","content":user_prompt}]},stream_mode="values"))
        partial=steps[-1]["messages"][-1].content
        partial=remove_think_tokens(partial)
        partial_summaries.append(partial)

    # combine partials into one summary
    combined="\n\n".join(partial_summaries)
    combine_prompt=f"Combine these partial summaries into a cohesive final summary:\n\n{combined}"
    agent2=create_summarizer_agent()
    steps2=list(agent2.stream({"messages":[{"role":"user","content":combine_prompt}]},stream_mode="values"))
    final_summary=steps2[-1]["messages"][-1].content
    final_summary=remove_think_tokens(final_summary)
    return final_summary.strip()

def compute_mews(hr=None, rr=None, sbp=None, temp=None, consciousness="alert"):
    """
    Uses MEWS (Modified Early Warning Score) to assess how sick a patient might be based on heart rate, 
    respiratory rate, blood pressure, temperature and alertness
    """
    score = 0

    # respiratory rate
    if rr is not None:
        if rr >= 30:
            score += 3
        elif 21 <= rr <= 29:
            score += 2
        elif 15 <= rr <= 20:
            score += 1
        elif 9 <= rr <= 14:
            score += 0
        elif rr < 9:
            score += 3

    # heart rate
    if hr is not None:
        if hr >= 130:
            score += 3
        elif 111 <= hr <= 129:
            score += 2
        elif 101 <= hr <= 110:
            score += 1
        elif 51 <= hr <= 100:
            score += 0
        elif 41 <= hr <= 50:
            score += 1
        elif hr < 40:
            score += 3

    # systolic blood pressure
    if sbp is not None:
        if sbp <= 70:
            score += 3
        elif 71 <= sbp <= 80:
            score += 2
        elif 81 <= sbp <= 100:
            score += 1
        elif 101 <= sbp <= 199:
            score += 0
        elif sbp >= 200:
            score += 2

    # temperature
    if temp is not None:
        if temp > 38.5:
            score += 2
        elif 35.0 <= temp <= 38.4:
            score += 0
        elif temp < 35.0:
            score += 2
    
    # consciousness (AVPU)
    if consciousness != "alert":
        score += 3
    return score

def determine_risk(mews):
    """
    Determines patient triage risk based on MEWS score.
    """
    if mews >= 5:           
        risk = "High"       # score: 5+
    elif mews >= 2:
        risk = "Moderate"   # score: 2-4
    else:
        risk = "Low"        # score: 0-1
    return risk

def process_patient_row(row):
    """
    Process a patient row to get triage risk.
    """
    note = str(row.get("assessment", "") + row.get("history_of_illness", ""))

    hr = pd.to_numeric(row.get("Heart rate"), errors='coerce')
    rr = pd.to_numeric(row.get("Respiratory rate"), errors='coerce')
    sbp = pd.to_numeric(row.get("Systolic Blood Pressure"), errors='coerce')
    temp = pd.to_numeric(row.get("Body temperature"), errors='coerce')
    consciousness = "alert" 

    mews_score = compute_mews(hr, rr, sbp, temp, consciousness)
    risk = determine_risk(mews_score)

    return {
        "patient_id": row["patient_id"],
        "patient_name": row["patient_name"],
        "MEWS Score": mews_score,
        "Risk": risk,
    }

def remove_trailing_punctuation(token):
    """
    Removes trailing punctuation (ie. periods and commas) from each token.
    """
    while token and re.search(r'[^\w\s\']', token[-1]):
        token = token[:-1]
    return token

def split_text(text):
    """
    Splits note line by line, extracts tokens/characters and collects sentence break info.
    """
    regex_match = r'[^\s\u200a\-\u2010-\u2015\u2212\uff0d]+'
    tokens, start_end_ranges, sentence_breaks = [], [], []
    start_idx = 0

    for sentence in text.split('\n'):
        # finds words using regex & remove punctuation from them
        words = [match.group(0) for match in re.finditer(regex_match, sentence)]
        processed_words = list(map(remove_trailing_punctuation, words))

        # tracks start & end positions of each token
        sentence_indices = [(match.start(), match.start() + len(token)) for match, token in
                            zip(re.finditer(regex_match, sentence), processed_words)]
        sentence_indices = [(start_idx + start, start_idx + end) for start, end in sentence_indices]
        
        # appends current sentence's tokens & position
        start_end_ranges.extend(sentence_indices)
        tokens.extend(processed_words)
        sentence_breaks.append(len(tokens))
        start_idx += len(sentence) + 1

    return tokens, start_end_ranges, sentence_breaks

def clean_word(word):
    """
    Strips non-alphanumeric characters, converts to lowercase and lemmatizes using spaCy ("running" -> "run).
    """
    word = re.sub(r'[^\w\s]', '', word) 
    word = re.sub(r'\s+', ' ', word)
    word = word.lower()
    if not word:
        return ""

    doc = nlp(word) # lemmatizes using spaCy
    if len(doc) == 0:
        return word

    return doc[0].lemma_

def preprocess_text_for_prediction(text, tokenizer, max_length):
    """
    Converts tokens into padded sequences, applies tokenizer lookups and returns everything needed for
    NER to make predictions.
    """
    # prepares text to be fed into NER model
    tokens, token_ranges, sentence_breaks = split_text(text)
    cleaned_tokens = [clean_word(token) for token in tokens]

    # converts each token into integer index using tokenizer
    token_indices = [tokenizer.word_index.get(token, 1) for token in cleaned_tokens]  # OOV token = 1
    
    # pads to match model's max input length
    X_padded = pad_sequences([token_indices], maxlen=max_length, padding='post', truncating='post')
    return tokens, token_indices, X_padded, token_ranges, sentence_breaks

def predict(text, model, index_to_label, acronyms_to_entities, max_length):
    """
    Uses NER on clinical notes with trained model.
    """
    # loads tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # preprocesses clinical notes
    tokens, _, X_padded, token_ranges, _ = preprocess_text_for_prediction(text, tokenizer, max_length)
    
    # predicts label for token
    y_pred = model.predict(X_padded)[0]
    pred_indices = np.argmax(y_pred, axis=1)

    predicted_labels = []
    predicted_entities = []

    # translates indices into readable entity labels
    for i in range(min(len(tokens), max_length)):
        label = index_to_label[str(pred_indices[i])]
        predicted_labels.append(label)

        if label.startswith('B-') or label.startswith('I-'):
            # maps short entity tags like "MED" to full ones like "MEDICATION"
            entity_acronym = label[2:]
            entity_type = acronyms_to_entities.get(entity_acronym, entity_acronym)
            predicted_entities.append(entity_type)
        else:
            predicted_entities.append('O') # 'O' means not an entity

    # builds list of entities w/ start & end character positions
    ents = []
    last_entity = None
    start_char = None

    for i, (token, entity, (start, end)) in enumerate(zip(tokens, predicted_entities, token_ranges)):
        if entity != 'O':
            if entity != last_entity:
                if last_entity and start_char is not None:
                    ents.append({
                        "start": start_char,
                        "end": token_ranges[i-1][1],
                        "label": last_entity
                    })
                start_char = start
                last_entity = entity
        else:
            if last_entity and start_char is not None:
                ents.append({
                    "start": start_char,
                    "end": token_ranges[i-1][1],
                    "label": last_entity
                })
                start_char = None
                last_entity = None

    # final entity after loop
    if last_entity and start_char is not None:
        ents.append({
            "start": start_char,
            "end": token_ranges[len(predicted_entities)-1][1],
            "label": last_entity
        })

    # creates html to visualize the entities
    ner_html = displacy.render(
        {"text": text, "ents": ents, "title": None},
        style="ent",
        manual=True,
        page=False
    )

    return tokens, predicted_labels, predicted_entities, token_ranges, ner_html

def apply_ner(text):
    """
    Runs NER prediction and group tokens into full entities for post processing.
    """
    tokens, labels, entities, ranges, ner_html = predict(
        text,
        model=ner_model,
        index_to_label=index_to_label,
        acronyms_to_entities=acronyms_to_entities,
        max_length=MAX_LENGTH
    )

    grouped_entities = []
    current_entity = None
    text_accumulator = []

    for token, label, entity, _ in zip(tokens, labels, entities, ranges):
        if label.startswith("B-"):
            # begins new entity
            if current_entity and text_accumulator:
                grouped_entities.append({
                    "text": " ".join(text_accumulator).strip("()"),
                    "type": current_entity
                })
            current_entity = entity
            text_accumulator = [token]
        elif label.startswith("I-") and current_entity == entity:
            # continues existing entity
            text_accumulator.append(token)
        else:
            # finishes the current entity
            if current_entity and text_accumulator:
                grouped_entities.append({
                    "text": " ".join(text_accumulator).strip("()"),
                    "type": current_entity
                })
            current_entity = None
            text_accumulator = []

    # gets final entity
    if current_entity and text_accumulator:
        grouped_entities.append({
            "text": " ".join(text_accumulator).strip("()"),
            "type": current_entity
        })

    return grouped_entities, ner_html


# Flask home page route: /
# Handles both GET (load page) & POST (search form submission)
@app.route("/", methods=["GET", "POST"])
def index():
    # if nurse searches up patient by name, redirect them to search results page
    if request.method == "POST":
        search_term = request.form.get("search_term", "")
        return redirect(url_for("search_results", query=search_term))

    try:
        with engine.connect() as conn:
            # clinical notes
            clinical_query = """
            SELECT 
                p.Id AS patient_id,
                CONCAT(p.FIRST, ' ', p.LAST) AS patient_name,
                cn.date AS note_date,
                cn.history_of_illness,
                cn.medications,
                cn.assessment
            FROM patients p
            JOIN (
                SELECT cn1.*
                FROM clinical_notes cn1
                JOIN (
                    SELECT patient_id, MAX(date) AS latest_date
                    FROM clinical_notes
                    GROUP BY patient_id
                ) latest_note
                ON cn1.patient_id = latest_note.patient_id AND cn1.date = latest_note.latest_date
            ) cn ON p.Id = cn.patient_id;
            """
            clinical_df = pd.read_sql(text(clinical_query), conn)

            # vitals
            desc_query = "SELECT DISTINCT DESCRIPTION FROM observations"
            descriptions = pd.read_sql(text(desc_query), conn)["DESCRIPTION"].tolist()

            # pivot table to show one row of observation per patient
            pivot_lines = []
            for desc in descriptions:
                line = "MAX(CASE WHEN o.DESCRIPTION = '{}' THEN o.VALUE END) AS `{}`".format(
                    desc.replace("'", "''"), desc.replace("`", "")
                )
                pivot_lines.append(line)
            pivot_columns = ",\n    ".join(pivot_lines)

            vitals_query = f"""
            SELECT
                p.Id AS patient_id,
                CONCAT(p.FIRST, ' ', p.LAST) AS patient_name,
                e.Id AS encounter_id,
                e.START AS encounter_start,
                {pivot_columns}
            FROM patients p
            JOIN (
                SELECT e1.PATIENT, e1.Id, e1.START
                FROM encounters e1
                JOIN (
                    SELECT PATIENT, MAX(START) AS max_start
                    FROM encounters
                    GROUP BY PATIENT
                ) latest_encounter
                ON e1.PATIENT = latest_encounter.PATIENT AND e1.START = latest_encounter.max_start
            ) e ON p.Id = e.PATIENT
            LEFT JOIN observations o ON e.Id = o.ENCOUNTER
            GROUP BY p.Id, patient_name, e.Id, e.START;
            """
            vitals_df = pd.read_sql(text(vitals_query), conn)

        # filters out dead patients
        if "Cause of Death [US Standard Certificate of Death]" in vitals_df.columns:
            vitals_df = vitals_df[vitals_df["Cause of Death [US Standard Certificate of Death]"].isna()]

        # merges clinical notes & vitals, drop duplicates
        merged = pd.merge(clinical_df, vitals_df, on=["patient_id", "patient_name"], how="inner")
        merged = merged.sort_values(by="encounter_start")
        merged = merged.drop_duplicates(subset="patient_id", keep="last")

        # compute MEWS & risk per patient
        patient_risks = []
        for _, row in merged.iterrows():
            result = process_patient_row(row)
            patient_risks.append(result)

        # sorts by MEWS score (from highest to lowest)
        patient_risks.sort(key=lambda x: x["MEWS Score"], reverse=True)

    except Exception as e:
        print(f"[ERROR]: {e}")
        patient_risks = []

    return render_template("index.html", patient_risks=patient_risks)

# Route: /search
# Searches for patients by first or last name
@app.route("/search")
def search_results():
    query=request.args.get("query","")
    search_sql=f"SELECT Id, FIRST, LAST, RACE FROM patients WHERE FIRST LIKE '%{query}%' OR LAST LIKE '%{query}%' LIMIT 20;"
    try:
        col_names, data=run_query_with_columns(search_sql)
    except Exception as e:
        col_names, data=["Error"],[[str(e)]]
    return render_template("search_results.html", query=query, col_names=col_names, results=data)

# Route: /patient/<patient_id>
# Shows patient's basic information and accepts nurse queries to generate SQL-based insights
@app.route("/patient/<patient_id>", methods=["GET","POST"])
def patient_detail(patient_id):
    columns=["Id","FIRST","MIDDLE","LAST","BIRTHDATE","DEATHDATE","GENDER","RACE","ETHNICITY","ADDRESS","CITY","STATE","ZIP","MARITAL"]
    patient_sql=f"SELECT * FROM patients WHERE Id='{patient_id}';"
    
    try:
        # turns results to a list of dicts
        patient_result=db.run(patient_sql)
        if isinstance(patient_result, list):
            parsed=patient_result
        else:
            s=patient_result.strip()
            if not s.startswith("["):
                s=f"[{s}]"
            try:
                parsed=ast.literal_eval(s)
            except:
                try:
                    parsed=eval(s,{"datetime":datetime})
                except:
                    parsed=None
        if parsed and isinstance(parsed,list) and len(parsed)>0:
            if isinstance(parsed[0],tuple):
                patient_details=[dict(zip(columns,x)) for x in parsed]
            elif isinstance(parsed[0],dict):
                patient_details=parsed
            else:
                patient_details=[{"Result":parsed}]
        else:
            patient_details=[{"Result":patient_result}]
    except Exception as e:
        patient_details=[{"error":str(e)}]

    nurse_query=""
    final_query=""
    agent_result=None

    # if nurse submits a question, generate & run query
    if request.method=="POST":
        nurse_query=request.form.get("nurse_query","")
        final_query,query_result=generate_valid_query(nurse_query,patient_id)
        if isinstance(query_result,str):
            agent_result=query_result
        else:
            agent_result=query_result 

    return render_template(
        "patient_detail.html",
        patient=patient_details,
        patient_id=patient_id,
        nurse_query=nurse_query,
        agent_result=agent_result,
        final_query=final_query
    )

# Route: /patient/<patient_id>/clinical_notes
# Lets nurse pick a date or year to summarize patient clinical notes using LLM & NER
@app.route("/patient/<patient_id>/clinical_notes", methods=["GET","POST"])
def patient_clinical_notes(patient_id):
    # gets all notes in a dropdown format
    sql_get_dates=sqlalchemy.text("""
        SELECT DISTINCT date
        FROM clinical_notes
        WHERE patient_id=:pid
        ORDER BY date ASC
    """)
    with engine.connect() as conn:
        dd=conn.execute(sql_get_dates,{"pid":patient_id}).fetchall()
    distinct_dates=[r[0] for r in dd]

    # gets all years in a dropdown format
    sql_get_years=sqlalchemy.text("""
        SELECT DISTINCT YEAR(date) as y
        FROM clinical_notes
        WHERE patient_id=:pid
        ORDER BY y ASC
    """)
    with engine.connect() as conn:
        dy=conn.execute(sql_get_years,{"pid":patient_id}).fetchall()
    distinct_years=[r[0] for r in dy]

    # values to display for notes
    single_note_full_text=None
    single_note_summary=None
    single_note_entities = []
    single_note_html = None
    selected_date_str=None
    year_summary=None
    year_summary_entities = []
    year_summary_html = None
    selected_year=None

    if request.method=="POST":
        # if a date was selected
        if "note_date" in request.form:
            selected_date_str=request.form["note_date"]
            if selected_date_str:
                sql_one=sqlalchemy.text("""
                    SELECT date,chief_complaint,history_of_illness,social_history,
                           allergies,medications,assessment
                    FROM clinical_notes
                    WHERE patient_id=:pid
                      AND date=:chosen_date
                    LIMIT 1
                """)
                with engine.connect() as conn:
                    row=conn.execute(sql_one,{"pid":patient_id,"chosen_date":selected_date_str}).fetchone()
                if row:
                    # extracts note text
                    nd=row[0]
                    c=row[1] or ""
                    hh=row[2] or ""
                    ss=row[3] or ""
                    aa=row[4] or ""
                    mm=row[5] or ""
                    asmt=row[6] or ""
                    single_note_full_text=(
                        f"DATE: {nd}\n"
                        f"Chief Complaint: {c}\n"
                        f"History of Illness: {hh}\n"
                        f"Social History: {ss}\n"
                        f"Allergies: {aa}\n"
                        f"Medications: {mm}\n"
                        f"Assessment: {asmt}\n"
                    )
                    
                    # generates summary & NER tags
                    single_note_summary=summarize_text(
                        single_note_full_text,
                        llm_prompt="Summarize this single clinical note. Focus on major changes in illness, medications, etc."
                    )
                    single_note_entities, single_note_html = apply_ner(single_note_full_text)

        # if a year is selected to summarize all notes in that year
        if "year_for_summary" in request.form:
            selected_year=request.form["year_for_summary"]
            if selected_year:
                sql_yr=sqlalchemy.text("""
                    SELECT date,chief_complaint,history_of_illness,social_history,
                           allergies,medications,assessment
                    FROM clinical_notes
                    WHERE patient_id=:pid
                      AND YEAR(date)=:yearval
                    ORDER BY date ASC
                """)
                with engine.connect() as conn:
                    rows=conn.execute(sql_yr,{"pid":patient_id,"yearval":selected_year}).fetchall()
                if rows:
                    # combines all notes from the year into one large string
                    big_text=""
                    for r in rows:
                        nd=r[0]
                        c=r[1] or ""
                        h=r[2] or ""
                        s=r[3] or ""
                        a=r[4] or ""
                        m=r[5] or ""
                        asmt=r[6] or ""
                        big_text+=(
                            f"DATE: {nd}\n"
                            f"Chief Complaint: {c}\n"
                            f"History of Illness: {h}\n"
                            f"Social History: {s}\n"
                            f"Allergies: {a}\n"
                            f"Medications: {m}\n"
                            f"Assessment: {asmt}\n\n"
                        )

                    # summarizes & applies NER to text
                    year_summary=summarize_text(
                        big_text,
                        llm_prompt=f"Summarize all clinical notes from year {selected_year}, focusing on major changes in illness, medications, allergies, etc. Start will telling the number of visits the patient had in {selected_year}."
                    )
                    year_summary_entities, year_summary_html = apply_ner(year_summary)

    return render_template(
        "patient_clinical_notes.html",
        patient_id=patient_id,
        distinct_dates=distinct_dates,
        distinct_years=distinct_years,
        single_note_full_text=single_note_full_text,
        single_note_summary=single_note_summary,
        single_note_entities=single_note_entities,
        single_note_html=single_note_html,
        selected_date_str=selected_date_str,
        year_summary=year_summary,
        year_summary_entities=year_summary_entities,
        year_summary_html=year_summary_html,
        selected_year=selected_year
    )

if __name__=="__main__":
    app.run(debug=True)