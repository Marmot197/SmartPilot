import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from random import choice
from assets.DataUtils import AssetLoader
from copilots.Memory_Utils import Knowledge_Representation, Retr, Symbolic_Model
from copilots.Agents import LLM
import pandas as pd

def load_anomaly_prediction_model():
    model_checkpoint = 'final_best_model'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    df = pd.read_excel('./LLM_FT_dataset.csv')
    unique_labels = df['predicted_label'].unique().tolist()
    id2label = {i: label for i, label in enumerate(unique_labels)}
    label2id = {label: i for i, label in enumerate(unique_labels)}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=len(unique_labels), id2label=id2label, label2id=label2id
    )
    return tokenizer, model, id2label

def get_anomaly_prediction(tokenizer, model, id2label, user_query, time_series_data):
    new_text_inputs = [f"{series} {user_query}" for series in time_series_data]
    tokenized_inputs = tokenizer(new_text_inputs, padding=True, truncation=True, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenized_inputs = {key: value.to(device) for key, value in tokenized_inputs.items()}
    model.to(device)
    with torch.no_grad():
        logits = model(**tokenized_inputs).logits
    predicted_labels = [id2label[label.item()] for label in torch.argmax(logits, axis=1)]
    return predicted_labels

st.title("MTSS Copilot - Anomaly Prediction Assistant")

st.sidebar.title("🛠 User Simulation")
users_and_queries = AssetLoader.get_queries()
user_roles = list(users_and_queries.keys())
selected_role = st.sidebar.selectbox("Select User Role", user_roles)
user_query = st.sidebar.selectbox("Select Query", users_and_queries[selected_role])

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.write("**Chat with the AI:**")
user_input = st.chat_input("Ask a question about anomaly detection or documentation...")

tokenizer, model, id2label = load_anomaly_prediction_model()
if user_input or st.sidebar.button("Run Simulation"):
    st.session_state["messages"].append({"role": "user", "content": user_input or user_query})
    if selected_role == 'Anomaly Prediction and Sensor Values':
        time_series_data = ["[663. 463. 500.]"]
        predicted_labels = get_anomaly_prediction(tokenizer, model, id2label, user_query, time_series_data)
        response = f"Predicted anomaly labels: {', '.join(predicted_labels)}"
    else:
        data = Knowledge_Representation.organize_data(AssetLoader.read_data())
        context = Retr.retrieve_context(data, user_query, symb_model=Symbolic_Model(), top_k=1)[0]
        system_template = AssetLoader.get_templates()[selected_role]
        llm = LLM()
        llm.set_prompt(system_template, user_query, context)
        response = llm.respond_to_prompt()
    st.session_state["messages"].append({"role": "assistant", "content": response})

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])
