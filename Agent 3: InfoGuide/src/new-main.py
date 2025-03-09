import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

# Streamlit UI
st.title("MTSS Copilot - Anomaly Prediction Assistant")

# Sidebar for user role selection
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

# Inject custom CSS for chat UI enhancements
st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        overflow-y: auto;
        max-height: 500px;
    }
    .user-bubble {
        background-color: #73000a;
        color: white;
        border: 1px solid #73000a;
        border-radius: 15px;
        padding: 10px 20px;
        max-width: 60%;
        margin: 10px 0;
        text-align: left;
        float: right;
        clear: both;
    }
    .ai-bubble {
        background-color: #f0f0f0;
        color: black;
        border-radius: 15px;
        padding: 10px 20px;
        max-width: 60%;
        margin: 10px 0;
        text-align: left;
        float: left;
        clear: both;
    }
    .user-bubble img {
        position: absolute;
        top: -10px;
        right: -50px;
        width: 30px;
        height: 30px;
    }
    .ai-bubble img {
        position: absolute;
        top: -10px;
        left: -50px;
        width: 50px;
        height: 50px;
    }
    .chat-container > div {
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Display chat messages with styled bubbles
st.write('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='user-bubble'><img src='https://cdn-icons-png.flaticon.com/512/747/747545.png' />{msg['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='ai-bubble'><img src='https://cdn-icons-png.freepik.com/512/6783/6783338.png' /><strong>MTSS Copilot</strong>: {msg['content']}</div>",
            unsafe_allow_html=True
        )

st.write('</div>', unsafe_allow_html=True)
