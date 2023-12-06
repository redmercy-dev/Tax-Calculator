import streamlit as st
import openai
import os
from llama_index.agent import OpenAIAssistantAgent
import tempfile

# Streamlit app
def main():
    st.set_page_config(page_title="Tax Provider", layout="wide")

    # Custom CSS to inject into the Streamlit app
    custom_css = """
    <style>
        .css-qrbaxs {background-color: #f0f2f6;}
        .stButton>button {background-color: #0d6efd; color: white;}
        .stTextInput>div>div>input {color: blue;}
        .stTextArea>div>div>textarea {background-color: #e9ecef;}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    st.sidebar.title("Configuration")
    st.sidebar.markdown("## Provide Your Details Here")

    api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    os.environ['OPENAI_API_KEY'] = api_key

    uploaded_file = st.sidebar.file_uploader("Upload a text file", type=["txt"])

    # Initialize or update the conversation in session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    def add_to_conversation(participant, text):
        st.session_state.conversation.append((participant, text))

    def format_conversation():
        return "\n".join([f"{participant}: {text}" for participant, text in st.session_state.conversation])

    # Use Streamlit's session state to store the agent
    if 'global_agent' not in st.session_state:
        st.session_state.global_agent = None

    @st.cache_resource
    def initialize_agent(file_path):
        return OpenAIAssistantAgent.from_new(
            name="Tax Provider",
            instructions="You are GPT a Tax Rate Matcher's. Your primary objective is to provide the exact tax value for products based on user queries. It interprets product descriptions, categorizes them, and references a text file to extract the precise tax rate for the specified product. The GPT adheres strictly to the data in the text file, providing tax rates only from this source. You have to accurately categorizes products and finds the matching tax rate(a single product), asking for clarification if needed. The GPT adapts its communication style for technical or casual interactions and updates responses in line with changes to the text file.Make sure to add the category from the text file for the user ",
            openai_tools=[{"type": "retrieval"}],
            instructions_prefix="You provide tax rates for products.Please provide the tax rate for the user product.You need to provide only the tax rate for the user product not a list of similar products. Each output should include the product name,and the corresponding tax rate (import tax rate and local tax rate) Make sure to add the category from the text file so that the user can check the accuracy",
            files=[file_path],
            verbose=True,
        )

    if uploaded_file is not None and st.session_state.global_agent is None:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name

        st.sidebar.success(f'Uploaded File: {uploaded_file.name}')
        st.session_state.global_agent = initialize_agent(temp_file_path)

    input_text = st.text_input("Input:")
    execute_button = st.button("Execute")

    if execute_button and st.session_state.global_agent:
        try:
            agent_response = st.session_state.global_agent.chat(input_text)
            response_text = agent_response.response if not isinstance(agent_response, str) else agent_response
            st.text_area("Output:", value=response_text, height=300)
            add_to_conversation("User", input_text)
            add_to_conversation("Agent", response_text)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Download button in the sidebar
    if st.sidebar.download_button(label="Download Conversation", data=format_conversation(), file_name="conversation.txt", mime="text/plain"):
        st.sidebar.success("Conversation downloaded!")

if __name__ == "__main__":
    main()
