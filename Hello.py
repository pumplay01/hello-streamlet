from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
import random
import shutil
import torch

st.set_page_config(page_title="HOSPITAL STANDARD AI", page_icon="🏥")
st.title("HOSPITAL STANDARD AI RECOMMEND")
st.write("Frist time we need preparation system a few minutes")

tokenizer = AutoTokenizer.from_pretrained("pumplay01/JCI7-GPT-DOCX", bos_token='<|endoftext|>',
                                          eos_token='<|endoftext|>', pad_token='<|pad|>')

# Download the model weights (this is where the actual download happens)
with st.spinner('Downloading model installation...'):
    model = AutoModelForCausalLM.from_pretrained("pumplay01/JCI7-GPT-DOCX")

# Once the download is complete, display a success message
st.success("Model downloaded successfully!")

# Add input fields for user parameters
inputx = st.text_input("Input Text:", "Type a word at here")
max_length_num = st.number_input("Max Length:", min_value=1, value=310)
num_samples = st.number_input("Number of Answer:", min_value=1, value=10)

if st.button("Generate Text"):
    # Call your code with user-provided parameters
    generated_results = []

    generated = tokenizer(f"{inputx}", return_tensors="pt").input_ids
    sample_outputs = model.generate(generated,
                                    do_sample=True,
                                    top_k=500,
                                    max_length=max_length_num,
                                    top_p=0.95,
                                    temperature=1.5,
                                    num_return_sequences=num_samples)

    for sample_output in sample_outputs:
        generated_text = tokenizer.decode(sample_output, skip_special_tokens=True)
        generated_results.append(generated_text)

    # Shuffle the generated results
    random.shuffle(generated_results)

    # Combine all generated results into one text with spaces between lines
    combined_results = ' '.join(generated_results)

    # Get the width of the terminal window
    terminal_width = shutil.get_terminal_size().columns

    # Define the line length to be 80% of the terminal width
    line_length = int(terminal_width * 1.2)

    # Split the text into lines
    lines = [combined_results[i:i+line_length] for i in range(0, len(combined_results), line_length)]

    # Print the combined results
    st.write(combined_results, unsafe_allow_html=True)
