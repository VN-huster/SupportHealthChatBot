import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from typing import Optional, Tuple
from chains.slot_memory import SlotMemory
from chains.prompt import CHAT_PROMPT
from configs.params import ModelParams


# from model import language_model

model_config = ModelParams()

chain: ConversationChain


def initial_chain():
    """
        Init multiple key to avoid limited request (3 requests per min)
    """
    llm0 = ChatOpenAI(temperature=model_config.temperature, openai_api_key="")
    llm1 = ChatOpenAI(temperature=model_config.temperature, openai_api_key="")
    llm2 = ChatOpenAI(temperature=model_config.temperature, openai_api_key="")
    llm3 = ChatOpenAI(temperature=model_config.temperature, openai_api_key="")
    llm4 = ChatOpenAI(temperature=model_config.temperature, openai_api_key="")

    memory0 = SlotMemory(llm=llm0)
    memory1 = SlotMemory(llm=llm1)
    memory2 = SlotMemory(llm=llm2)
    memory3 = SlotMemory(llm=llm3)
    memory4 = SlotMemory(llm=llm4)

    """
    Using these script if you want to use your LLM
    """
    # llm = language_model()
    # memory = SlotMemory(llm=llm)

    global chain
    chain = ConversationChain(llm=llm1, memory=memory2, prompt=CHAT_PROMPT)


def clear_session():
    initial_chain()
    return [], []


def slot_format(slot_dict):
    result = f"main symptom: {slot_dict['main_symptom']}\nrelative_symptom: {slot_dict['relative_symptom']}\nsymptom_time: {slot_dict['symptom_time']}\n"
    return result


def predict(command, history: Optional[Tuple[str, str]]):
    history = history or []
    response = chain.run(input=command)
    current_slot = chain.memory.current_slots
    history.append((command, response))
    return history , history, '', slot_format(current_slot)


if __name__ == "__main__":
    title = """
    # Dialogue Slot Filling Demo
    """
    with gr.Blocks() as demo:
        gr.Markdown(title)

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot()
                user_input = gr.Textbox(show_label=False, placeholder="Input...", container=False)
                with gr.Row():
                    submitBtn = gr.Button("🚀Submit", variant="primary")
                    emptyBtn = gr.Button("🧹Clear History")
            slot_show = gr.Textbox(label="current_slot", lines=20, interactive=False, scale=1)

        initial_chain()
        state = gr.State([])

        submitBtn.click(fn=predict, inputs=[user_input, state], outputs=[chatbot, state, user_input, slot_show])
        emptyBtn.click(fn=clear_session, inputs=[], outputs=[chatbot, state])

    demo.queue().launch(share=False, inbrowser=True, server_name="0.0.0.0", server_port=8000)

# initial_chain()
# history = []
# while True:
#     a = input("Patient: ")
#     print(predict(a, history))