from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import warnings
import re

warnings.filterwarnings("ignore")


def indochat(input, CONFIG_DATA):
    try:
        print("Start GPT Model...")
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(CONFIG_DATA["model_gpt"])
        model = AutoModelForCausalLM.from_pretrained(CONFIG_DATA["model_gpt"])

        model.config.pad_token_id = 50256
        model.config.eos_token_id = 50256
        # Create a chatbot pipeline
        chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

        # Usage generator
        user_input = input
        bot_response = chatbot(user_input, max_length=200, num_return_sequences=5)

        generated_text = bot_response[0]["generated_text"]

        if "Assistant:" in generated_text:
            # Find the indices of both assistant variations
            index_normal = generated_text.find("Assistant:")
            index_additional = generated_text.find(
                "Assistant: Tentu, ini beberapa lagi:"
            )

            # Determine which variation comes first
            if index_normal == -1 or (
                index_additional != -1 and index_additional < index_normal
            ):
                index = index_additional
            else:
                index = index_normal

            # Extract the text after the respective assistant variation
            modified_text = generated_text[
                index
                + len(
                    "Assistant: Tentu, ini beberapa lagi:"
                    if index == index_additional
                    else "Assistant:"
                ) :
            ]

            # Remove the specific phrase "Assistant: Tentu, ini beberapa lagi:" from the modified text
            modified_text = modified_text.replace(
                "User: Beri saya lebih banyak. Assistant: Tentu, ini beberapa lagi:", ""
            )

            modified_text = modified_text.replace(
                "User: Lebih banyak lagi. Assistant: Tentu, ini beberapa lagi:", ""
            )
            return modified_text
        else:
            # Handle the case where "Assistant:" is not found
            return generated_text

    except Exception as e:
        return f"Error in Model GPT: {e}"
