from random import randrange
from functools import partial
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from huggingface_hub import login
from tqdm import tqdm 
from data_processing_ketod import ketod_data_processing

class RAGate_Llama:
    def __init__(self, use_knowledge=False, load_in_4bit=True, bnb_4_bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, max_memory=22960):
        """
            Configure model quantization using bitsandbytes to speed up training and inference
            :param model_name: model name of finetuned llama model
            :param use_knowledge: whether to use knowledge for classifying augmentation necessity
            :param load_in_4bit: Load the model in 4-bit precision mode
            :param bnb_4_bit_use_double_quant: nested quantization for 4-bit model
            :param bnb_4bit_quant_type: The quantization type for 4-bit model
            :param bnb_4bit_compute_dtype: The compute dtype for 4-bit model
        """
        model_list = ["XiWangEric/IfAug_classification_context_system_syn_res_ner_input_llama2_7b",
                  "XiWangEric/IfAug_classification_context_system_syn_res_ner_know_input_llama2_7b"]
        self.use_knowledge = use_knowledge
        if self.use_knowledge:
            model_name = model_list[1]
        else:
            model_name = model_list[0]
        self.model_name = model_name 
        self.load_in_4bit = load_in_4bit
        self.bnb_4_bit_use_double_quant = bnb_4_bit_use_double_quant
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.max_memory = max_memory
        self.create_bnb_config()
        self.load_model()


    def create_bnb_config(self):
        """
            Configure model quantization using bitsandbytes to speed up training and inference
            :param load_in_4bit: Load the model in 4-bit precision mode
            :param bnb_4_bit_use_double_quant: nested quantization for 4-bit model
            :param bnb_4bit_quant_type: The quantization type for 4-bit model
            :param bnb_4bit_compute_dtype: The compute dtype for 4-bit model
        """

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4_bit_use_double_quant=self.bnb_4_bit_use_double_quant,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype
        )


    def load_model(self):
        """
            Load the model and tokenizer
            :param model_name: Huggingface model name

        """
        n_gpus = torch.cuda.device_count()
        max_memory = f'{self.max_memory}MB'
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                    quantization_config=self.bnb_config,
                                                    max_memory= {i: max_memory for i in range(n_gpus)})
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def create_prompt_format(self, input, instruction, knowledge=None):
        """
            Create a formatted prompt template for a prompt in the instruction dataset
        """
        # Initialize static strings for the prompt template
        INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        INSTRUCTION_KEY = "### Instruction:"
        INPUT_KEY = "Input:"
        RESPONSE_KEY = "### Response:"
        # Combine a prompt with the static strings
        basic = f"{INTRO_BLURB}"
        instruction = f"{INSTRUCTION_KEY}\n{instruction}"
        if self.use_knowledge:
            input_text = f"{INPUT_KEY}\n{input}\n Retrieved Knowledge:{knowledge}"
        else:
            input_text = f"{INPUT_KEY}\n{input}"
        response = f"{RESPONSE_KEY}"

        # Create a list of prompt template elements
        parts = [part for part in [basic, instruction, input_text, response] if part]
        # Combine the prompt template elements into a single string
        formatted_prompt = "\n\n".join(parts)
        return formatted_prompt
    
    
    def predict(self, input, knowledge=None):
        """
            Predict the output of the model
            :param input: The input to the model
            :param knowledge: The retrieved knowledge
        """
        if self.use_knowledge:
            instruction = "Analyse the conversational context so far. Generate an appropriate system response. \
                Consider the invovled entites and retrieved knowledge. Estimate if augmenting the response with \
                    retrieved knowledge is helpful with an output of ""True"" or ""False"" only."
        else:
            instruction = 'Analyse the conversational context so far. Generate an appropriate response. Consider the invovled entites. Estimate if augmenting the response with external knowledge is helpful with an output of "True" or "False" only.'
        formatted_input = self.create_prompt_format(input=input, instruction=instruction, knowledge=knowledge) if self.use_knowledge else self.create_prompt_format(input=input, instruction=instruction)
        # print(formatted_input)
        input_ids = self.tokenizer.encode(formatted_input, return_tensors="pt").cuda()
        # Generate output and limit the length of the output
        output = self.model.generate(input_ids, max_length=(input_ids.shape[1] + 20)) 
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        output_text = output_text[len(formatted_input):].lower()
        if 'false' in output_text:
            return 0
        elif 'true' in output_text:
            return 1
        else:
            print(f"Error: {output_text}")
            return 0


if __name__ == "__main__":
    login(token='your_token_here')
    use_knowledge = True
    # Load the model
    ragate_llama = RAGate_Llama()
    print("Model loaded successfully")

    # Load the dataset
    root_dir = "data storage dir/"
    output_dir = "output dir/"
    dataset_dir = root_dir + "test_data_with_snippets.json"
    ketod_data = ketod_data_processing(dataset_dir)
    dialogue_ids, turn_idx, contexts, contexts_and_system_responses, retrieved_knowledge, labels = ketod_data.process_data()
    predictions = {}
    for idx in tqdm(range(len(contexts))):
        input = contexts[idx]
        if use_knowledge:
            knowledge = retrieved_knowledge[idx]
        prediction_idx = dialogue_ids[idx] + '_' + turn_idx[idx]
        prediction = ragate_llama.predict(input=input, knowledge=knowledge) if use_knowledge else ragate_llama.predict(input=input)
        predictions[prediction_idx] = prediction

    ketod_data.add_prediction(predictions)
    ketod_data.save_data(output_dir + "test_data_with_snippets_enrich_pred_llm_know.json")