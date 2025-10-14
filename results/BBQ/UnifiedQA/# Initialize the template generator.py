# Initialize the template generator
from BBQ_prompt_template import BBQPromptTemplate, load_bbq_data

template_generator = BBQPromptTemplate()

# Load your BBQ data
questions = load_bbq_data('/path/to/your/Age.jsonl')

# Generate different prompt types
standard_prompt = template_generator.generate_prompt(questions[0], 'standard')
bias_aware_prompt = template_generator.generate_prompt(questions[0], 'bias_aware')

# For vLLM inference
vllm_prompts = prepare_vllm_prompts(questions, 'bias_aware')

# For HuggingFace
hf_dataset = prepare_hf_dataset(questions, 'chain_of_thought')