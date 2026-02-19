from llama_cpp import Llama

class LLMGenerator:
    def __init__(self, model_path="tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf", n_ctx=2048):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False
        )
    
    def generate(self, prompt, max_tokens=512, temperature=0.2, top_p=0.2):
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False,  # Don't include prompt in output
            stop=["</s>", "<|im_end|>", "User:", "Question:"]  # Stop tokens
        )
        
        # Extract the generated text
        response = output['choices'][0]['text'].strip()
        
        return response
    
    def generate_with_context(self, query, context_docs, max_tokens=256):
        prompt = self.create_augmented_prompt(query, context_docs)
        response = self.generate(prompt, max_tokens=max_tokens)
        
        return response
    
    def create_augmented_prompt(self, query, context_docs):
        context_text = ""
        for i, doc in enumerate(context_docs, 1):
            context_text += f"\n{doc['text']}\n"
        
        # Format with ChatML template for TinyLlama
        prompt = f"""<|im_start|>system
You are a helpful assistant. Answer questions based on the information provided in the documents below.
<|im_start|>context
{context_text}
<|im_end|>
<|im_start|>user

Question: {query}

<|im_start|>assistant
Answer: 
"""
#         prompt = f"""
# Question: {query}
# Top Documents: {context_text}
# <|im_start|>assistant
# """
        return prompt


def main():
    model_path = "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
    generator = LLMGenerator(model_path)
    
    test_query = "What causes squirrels to lose fur?"
    test_docs = [
        {
            'id': 0,
            'text': "Squirrels may lose fur due to mange, caused by mites. This condition leads to hair loss and itchy skin. Treatment involves veterinary care."
        },
        {
            'id': 1,
            'text': "Fungal infections can also cause fur loss in squirrels. These infections spread through contact with contaminated surfaces."
        },
        {
            'id': 2,
            'text': "Nutritional deficiencies may result in poor coat quality and fur loss in wild animals including squirrels."
        }
    ]
    
    print("\n=== Testing LLM Generation ===")
    print(f"Query: {test_query}\n")
    
    response = generator.generate_with_context(test_query, test_docs, max_tokens=200)
    
    print(f"Generated Response:\n{response}\n")


if __name__ == "__main__":
    main()

