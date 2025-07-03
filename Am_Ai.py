import torch
import torch.nn as nn
import subprocess
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
import random
import time


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_size):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.Transformer(embed_size, num_heads, num_layers, num_layers, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output


class MemoryAugmentedTransformerModel(TransformerModel):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_size, memory_size):
        super(MemoryAugmentedTransformerModel, self).__init__(vocab_size, embed_size, num_heads, num_layers, hidden_size)
        self.memory = torch.zeros(memory_size, hidden_size)  # Initialize external memory
        self.memory_size = memory_size

    def store_memory(self, memory_data):
        """Store new memory data."""
        self.memory = memory_data

    def retrieve_memory(self):
        """Retrieve memory for processing."""
        return self.memory


class ModelParser:
    def __init__(self):
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token

    def parse_task(self, input_text, model_type="t5"):
        if model_type == "t5":
            input_ids = self.t5_tokenizer.encode(input_text, return_tensors='pt', padding=True, truncation=True)
            output_ids = self.t5_model.generate(input_ids, max_length=50, attention_mask=input_ids.ne(self.t5_tokenizer.pad_token_id))
            output = self.t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        elif model_type == "gpt2":
            input_ids = self.gpt2_tokenizer.encode(input_text, return_tensors='pt', padding=True, truncation=True)
            output_ids = self.gpt2_model.generate(input_ids, max_length=50, attention_mask=input_ids.ne(self.gpt2_tokenizer.pad_token_id))
            output = self.gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            raise ValueError("Model type must be either 't5' or 'gpt2'.")
        return output


def execute_code(code):
    try:
        result = subprocess.run(['python', '-c', code], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error executing code: {e}"


class DeepLearningModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_size, memory_size):
        super(DeepLearningModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.Transformer(embed_size, num_heads, num_layers, num_layers, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.memory = torch.zeros(memory_size, hidden_size)  # External memory
        self.memory_size = memory_size
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output

    def train_model(self, src, tgt):
        """Train the model with new data."""
        self.optimizer.zero_grad()
        output = self(src, tgt)
        loss = nn.CrossEntropyLoss()(output.view(-1, output.size(-1)), tgt.view(-1))
        loss.backward()
        self.optimizer.step()
        return loss.item()


class TaskManager:
    def __init__(self):
        self.parser = ModelParser()
        self.deep_model = DeepLearningModel(vocab_size=10000, embed_size=512, num_heads=8, num_layers=6, hidden_size=512, memory_size=100)

    def handle_task(self, input_text, model_type="t5"):
        start_time = time.time()

        # پردازش ورودی با استفاده از مدل‌های T5 یا GPT-2
        parsed_task = self.parser.parse_task(input_text, model_type)

        # آموزش مدل یادگیری تدریجی (DeepLearningModel)
        loss = self.deep_model.train_model(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[1, 2, 3, 4]]))

        execution_time = time.time() - start_time

        # اجرای کد پردازش شده
        output = execute_code(parsed_task)

        return output, loss, execution_time


def run_program():
    task_manager = TaskManager()
    while True:
        task = input("Please enter a command (type 'exit' to quit): ")
        if task.lower() == "exit":
            print("Exiting the program.")
            break


        model_type = random.choice(["t5", "gpt2"])


        output, loss, execution_time = task_manager.handle_task(task, model_type)

        print(f"\nExecution Time: {execution_time:.4f} seconds")
        print(f"Output: {output}")
        print(f"Training Loss: {loss}")


if __name__ == "__main__":
    run_program()
