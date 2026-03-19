***

# Task 1: The "Glorified Autocomplete" (GPT-2 Pretraining)

**Pretraining**. Your objective is to build a scaled-down, decoder-only Transformer model (similar to the GPT-2 architecture) from scratch and train it to predict the next character in a sequence. 

Essentially, you are building a highly complex autocomplete function trained exclusively on the works of William Shakespeare.

## 📋 Task 1 Overview
You will implement a sub-word level causal language model using PyTorch (only Pytorch). By the end of this task, your model should take a seed prompt and continuously predict the next token to generate novel, Shakespeare-esque text. **The model architecture should be based on GPT-2 but you are free modify it**.

### Core Deliverables:

**All deliverable code should python files, other formats like jupyter notebooks are not allowed**.

1.  **Data Loader:** Process and tokenize the `tinyshakespeare` dataset into inputs (`x`) and targets (`y` shifted by one token).
2.  **The Architecture:** Implement a Decoder-only Transformer block (Masked Self-Attention, Feed-Forward Networks, and Positional/Token Embeddings).
3.  **The Pretraining Loop:** Write a training loop to optimize the model using Cross-Entropy Loss over multiple epochs.
4.  **The Autocomplete:** A simple generation function that samples from your model's probability distribution to generate text.

## 🗂️ The Dataset
We will be using the **Tiny Shakespeare** dataset, containing roughly 40,000 lines of Shakespeare's plays. 
* **Download Link:** [tinyshakespeare/input.txt](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

## 🗺️ Suggested Milestones

* **Step 1: Load the dataset**
    * Use the given dataloader or write your own.
* **Step 2: The Core Mechanism (Masked Self-Attention)**
    * Implement a single Masked Self-Attention head.
* **Step 3: Scaling Up (Multi-Head Attention)**
    * Expand to Multi-Head Attention and build a complete Transformer block (adding LayerNorm and Feed-Forward layers).
* **Step 4: Putting it Together**
    * Stack multiple blocks, add your embeddings, and finalize the model architecture.
* **Step 5: Let it Train**
    * Run your pretraining loop until the validation loss stops improving. Generate a block of text (e.g., 500 characters) to see your "autocomplete" in action.

## 📤 Submission Guidelines

1.  **Fork** this repository.
2.  Develop your solution in your repo.
3.  Ensure your code is clean, modular, and easy to follow.
4.  Create a **README** in your repo to include:
    * Instructions on how to run your pretraining script.
    * Instructions on how to run your generation/autocomplete script.
    * A sample of the text your model generated.
    * A brief explanation of your model's hyperparameters (number of layers, heads, embedding size, etc.).
5.  Submit the link to your forked repository via the induction submission form.

## ⚖️ Evaluation Criteria

For Task 1, we are evaluating you on the following:
- **Architecture Accuracy**: Not just the output we want a sensible architecture as well.
- **Code Quality**: Is the code organized, documented and hand written (No vibe coding allowed).
- **Conceptual Grasp**: Can you explain the math and logic behind your code during your induction interview?
- **Learning Proof**: Does the training loss go down, and does the output look vaguely like English words/structures? (We aren't looking for a massive, state-of-the-art model—just proof that your architecture learns).


## 📚 Helpful Resources
* [Let's build GPT: from scratch, in code, spelled out by Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) (Highly recommended—it covers this exact task perfectly!)
* [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
* [Attention Is All You Need (Original Paper)](https://arxiv.org/abs/1706.03762)
* [Attention is all you need - Video explanation](https://www.youtube.com/watch?v=bCz4OMemCcA)
***
