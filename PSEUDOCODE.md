# Banking RAG Chatbot - Code Explanation (Pseudocode)

## Overview
This document explains the entire Banking RAG (Retrieval-Augmented Generation) Chatbot code line by line so readers can understand the logic and flow.

---

## SECTION 1: IMPORTS AND INITIALIZATION

```
// Import necessary libraries for the chatbot to work
import json                              // For reading/writing JSON files
import numpy                             // For mathematical operations (dot product, norm)
import warnings                          // For controlling warning messages
import os                                // For OS operations (not directly used, but imported for env)
warnings.filterwarnings('ignore')        // Suppress all warning messages for cleaner output

from sentence_transformers import SentenceTransformer  // For converting text to embeddings
from dotenv import load_dotenv           // For loading environment variables from files
load_dotenv('API.env')                   // Load API keys from API.env file
import anthropic                         // For connecting to Claude API
```

**What's happening here?**
- We're importing all the tools needed for the chatbot
- We suppress warnings to keep the console clean
- We load environment variables (API keys) from API.env so they're not hardcoded

---

## SECTION 2: LOAD THE BANKING DATASET

```
// Read the JSON file containing banking Q&A pairs
with open("dataset.json", "r") as file:
    data = json.load(file)               // Parse JSON and store in 'data' variable

// Extract questions into a separate list for easier access
questions = [item["question"] for item in data]

// Extract answers into a separate list for easier access
answers = [item["answer"] for item in data]
```

**What's happening here?**
- We open dataset.json which contains 10 banking Q&A pairs
- We split them into two lists: one for questions, one for answers
- This makes it easier to work with them separately later

**Example:**
```
questions = ["What is KYC?", "What is a credit score?", ...]
answers = ["KYC stands for Know Your Customer...", "A credit score is a three-digit number...", ...]
```

---

## SECTION 3: LOAD THE EMBEDDING MODEL

```
// Print a status message to the user
print("Loading embedding model...")

// Load the SentenceTransformer model
// This model converts text into numerical vectors (embeddings)
// "all-MiniLM-L6-v2" is a lightweight, fast model suitable for similarity matching
model = SentenceTransformer("all-MiniLM-L6-v2")

// Print a status message
print("Embedding dataset...")

// Convert all banking questions into embeddings (numerical vectors)
// This allows us to compare user queries with questions mathematically
question_embeddings = model.encode(questions)
// Result: question_embeddings is a 2D array with 10 rows and 384 dimensions
```

**What's happening here?**
- We load a pre-trained embedding model that converts text to numbers
- Each question becomes a 384-dimensional vector
- These vectors capture the semantic meaning of the questions
- We store these embeddings so we don't have to recalculate them for each user query

---

## SECTION 4: FIND BEST MATCH FUNCTION

```
def find_best_match(user_query):
    """
    This function finds the most relevant banking question from the dataset
    that matches what the user is asking about
    """
    
    // Step 1: Convert user's question into an embedding (numerical vector)
    query_embedding = model.encode(user_query)
    
    // Step 2: Calculate similarity between user query and all stored questions
    similarities = []  // Create empty list to store similarity scores
    
    for q_embedding in question_embeddings:
        // Calculate dot product (basic multiplication of vectors)
        dot_product = numpy.dot(query_embedding, q_embedding)
        
        // Calculate the magnitude (length) of user query vector
        magnitude1 = numpy.linalg.norm(query_embedding)
        
        // Calculate the magnitude (length) of stored question vector
        magnitude2 = numpy.linalg.norm(q_embedding)
        
        // Cosine similarity formula: dot_product / (magnitude1 * magnitude2)
        // This gives us a score between -1 and 1 (closer to 1 is more similar)
        cosine_sim = dot_product / (magnitude1 * magnitude2)
        
        // Add this similarity score to our list
        similarities.append(cosine_sim)
    
    // Step 3: Find the question with the highest similarity score
    best_index = numpy.argmax(similarities)  // Get the index of highest similarity
    best_similarity = float(similarities[best_index])  // Get the actual similarity score
    
    // Step 4: Return three things:
    // - The most similar question from our dataset
    // - The answer to that question
    // - How confident we are (similarity score)
    return questions[best_index], answers[best_index], best_similarity
```

**What's happening here?**
- This is the RAG (Retrieval) part - finding relevant information
- We convert user input to an embedding
- We compare it with all stored question embeddings using cosine similarity
- We return the most similar question and its answer

**Example:**
```
User asks: "Tell me about UPI"
Query embedding: [0.2, 0.5, 0.8, ...]
Compare with 10 stored question embeddings
Similarity scores: [0.15, 0.19, 0.22, 0.89, 0.14, 0.18, 0.21, 0.16, 0.25, 0.13]
Best match index: 3 (with similarity 0.89)
Returns: ("What is UPI and how does it work?", "UPI stands for...", 0.89)
```

---

## SECTION 5: GET ANSWER FUNCTION

```
def get_answer(user_query, context):
    """
    This function generates an intelligent answer using Claude API
    It takes the user's question and the matched banking context
    """
    
    // Create a client to connect to Anthropic's Claude API
    client = anthropic.Anthropic()
    // The API key is automatically read from ANTHROPIC_API_KEY environment variable
    
    // Step 1: Try to get an answer from Claude API
    try:
        // Call Claude API with specific parameters
        message = client.messages.create(
            model="claude-opus-4-1",      // Use Opus model (latest available)
            max_tokens=300,               // Limit response to 300 tokens
            messages=[
                {
                    "role": "user",
                    // Craft a prompt that includes context and user question
                    "content": f"You are a helpful banking assistant. Here is some context: {context}. Here is the user's question: {user_query}. Please answer clearly."
                }
            ]
        )
        // Extract and clean the response text
        return message.content[0].text.strip()
    
    except:
        // Step 2: If API fails (no credits, network error, etc.)
        // Just return the matched answer from our dataset
        // This is a graceful fallback so the chatbot still works
        return context
```

**What's happening here?**
- This is the AG (Augmented Generation) part - creating context-aware responses
- We call Claude API with the matched context from our database
- Claude enhances the answer with its knowledge
- If API fails, we fallback to the original answer from our dataset

**Example Flow:**
```
User: "Tell me about UPI"
matched_answer: "UPI stands for Unified Payments Interface..."
Prompt sent to Claude: "You are a helpful banking assistant. Here is some context: 
UPI stands for Unified Payments Interface... 
Here is the user's question: Tell me about UPI. Please answer clearly."
Claude's response: "UPI (Unified Payments Interface) is a revolutionary payment system..."
Return: Enhanced answer from Claude
```

---

## SECTION 6: MAIN CHATBOT FUNCTION

```
def main():
    """
    This is the main function that runs the chatbot in a loop
    It handles all user interactions
    """
    
    // Step 1: Display welcome banner to the user
    print("\n" + "="*60)
    print("Welcome to the Banking RAG Chatbot")
    print("Type 'quit' or 'exit' to end the conversation")
    print("="*60)
    
    // Display example questions users can ask
    print("\nExample questions you can ask:")
    print("  • What is KYC?")
    print("  • How do I open a savings account?")
    print("  • What documents are needed to open a bank account?")
    print("  • What is a fixed deposit?")
    print("  • Tell me about current accounts")
    print("  • How does net banking work?")
    print("  • What is a credit score?")
    print("  • Explain loan EMI")
    print("  • What is UPI?")
    print("  • What is a nominee in a bank account?")
    print("="*60 + "\n")
    
    // Step 2: Start the main conversation loop
    while True:
        // Get user input with "You: " prompt
        user_input = input("You: ").strip()  // .strip() removes extra spaces
        
        // Skip if user just pressed Enter (empty input)
        if not user_input:
            continue
        
        // Check if user wants to exit
        if user_input.lower() in ["quit", "exit"]:
            print("Thank you for using the Banking Chatbot. Goodbye!")
            break  // Exit the loop and end the program
        
        // Step 3: Find the most relevant banking Q&A for user's query
        matched_question, matched_answer, similarity_score = find_best_match(user_input)
        
        // Step 4: Show what question was matched (with confidence score)
        print(f"[Matched: {matched_question} (Similarity: {similarity_score:.2f})]")
        // .2f formats the similarity score to 2 decimal places
        
        // Step 5: Get an intelligent answer using Claude
        bot_response = get_answer(user_input, matched_answer)
        
        // Step 6: Display the bot's response to the user
        print(f"Bot: {bot_response}\n")
        // \n adds a blank line for readability
```

**What's happening here?**
- This is the main conversation loop
- It displays a welcome message with example questions
- It continuously takes user input
- For each input, it finds the best match and generates an answer
- It exits when user types "quit" or "exit"

**Example Interaction:**
```
============================================================
Welcome to the Banking RAG Chatbot
Type 'quit' or 'exit' to end the conversation
============================================================

Example questions you can ask:
  • What is KYC?
  • How do I open a savings account?
  ... (more examples)
============================================================

You: What is a credit score?
[Matched: What is a credit score and how does it affect you? (Similarity: 0.87)]
Bot: A credit score is a three-digit number that represents your creditworthiness...

You: quit
Thank you for using the Banking Chatbot. Goodbye!
```

---

## SECTION 7: PROGRAM ENTRY POINT

```
// This checks if the script is being run directly (not imported as a module)
if __name__ == "__main__":
    // Call the main() function to start the chatbot
    main()
```

**What's happening here?**
- This ensures main() only runs when we execute the script directly
- If someone imports this file as a module, main() won't automatically run

---

## COMPLETE FLOW DIAGRAM

```
User starts the chatbot
    ↓
Main loop asks: "You: "
    ↓
User types a question
    ↓
find_best_match(user_question)
    ├─ Convert question to embedding
    ├─ Calculate similarity with all stored questions
    ├─ Find highest similarity match
    └─ Return matched question, answer, and score
    ↓
Display: "[Matched: ... (Similarity: 0.XX)]"
    ↓
get_answer(user_question, matched_answer)
    ├─ Try to call Claude API
    ├─ Claude enhances the answer with AI knowledge
    ├─ If API fails, return original dataset answer
    └─ Return the final response
    ↓
Display: "Bot: [answer]"
    ↓
Loop back to ask for next question
    ↓
User types "quit" or "exit"
    ↓
Program ends
```

---

## KEY CONCEPTS EXPLAINED

### 1. **Embeddings**
- Text converted to 384-dimensional numerical vectors
- Similar text has similar vectors
- Allows mathematical comparison of text meaning

### 2. **Cosine Similarity**
- Measures angle between two vectors
- Score between -1 and 1 (1 = identical, 0 = unrelated)
- Formula: (A·B) / (|A| × |B|)

### 3. **RAG (Retrieval-Augmented Generation)**
- RETRIEVAL: Find relevant information from database (find_best_match)
- AUGMENTATION: Add context to the query
- GENERATION: Use Claude to create intelligent response (get_answer)

### 4. **Fallback Mechanism**
- If Claude API fails, chatbot still works using dataset answers
- Makes the chatbot resilient to API issues

### 5. **Environment Variables**
- API keys stored in API.env (not in code)
- Loaded at startup
- Keeps sensitive data secure

---

## HOW TO USE THIS EXPLANATION

1. **For Beginners**: Start from Section 1 and read sequentially
2. **For Understanding Flow**: Read "COMPLETE FLOW DIAGRAM" section
3. **For Specific Features**: Jump to the relevant section
4. **For Key Concepts**: Read "KEY CONCEPTS EXPLAINED" section

---

## DEPENDENCIES

- **json**: Built-in library, no installation needed
- **numpy**: Mathematical operations (similarity calculation)
- **sentence-transformers**: Text to embeddings conversion
- **python-dotenv**: Load API keys from environment
- **anthropic**: Connect to Claude API

Install with: `pip install -r requirements.txt`

---

## SECURITY NOTES

- API keys read from environment variables (never hardcoded)
- API.env file should never be committed to Git
- .gitignore protects API.env from being uploaded
- Users should copy API.env.example to API.env and add their own keys

---

**End of Pseudocode Explanation**
