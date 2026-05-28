import json
import numpy
import warnings
import os
warnings.filterwarnings('ignore')
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv('API.env')
import anthropic
# Load the banking Q&A dataset from JSON file
with open("dataset.json", "r") as file:
    data = json.load(file)

# Extract questions and answers into separate lists for easier processing
questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

# Load the sentence transformer embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode all questions into vector embeddings for similarity matching
print("Embedding dataset...")
question_embeddings = model.encode(questions)


# Function to find the most relevant banking question based on user query
def find_best_match(user_query):
    # Encode the user's query into an embedding
    query_embedding = model.encode(user_query)
    
    # Manually compute cosine similarity between query and all questions
    # Cosine similarity = dot product / (magnitude of vector1 * magnitude of vector2)
    similarities = []
    for q_embedding in question_embeddings:
        dot_product = numpy.dot(query_embedding, q_embedding)
        magnitude1 = numpy.linalg.norm(query_embedding)
        magnitude2 = numpy.linalg.norm(q_embedding)
        cosine_sim = dot_product / (magnitude1 * magnitude2)
        similarities.append(cosine_sim)
    
    # Find the index of the highest similarity score
    best_index = numpy.argmax(similarities)
    best_similarity = float(similarities[best_index])
    
    # Return the matched question, its answer, and the similarity score
    return questions[best_index], answers[best_index], best_similarity


# Function to generate an answer using Claude API with the matched context
def get_answer(user_query, context):
    # Create an Anthropic API client (reads ANTHROPIC_API_KEY from environment automatically)
    client = anthropic.Anthropic()
    
    # Try to call the Claude API, fallback to context if API fails
    try:
        message = client.messages.create(
            model="claude-opus-4-1",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": f"You are a helpful banking assistant. Here is some context: {context}. Here is the user's question: {user_query}. Please answer clearly."
                }
            ]
        )
        return message.content[0].text.strip()
    except:
        # If API fails, return the matched context directly
        return context


# Main function that runs the chatbot interactive loop
def main():
    # Print welcome banner
    print("\n" + "="*60)
    print("Welcome to the Banking RAG Chatbot")
    print("Type 'quit' or 'exit' to end the conversation")
    print("="*60)
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
    
    # Main conversation loop
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Skip empty input
        if not user_input:
            continue
        
        # Exit on quit or exit command
        if user_input.lower() in ["quit", "exit"]:
            print("Thank you for using the Banking Chatbot. Goodbye!")
            break
        
        # Find the best matching question from dataset
        matched_question, matched_answer, similarity_score = find_best_match(user_input)
        
        # Print the matched context in square brackets
        print(f"[Matched: {matched_question} (Similarity: {similarity_score:.2f})]")
        
        # Get AI-generated answer using the matched context
        bot_response = get_answer(user_input, matched_answer)
        
        # Print the bot's response
        print(f"Bot: {bot_response}\n")


# Entry point of the program
if __name__ == "__main__":
    main()
