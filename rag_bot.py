## Now we will make a streamlit chatbot app
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import os
from argparse import ArgumentParser
from gemini_utility import load_genai_model

# Step 1: Extract Transcript
def get_youtube_transcript(video_url):
    video_id = video_url.split("/")[-1].split("?")[0]  # Extract video ID from URL
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    final_transcript = "\n".join([t['text'] for t in transcript])
    return final_transcript

# Step 2: Chunk the text
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Step 3: Embed the chunks
def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings

# Step 4: Store in FAISS
def create_vector_store(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


# Step 5: Question answering

def ask_question(question, index, chunks, model, top_k=3):
    # Step 1: Embed the question
    q_emb = model.encode([question])
    D, I = index.search(q_emb, top_k)

    # Step 2: Get top matching transcript chunks
    context = "\n\n".join([chunks[i] for i in I[0]])

    # Step 3: Call Gemini
    gemini_model = load_genai_model()  # Load the generative AI model

    prompt = f"""Answer the question based only on the following video transcript context. 
                If the answer is not available, say "I don't know".

            Context:
            {context}

            Question: {question}
            """
    response = gemini_model.generate_content(prompt)
    return response.text.strip()



parser = ArgumentParser(description="Get YouTube video transcript") # this is used to parse command line arguments which means it will take the YouTube video URL as an argument
parser.add_argument("url", type=str, help="YouTube video URL") # this is used to get the YouTube video URL from command line arguments
args = parser.parse_args() # .parse_args() parses the command line arguments and returns them as an object
url = args.url

text = get_youtube_transcript(url)
chunks = chunk_text(text)
# print(chunks)
embeddings = embed_chunks(chunks)
# print(len(embeddings))
# print(embeddings[:2])
index = create_vector_store(embeddings)
# print(index.ntotal)  # Print the number of vectors in the index
qn = "What are autoencoders?"
model = SentenceTransformer('all-MiniLM-L6-v2')
answer = ask_question(qn, index, chunks, model)
# print(answer)
# print("\n" + "_"*60)
# print("🤖 Gemini's Answer:")
# print("_"*60)
# print(answer)

from rich.console import Console
from rich.markdown import Markdown

console = Console()
answer = ask_question(qn, index, chunks, model)

console.rule("[bold green]🤖 Gemini's Answer")
console.print(Markdown(answer))
console.rule()




# # --- Example usage ---
# video_id = "687zEGODmHA"
# text = get_transcript(video_id)
# chunks = chunk_text(text)
# embeddings, model = embed_chunks(chunks)
# index = create_vector_store(embeddings)

# question = "What is the main idea of the video?"
# answer = ask_question(question, index, chunks, model)
# print(answer)
