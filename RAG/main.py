import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import BGEM3FlagModel

# --- Setup and Embedding Generation ---
print("Step 1: Initializing model and generating embeddings...")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

sentences = [
  "Artificial intelligence, or AI, is a field of computer science.", "It focuses on creating machines that can think and learn like humans.",
  "Machine learning is a subfield of AI.", "Deep learning is a more advanced type of machine learning.",
  "AI is used in many everyday applications, such as smartphone assistants.", "Virtual assistants like Siri and Alexa are examples of AI.",
  "AI powers recommendation systems on platforms like Netflix and Amazon.", "Self-driving cars rely heavily on AI to navigate roads safely.",
  "In healthcare, AI helps in diagnosing diseases and developing new drugs.", "Natural language processing, or NLP, allows computers to understand human language.",
  "Computer vision enables machines to 'see' and interpret images and videos.", "Robotics often combines AI with mechanical engineering.",
  "AI can analyze vast amounts of data much faster than humans.", "The goal of general AI is to create machines with human-level intelligence.",
  "Narrow AI is designed for specific tasks, like playing chess.", "AI is changing the way we work and live.",
  "There are ethical considerations surrounding the use of AI.", "Concerns about data privacy and algorithmic bias are common.",
  "AI is being used to combat climate change through predictive modeling.", "Generative AI can create new content, such as images, text, and music.",
  "Large language models (LLMs) are a type of generative AI.", "ChatGPT is a well-known example of an LLM.",
  "AI is used in finance for fraud detection and algorithmic trading.", "Predictive analytics, powered by AI, helps businesses make better decisions.",
  "AI in education can personalize learning experiences for students.", "The development of AI has a long history, dating back to the 1950s.",
  "Turing's test is a measure of a machine's ability to exhibit intelligent behavior.", "Reinforcement learning involves training AI through a system of rewards and penalties.",
  "Many experts believe AI will have a profound impact on the future.", "The future of AI is full of both potential and challenges.",
  "Space exploration is the scientific research of space.",
  "NASA is the United States' space agency.",
  "The first man to walk on the Moon was Neil Armstrong in 1969.",
  "The Hubble Space Telescope has revolutionized astronomy.",
  "The International Space Station (ISS) is a laboratory in orbit.",
  "Satellites orbit Earth for various purposes, such as communication and navigation.",
  "Mars is often called the 'Red Planet'.",
  "The Perseverance rover is currently exploring the surface of Mars.",
  "Zero gravity is a unique experience for astronauts.",
  "Space travel can have a significant impact on the human body.",
  "The Apollo 11 spacecraft took Armstrong and Aldrin to the Moon.",
  "The Milky Way is our galaxy.",
  "Black holes are regions of space with extremely strong gravity.",
  "The search for extraterrestrial life is an important goal of space exploration.",
  "The James Webb Space Telescope is the successor to Hubble.",
  "Rockets are the primary means of leaving Earth's atmosphere.",
  "Low Earth orbit (LEO) is the region closest to Earth in space.",
  "Astronauts train for years before going on a mission.",
  "Comets are celestial bodies made of ice, rock, and dust.",
  "A solar eclipse occurs when the Moon passes between the Earth and the Sun.",
  "Empty space is not a true vacuum but contains particles and magnetic fields.",
  "Space exploration has led to numerous technological innovations.",
  "The Big Bang theory explains the origin of the universe.",
  "Galaxies are groupings of stars, gas, and dust.",
  "Jupiter is the largest planet in our solar system.",
  "Asteroids are small, rocky bodies that orbit the Sun.",
  "The Voyager 1 mission is the spacecraft farthest from Earth.",
  "Earth is not a perfect sphere but a geoid.",
  "Earth's atmosphere protects us from cosmic radiation and space debris.",
  "Space exploration inspires human imagination and curiosity."
]
query = "How big is Jupiter?"

all_texts = sentences + [query]
all_embeddings = model.encode(all_texts)['dense_vecs']

# --- DATA CLEANING ---
# Convert to float32 first
all_embeddings = all_embeddings.astype('float32')
# **FIX 1**: Replace any NaN/inf values with finite numbers.
# This function replaces NaN with 0.0, positive infinity with a large number,
# and negative infinity with a small (large negative) number.
np.nan_to_num(all_embeddings, copy=False, nan=0.0, posinf=None, neginf=None)


# --- Cosine Similarity Calculation ---
print("\nStep 2: Calculating cosine similarity...")

sentence_embeddings = all_embeddings[:-1]
query_embedding = all_embeddings[-1:]

similarity_scores = cosine_similarity(query_embedding, sentence_embeddings)[0]
results = list(zip(sentences, similarity_scores))
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

print("\n--- Similarity Scores for the Query ---")
print(f"Query: '{query}'\n")
for sentence, score in sorted_results:
    print(f"Score: {score:.4f}  |  Sentence: {sentence}")
print("-" * 50)


# --- t-SNE and Data Preparation for Plotly ---
print("\nStep 3: Running t-SNE and preparing data for interactive plot...")
# This avoids the part of the sklearn code that was producing the warnings.
tsne = TSNE(n_components=2, perplexity=20, random_state=42, init='random', learning_rate='auto')
tsne_results = tsne.fit_transform(all_embeddings)

df = pd.DataFrame()
df['text'] = all_texts
df['tsne_x'] = tsne_results[:, 0]
df['tsne_y'] = tsne_results[:, 1]
df['type'] = ['Sentence'] * len(sentences) + ['Query']

# --- Create and show the interactive plot ---
print("\nStep 4: Generating interactive plot...")
fig = px.scatter(
    df,
    x='tsne_x',
    y='tsne_y',
    color='type',
    hover_name='text',
    color_discrete_map={'Query': 'red', 'Sentence': 'blue'},
    title='Interactive t-SNE Visualization of Sentences'
)
fig.update_traces(
    marker=dict(size=12, symbol='star'),
    selector=dict(mode='markers', type='scatter', customdata=df[df['type']=='Query'].index)
)
fig.show()