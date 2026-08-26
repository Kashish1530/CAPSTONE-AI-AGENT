from sentence_transformers import SentenceTransformer
import numpy as np

print("loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("model loaded!")

sentences = [
    "The cat sat on the mat.",
    "A dog was resting on the rug.",
    "Cats and dogs are common household pets.",
    "My kitten loves to sleep all day.",
    "The puppy chased its tail in circles.",
    "I love eating pizza on weekends.",
    "Pasta is my favorite Italian dish.",
    "She ordered a burger and fries for lunch.",
    "Sushi is a popular Japanese cuisine.",
    "Ice cream melts quickly in summer heat.",
    "The stock market crashed yesterday.",
    "Investors are worried about inflation.",
    "The company's shares rose by 10 percent.",
    "Bitcoin prices fluctuated wildly this week.",
    "The central bank raised interest rates.",
    "It rained heavily in the city last night.",
    "The weather forecast predicts snow tomorrow.",
    "Summer temperatures are rising every year.",
    "A storm is approaching the coastline.",
    "The sun was shining brightly this morning.",
    "The football match ended in a draw.",
    "She scored the winning goal in the final.",
    "The basketball team won the championship.",
    "He broke the world record in swimming.",
    "The marathon runner finished in record time.",
    "Quantum computers use qubits instead of bits.",
    "Artificial intelligence is transforming industries.",
    "The new smartphone has a faster processor.",
    "Software updates fixed several security bugs.",
    "Robots are increasingly used in manufacturing.",
    "The novel explores themes of love and loss.",
    "She wrote a poem about the changing seasons.",
    "The movie received critical acclaim worldwide.",
    "The museum exhibit features ancient artifacts.",
    "The orchestra performed a symphony last night.",
    "The patient was diagnosed with a mild fever.",
    "Doctors recommend regular exercise for health.",
    "The hospital opened a new pediatric wing.",
    "Vaccines have reduced the spread of disease.",
    "A balanced diet improves overall wellbeing.",
    "The election results were announced today.",
    "The senator proposed a new tax reform bill.",
    "Voters lined up early at polling stations.",
    "The president addressed the nation on TV.",
    "Congress passed the infrastructure bill.",
    "The spacecraft landed successfully on Mars.",
    "Astronauts conducted experiments on the ISS.",
    "The telescope captured images of a distant galaxy.",
    "Scientists discovered a new exoplanet.",
    "The rocket launch was delayed due to weather.",
]

embeddings = model.encode(sentences)

query = "My pet dog loves playing outside."
query_embedding = model.encode(query)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = []
for i, s in enumerate(sentences):
    score = cosine_similarity(query_embedding, embeddings[i])
    scores.append((score, s))

scores.sort(reverse=True, key=lambda x: x[0])

print(f"Query: {query}\n")
print("Top 5 most similar:")
for score, sentence in scores[:5]:
    print(f"{score:.4f}  |  {sentence}")

print("\nBottom 3 least similar:")
for score, sentence in scores[-3:]:
    print(f"{score:.4f}  |  {sentence}")
print ("done encoding!")