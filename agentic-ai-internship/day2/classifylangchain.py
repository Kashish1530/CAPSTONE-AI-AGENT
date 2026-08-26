from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="llama3.1", temperature=0)

emails = [
    ("Your invoice #4521 is overdue. Please pay within 5 days.", "Billing"),
    ("Hey, are we still on for lunch tomorrow?", "Personal"),
    ("Your package has shipped and will arrive Friday.", "Shipping"),
    ("URGENT: Your account password was changed.", "Security"),
    ("Happy birthday! Hope you have a great day.", "Personal"),
]

categories = "Billing, Shipping, Personal, Security"

correct = 0

for text, true_label in emails:
    prompt = f"Classify this email into exactly one category ({categories}). Reply with ONLY the category name.\n\nEmail: {text}"
    
    response = llm.invoke([HumanMessage(content=prompt)])
    predicted = response.content.strip()
    
    is_correct = predicted.lower() == true_label.lower()
    if is_correct:
        correct += 1
    
    print(f"True: {true_label:10} | Predicted: {predicted} | {'success' if is_correct else 'fail'}")

print(f"\nAccuracy: {correct}/{len(emails)}")