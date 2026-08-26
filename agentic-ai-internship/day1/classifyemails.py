import ollama

emails = [
    {"text": "Your invoice #4521 is overdue. Please pay within 5 days.", "true_label": "Billing"},
    {"text": "Hey, are we still on for lunch tomorrow?", "true_label": "Personal"},
    {"text": "Your package has shipped and will arrive Friday.", "true_label": "Shipping"},
    {"text": "URGENT: Your account password was changed. If this wasn't you, click here.", "true_label": "Security"},
    {"text": "Reminder: your subscription renews next week for $9.99.", "true_label": "Billing"},
    {"text": "Can you send me the report before end of day?", "true_label": "Personal"},
    {"text": "Track your order — it's out for delivery today.", "true_label": "Shipping"},
    {"text": "We noticed a login from a new device. Was this you?", "true_label": "Security"},
    {"text": "Your credit card on file will be charged $49 tomorrow.", "true_label": "Billing"},
    {"text": "Happy birthday! Hope you have a great day.", "true_label": "Personal"},
    {"text": "Your item has been delivered to your front door.", "true_label": "Shipping"},
    {"text": "Two-factor authentication code: 583920.", "true_label": "Security"},
    {"text": "Your refund of $23.50 has been processed.", "true_label": "Billing"},
    {"text": "Let's catch up this weekend, it's been a while!", "true_label": "Personal"},
    {"text": "Shipping delay: your order will now arrive next Tuesday.", "true_label": "Shipping"},
    {"text": "Suspicious activity detected on your account.", "true_label": "Security"},
    {"text": "Your monthly statement is now available to view.", "true_label": "Billing"},
    {"text": "Dinner at my place on Saturday?", "true_label": "Personal"},
    {"text": "Your tracking number is 1Z999AA10123456784.", "true_label": "Shipping"},
    {"text": "Please verify your email to secure your account.", "true_label": "Security"},
    {"text": "Your refund for the returned shipment has beedn processed.", "true_label": "Billing"},
    {"text": "We updated the privacy policy-please review the changes.", "true_label": "Security"},

]

categories = ["Billing", "Shipping", "Personal", "Security"]

correct = 0

for i, email in enumerate(emails, 1):
    prompt = f"""Classify this email into exactly one of these categories: {', '.join(categories)}.
Reply with ONLY the category name, nothing else.

Email: "{email['text']}"

Category:"""

    response = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.0}
    )

    predicted = response['message']['content'].strip()
    is_correct = predicted.lower() == email['true_label'].lower()
    if is_correct:
        correct += 1

    status = "success" if is_correct else "fail"
    print(f"{i}. {status} True: {email['true_label']:10} | Predicted: {predicted}")

accuracy = (correct / len(emails)) * 100
print(f"\nAccuracy: {correct}/{len(emails)} = {accuracy:.1f}%")