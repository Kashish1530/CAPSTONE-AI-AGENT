import ollama
import json
from pydantic import BaseModel

class Invoice(BaseModel):
    name: str
    company: str
    amount: float
    date: str

invoices = [
    "Hey, this is Rohan from BrightTech Solutions. You owe us $450 by March 25, 2026.",
    "Client: Priya Sharma | Firm: GreenLeaf Co | Total: 1200.50 | Date: 2026-01-14",
    "Name: Meera Iyer  Company: SkyNet Systems  Amt Due: 675.00  Issued: Jan 30 2026",
    "This is regarding Ayesha Khan's account with CloudNine Ltd - Total: $95 - Date 12 Feb 2026",
    "Aditya Bhatt, owner of Skyline Traders, invoiced $99.00 on 2026-01-25",
    "billed to anita desai (WORKSPACE INC) ... amount: $89.99 ... due 03/02/2026",
    "Invoice from Kunal Verma, TechNova Pvt Ltd. Amount payable: $3,000. Dated: 5th Feb 2026.",
    "Please find attached charges for Arjun Rao (Orbit Media) — $150 — invoiced 2026-02-18",
    "Client Neha Kapoor from Studio Alpha owes $220.75, due on Feb 28, 2026.",
    "INVOICE: Vikram Singh / RedBrick Constructions / 5,600 dollars / 2026/03/01",
    "Sameer Joshi, PixelWorks — amount owed $410.00 — billing date 2026-03-10",
    "Invoice details: Kavya Nair (BlueWave Tech) $780 due 2026-02-20",
    "From Rahul Chawla at NextGen Solutions - please pay $2,150 by 2026-03-15",
    "Divya Menon / Quantum Labs / Bill amount 340.50 / date: March 3 2026",
    "Payment request: Isha Malhotra, EverGreen Foods, $560, dated 2026-02-05",   

]

for text in invoices:
    prompt = f"""Read this invoice text and reply with ONLY JSON like this:
{{"name": "...", "company": "...", "amount": 0.0, "date": "..."}}

Text: {text}"""

    response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': prompt}])
    reply = response['message']['content'].strip()
    print("Model said:", reply)
    
    if reply.startswith("```"):
        reply = reply.replace("```json", "").replace("```", "").strip()

    data = json.loads(reply)          # turn text into real data
    invoice = Invoice(**data)         # check it matches our rules
    print("Extracted:", invoice)
    print("-" * 40)
    