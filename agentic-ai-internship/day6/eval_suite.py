import ollama

test_cases = [
    {"question": "What is 15 + 27?", "expected": "42"},
    {"question": "What is the capital of France?", "expected": "Paris"},
    {"question": "What is 8 times 9?", "expected": "72"},
    {"question": "What color do you get mixing blue and yellow?", "expected": "green"},
    {"question": "How many continents are there?", "expected": "7"},
    {"question": "What is the boiling point of water in Celsius?", "expected": "100"},
    {"question": "Who wrote Romeo and Juliet?", "expected": "Shakespeare"},
    {"question": "What is 100 divided by 4?", "expected": "25"},
    {"question": "What planet is known as the Red Planet?", "expected": "Mars"},
    {"question": "What is the chemical symbol for water?", "expected": "H2O"},
    {"question": "How many days are in a leap year?", "expected": "366"},
    {"question": "What is 12 squared?", "expected": "144"},
    {"question": "What is the largest ocean on Earth?", "expected": "Pacific"},
    {"question": "What language is primarily spoken in Brazil?", "expected": "Portuguese"},
    {"question": "What is 50% of 200?", "expected": "100"},
    {"question": "Who painted the Mona Lisa?", "expected": "Vinci"},
    {"question": "What is the square root of 81?", "expected": "9"},
    {"question": "What gas do plants absorb from the air?", "expected": "carbon dioxide"},
    {"question": "How many sides does a hexagon have?", "expected": "6"},
    {"question": "What is the freezing point of water in Celsius?", "expected": "0"},
    {"question": "What is 9 times 9?", "expected": "81"},
    {"question": "What is the capital of Japan?", "expected": "Tokyo"},
    {"question": "What is 45 minus 18?", "expected": "27"},
    {"question": "How many legs does a spider have?", "expected": "8"},
    {"question": "What is the currency used in the USA?", "expected": "dollar"},
]


def ask_model(question):
    response = ollama.chat(
        model="llama3.1",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response["message"]["content"]


def score_answer(answer, expected):
    return expected.lower() in answer.lower()


correct = 0
results = []

print("=" * 70)
print("LLAMA 3.1 EVALUATION")
print("=" * 70)
print()

for i, case in enumerate(test_cases, 1):

    # Ask the model
    answer = ask_model(case["question"])

    # Check answer
    is_correct = score_answer(
        answer,
        case["expected"]
    )

    
    results.append({
        "question": case["question"],
        "expected": case["expected"],
        "answer": answer,
        "correct": is_correct
    })

    if is_correct:
        correct += 1
        status = "CORRECT"
    else:
        status = "WRONG"

    print(f"{i}. {status}")
    print(f"   Question: {case['question']}")
    print(f"   Expected: {case['expected']}")
    print(f"   Model Answer: {answer}")
    print("-" * 70)


accuracy = (correct / len(test_cases)) * 100


print()
print("=" * 70)
print("FINAL EVALUATION RESULT")
print("=" * 70)

print(f"Correct Answers : {correct}")
print(f"Total Questions : {len(test_cases)}")
print(f"Wrong Answers   : {len(test_cases) - correct}")
print(f"Accuracy        : {accuracy:.1f}%")

print("=" * 70)
