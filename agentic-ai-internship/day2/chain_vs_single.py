import ollama
problems = [
    ("A store had 120 apples. they sold 35% in the morning, then sold 28 more in the afternoon. how many apples are left?", 50),
    ("A has 2 mangoes and B has 50 mangoes. so together how many mangoes they have?", 52),
]
single_correct = 0
chained_correct = 0
# approah 1
for problem, correct_answer in problems:

    answer1 = ollama.chat(
        model='llama3.1',
        messages=[ 
            {
                'role': 'user','content': problem
            }
        ]
    )
    print("==='SINGLE SHOT'===", answer1['message']['content'])
    if str(correct_answer) in answer1['message']['content']:
        single_correct += 1

#approach 2
    chained_prompt = problem + "solve this step by step, show each calculation clearly, then give the final answer."
    answer2 = ollama.chat(
        model = 'llama3.1',
        messages = [
            {
                'role': 'user', 'content': chained_prompt
            }
        ]
    )
    print("\n===step by step===", answer2['message']['content'])
    if str(correct_answer) in answer2['message']['content']:
        chained_correct += 1

    print("-" * 40)

print("SINGLE SHOT accuracy:", single_correct, "out of", len(problems))
print("step by step accuracy:", chained_correct, "out of", len(problems))


