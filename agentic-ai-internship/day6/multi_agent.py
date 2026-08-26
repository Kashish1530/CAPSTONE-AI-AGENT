import ollama

def researcher_agent(topic):
    prompt = f"""You are a Researcher. Your job is to gather 3-4 key facts about the topic below.
Just list facts, no writing style, no fluff.

Topic: {topic}"""
    response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def writer_agent(topic, research_notes):
    prompt = f"""You are a Writer. Using ONLY the research notes below, write a short, engaging paragraph (3-4 sentences) about the topic.

Topic: {topic}

Research notes:
{research_notes}"""
    response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def supervisor(topic):
    print(f"SUPERVISOR: Starting task on topic '{topic}'\n")

    print("SUPERVISOR: Assigning to Researcher...")
    research_notes = researcher_agent(topic)
    print("RESEARCHER OUTPUT:")
    print(research_notes)
    print()

    print("SUPERVISOR: Assigning to Writer...")
    final_output = writer_agent(topic, research_notes)
    print("WRITER OUTPUT:")
    print(final_output)
    print()

    print("SUPERVISOR: Task complete.")
    return final_output

supervisor("The benefits of renewable energy")

