#!/usr/bin/env python3
# To run this file with the correct interpreter (.venv):
# Option 1: Activate venv first: source ../.venv/bin/activate && python day2_ex.py
# Option 2: Use venv directly: ../.venv/bin/python day2_ex.py
# Option 3: In VS Code/Cursor, select the .venv interpreter from the command palette

from scraper import fetch_website_contents
from IPython.display import Markdown, display
from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"
ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key='ollama')

# See how this function creates exactly the format above

def messages_for(website):
    system_prompt = """
    You are an assistant that analyzes the contents of a website,
    and provides a short summary, ignoring text that might be navigation related.
    Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
    """

    user_prompt_prefix = """
    Here are the contents of a website.
    Provide a short summary of this website.
    If it includes news or announcements, then summarize these too.
    """

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_prefix + website}
    ]

def summarize(url):
    website = fetch_website_contents(url)
    response = ollama.chat.completions.create(model="llama3.2", messages=messages_for(website))
    return response.choices[0].message.content
    
# A function to display this nicely in the output, using markdown
def display_summary(url):
    summary = summarize(url)
    print(summary)


if __name__ == "__main__":
    url = "https://edwarddonner.com"
    display_summary(url)