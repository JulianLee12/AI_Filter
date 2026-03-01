import re
import time
import serial
import tiktoken
from typing import Dict
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
# ======================================================
# CONFIG
# ======================================================


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SERIAL_PORT = "/dev/cu.usbmodem101"
BAUD_RATE = 9600
MAX_COMPLETION_TOKENS = 400
SERIAL_ENABLED = False

# ======================================================
# SERIAL SETUP (SAFE)
# ======================================================

arduino = None

if SERIAL_ENABLED:
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
        time.sleep(2)
        print("Arduino connected.")
    except Exception:
        print("Serial connection failed. Running without Arduino.")
        arduino = None

# ======================================================
# OPENAI CLIENT
# ======================================================

client = OpenAI(api_key=OPENAI_API_KEY)

# ======================================================
# FLASK
# ======================================================

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "Backend running."

# ======================================================
# TOKEN CACHE
# ======================================================

ENCODER = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))

# ======================================================
# WORD BANKS (UNCHANGED + ENHANCED CLEANING)
# ======================================================

INSTANT = {"fix","grammar","spell","typo","yes","no","short","quick","caption"}
GENERATE = {"generate","create","write","draft","produce","make","compose"}
CREATIVE = {"story","poem","lyrics","novel","fiction","dialogue","plot","character"}
REASONING = {"explain","why","how","compare","contrast","pros","cons","difference","outline","recommend","steps"}
MATH = {"calculate","solve","equation","formula","probability","statistics","derive","proof"}
CODE = {"code","program","debug","script","function","class","api","algorithm","pipeline"}
HEAVY = {"analyze","optimize","predict","forecast","strategy","simulate","architecture","tradeoff","system"}
RESEARCH = {"assumptions","limitations","uncertainty","ethics","bias","long-term","failure modes"}

PLEASANTRIES = [
    "please","pls","plz","thank you","thanks","thx","ty",
    "sorry","my bad","apologies","apologize",
    "excuse me","pardon","hi","hello","hey","hiya",
    "good morning","good afternoon","good evening",
    "bye","goodbye","see you","take care",
    "hope this helps","much appreciated",
    "if you dont mind","when you get a chance",
    "just wondering","quick question","no worries","all good","chatgpt"
]

REQUEST_PHRASES = [
    "can you","could you","would you","will you",
    "i need you to","i want you to",
    "do you mind","is it possible to",
    "can u","could u"
]

EXTRA_REQUEST_PATTERNS = [
    "tell me",
    "give me",
    "show me",
    "let me know",
    "i was wondering",
    "would you mind"
]

MODAL_VERBS = {"can", "could", "would", "will"}

STOP_AND_FILLER_WORDS = {
    "um","uh","er","ah","like","so","well","okay","you know","i mean",
    "kind of","sort of","basically","actually","literally","honestly",
    "a","an","the","or","but","if","as",
    "at","by","for","from","in","into","of","on","onto","to","with",
    "this","that","these","those","it","its","they","them","their",
    "we","us","you","your","i","me","my","he","she","him","her",
    "is","am","are","was","were","be","been","being",
    "do","does","did","have","has","had",
    "just","maybe","probably","really","very","pretty",
    "stuff","things","something","anything","sorry"
}

SMART_CONNECTORS = {"and", "to", "on", "of", "for", "about", "with"}
TRASH_LEFTOVERS = {"there", "much", "so"}

STOP_AND_FILLER_WORDS |= MODAL_VERBS
STOP_AND_FILLER_WORDS -= SMART_CONNECTORS

# ======================================================
# CLEANING (UPGRADED)
# ======================================================

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_prefix(text: str) -> str:
    text = normalize(text)

    # Remove greetings
    for phrase in sorted(PLEASANTRIES, key=len, reverse=True):
        if text.startswith(phrase):
            text = text[len(phrase):].strip()

    # Remove request phrases
    for phrase in sorted(REQUEST_PHRASES + EXTRA_REQUEST_PATTERNS, key=len, reverse=True):
        if text.startswith(phrase):
            text = text[len(phrase):].strip()

    return text


def semantic_compress(text: str) -> str:
    replacements = {
        "in simple terms": "simply",
        "in a simple way": "simply",
        "what is": "",
        "who is": "",
        "what are": "",
        "can you explain": "explain",
        "could you explain": "explain",
        "please explain": "explain",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def aggressive_trim(text: str) -> str:
    """
    Removes trailing generic verbs like 'works', 'does', etc.
    Only if they are non-critical.
    """
    words = text.split()
    if len(words) > 2 and words[-1] in {"works","work","does","do"}:
        words = words[:-1]
    return " ".join(words)


def clean_prompt(raw_text: str) -> str:
    text = strip_prefix(raw_text)
    words = text.split()

    cleaned = [
        w for w in words
        if w not in TRASH_LEFTOVERS
           and w not in STOP_AND_FILLER_WORDS
    ]

    cleaned_text = " ".join(cleaned)
    cleaned_text = semantic_compress(cleaned_text)
    cleaned_text = aggressive_trim(cleaned_text)

    return cleaned_text
# ======================================================
# MODEL ROUTING
# ======================================================

def route_model(text: str) -> str:
    words = set(text.split())
    length = len(words)

    instant   = len(words & INSTANT)
    generate  = len(words & GENERATE)
    creative  = len(words & CREATIVE)
    reasoning = len(words & REASONING)
    math      = len(words & MATH)
    code      = len(words & CODE)
    heavy     = len(words & HEAVY)
    research  = len(words & RESEARCH)

    if text.count("?") > 1:
        reasoning += 1
    if length > 25:
        heavy += 1
    if length > 40:
        research += 1

    if instant > 0 and length <= 7:
        return "gpt-4o-mini"
    if generate > 0 and heavy == 0 and reasoning <= 1:
        return "gpt-4o-mini"
    if math > 0 and heavy == 0:
        return "gpt-4o-mini"
    if creative > 0 and heavy == 0 and code == 0:
        return "gpt-4o"
    if reasoning > 0 and heavy == 0 and code == 0:
        return "gpt-4o-mini"
    if code > 0:
        return "gpt-4o"
    if heavy > 0 or research > 0:
        return "gpt-4o"

    return "gpt-4o-mini"

# ======================================================
# OPENAI CALL
# ======================================================

def call_openai(prompt: str, model: str):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        temperature=0.7
    )
    return response.choices[0].message.content

# ======================================================
# ROUTE
# ======================================================

@app.route("/chat", methods=["POST"])
def chat():
    global arduino

    data = request.json
    raw_prompt = data.get("prompt", "")

    optimized_prompt = clean_prompt(raw_prompt)
    model = route_model(optimized_prompt)

    # Dual calls
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(call_openai, raw_prompt, model)
        future_b = executor.submit(call_openai, optimized_prompt, model)

        response_a = future_a.result()
        response_b = future_b.result()

    original_tokens = count_tokens(raw_prompt)
    optimized_tokens = count_tokens(optimized_prompt)
    token_saved = original_tokens - optimized_tokens
    percent = round(100 * token_saved / max(1, original_tokens), 1)
    energy_saved_kwh = round(token_saved * 4e-7, 8)

    # Safe Arduino write
    if SERIAL_ENABLED and arduino:
        try:
            if arduino.is_open:
                arduino.write(f"{original_tokens},{optimized_tokens}\n".encode())
        except:
            arduino = None

    return jsonify({
        "optimized_prompt": optimized_prompt,
        "original_tokens": original_tokens,
        "optimized_tokens": optimized_tokens,
        "token_saved": token_saved,
        "percent": percent,
        "energy_saved": energy_saved_kwh,
        "response_a": response_a,
        "response_b": response_b
    })

# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":
    app.run(debug=True)