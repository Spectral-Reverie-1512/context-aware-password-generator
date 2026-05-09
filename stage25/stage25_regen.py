import json
import random
from pathlib import Path

base = Path(r'd:/DLP/RockYou Datasets')
input_path = base / 'Final Context Dataset' / 'id_password_context.json'
context_dir = base / 'context'
context_dir.mkdir(parents=True, exist_ok=True)
output_cpm = context_dir / 'merged_context_data.json'

seed = 4242
random.seed(seed)

name_templates = [
    "My name is {names}.",
    "I am {names}.",
    "People call me {names}.",
    "Name: {names}.",
    "They know me as {names}.",
]
username_templates = [
    "I go by {usernames}.",
    "My handle is {usernames}.",
    "You can find me as {usernames}.",
    "Username: {usernames}.",
    "I use the username {usernames}.",
]
number_templates = [
    "I often use numbers like {numbers}.",
    "Numbers I like include {numbers}.",
    "Common digits for me are {numbers}.",
    "Some numbers tied to me: {numbers}.",
    "I use numbers such as {numbers}.",
]
location_templates = [
    "I am associated with {locations}.",
    "I am linked to {locations}.",
    "My location is {locations}.",
    "Location: {locations}.",
    "I have ties to {locations}.",
]
chars_templates = [
    "My characters include {chars}.",
    "I use characters like {chars}.",
    "Typical characters for me are {chars}.",
    "Characters: {chars}.",
    "I tend to use {chars} as characters.",
]

prefixes = ["Also, ", "Additionally, ", "", ""]


def join_list(items, use_and=True):
    if len(items) == 1:
        return items[0]
    if use_and:
        return ", ".join(items[:-1]) + f" and {items[-1]}"
    return ", ".join(items)


def format_chars(chars, use_and=True):
    if not chars:
        return ""
    if len(chars) == 1:
        return chars[0]
    if len(chars) == 2:
        return f"{chars[0]} and {chars[1]}" if use_and else f"{chars[0]}, {chars[1]}"
    if use_and:
        return ", ".join(chars[:-1]) + f" and {chars[-1]}"
    return ", ".join(chars)


def build_segments(full):
    segments = []
    names = full.get('names') or []
    usernames = full.get('username') or []
    locations = full.get('location') or []
    numbers = full.get('numbers') or full.get('digits') or []
    chars = full.get('chars') or []

    if names:
        template = random.choice(name_templates)
        segments.append((
            template.format(names=join_list(names, use_and=True)),
            template.format(names=join_list(names, use_and=False)),
        ))
    if usernames:
        template = random.choice(username_templates)
        segments.append((
            template.format(usernames=join_list(usernames, use_and=True)),
            template.format(usernames=join_list(usernames, use_and=False)),
        ))
    if numbers:
        template = random.choice(number_templates)
        segments.append((
            template.format(numbers=join_list(numbers, use_and=True)),
            template.format(numbers=join_list(numbers, use_and=False)),
        ))
    if locations:
        template = random.choice(location_templates)
        segments.append((
            template.format(locations=join_list(locations, use_and=True)),
            template.format(locations=join_list(locations, use_and=False)),
        ))
    if chars:
        template = random.choice(chars_templates)
        segments.append((
            template.format(chars=format_chars(chars, use_and=True)),
            template.format(chars=format_chars(chars, use_and=False)),
        ))

    return segments


def build_raw_context(full):
    segments = build_segments(full)
    if not segments:
        return ""
    random.shuffle(segments)
    first, rest = segments[0], segments[1:]
    first_sentence = random.choice(prefixes) + first[1]
    other_sentences = [random.choice(prefixes) + s[0] for s in rest]
    return " ".join([first_sentence] + other_sentences).strip()


def stream_array(path):
    decoder = json.JSONDecoder()
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        buf = ''
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            buf += chunk
            while True:
                buf = buf.lstrip()
                if not buf:
                    break
                if buf[0] == '[':
                    buf = buf[1:]
                    continue
                if buf[0] == ',':
                    buf = buf[1:]
                    continue
                if buf[0] == ']':
                    return
                try:
                    obj, idx = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    break
                yield obj
                buf = buf[idx:]
        buf = buf.strip()
        if buf and buf[0] == ']':
            return

first_raw = None

with output_cpm.open('w', encoding='utf-8') as f_cpm:
    f_cpm.write('[\n')
    first_cpm = True

    for obj in stream_array(input_path):
        sample_id = obj.get('id')
        context = obj.get('context') or {}
        full = context.get('full') or {}

        structured = {
            'names': full.get('names') or [],
            'username': full.get('username') or [],
            'location': full.get('location') or [],
            'numbers': full.get('numbers') or full.get('digits') or [],
            'chars': full.get('chars') or [],
        }
        raw_context = build_raw_context(structured)
        if first_raw is None:
            first_raw = raw_context

        cpm_entry = {
            'sample_id': sample_id,
            'raw_context': raw_context,
            'structured_context': structured,
        }

        if not first_cpm:
            f_cpm.write(',\n')
        f_cpm.write(json.dumps(cpm_entry, ensure_ascii=False))
        first_cpm = False

    f_cpm.write('\n]\n')

print('Wrote', output_cpm)
print('sample_id 1 raw_context:', first_raw)