import json
from pathlib import Path

base = Path(r'd:/DLP/RockYou Datasets')
input_path = base / 'Final Context Dataset' / 'id_password_context.json'
context_dir = base / 'context'
context_dir.mkdir(parents=True, exist_ok=True)
output_cpm = context_dir / 'merged_context_data.json'
output_targets = context_dir / 'password_targets.json'

def build_raw_context(full):
    sentences = []
    names = full.get('names') or []
    usernames = full.get('username') or []
    locations = full.get('location') or []
    numbers = full.get('numbers') or full.get('digits') or []
    chars = full.get('chars') or []

    if names:
        if len(names) == 1:
            sentences.append(f"My name is {names[0]}.")
        else:
            sentences.append("My name is " + ", ".join(names[:-1]) + f" and {names[-1]}.")
    if usernames:
        if len(usernames) == 1:
            sentences.append(f"I go by {usernames[0]}.")
        else:
            sentences.append("I go by " + ", ".join(usernames[:-1]) + f" and {usernames[-1]}.")
    if numbers:
        if len(numbers) == 1:
            sentences.append(f"I often use numbers like {numbers[0]}.")
        else:
            sentences.append("I often use numbers like " + ", ".join(numbers[:-1]) + f" or {numbers[-1]}.")
    if locations:
        if len(locations) == 1:
            sentences.append(f"I am associated with {locations[0]}.")
        else:
            sentences.append("I am associated with " + ", ".join(locations[:-1]) + f" and {locations[-1]}.")
    if chars:
        if len(chars) == 1:
            sentences.append(f"My characters include {chars[0]}.")
        else:
            sentences.append("My characters include " + ", ".join(chars[:-1]) + f" and {chars[-1]}.")

    return " ".join(sentences)


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

with output_cpm.open('w', encoding='utf-8') as f_cpm, output_targets.open('w', encoding='utf-8') as f_tgt:
    f_cpm.write('[\n')
    f_tgt.write('[\n')
    first_cpm = True
    first_tgt = True

    for obj in stream_array(input_path):
        sample_id = obj.get('id')
        password = obj.get('password') or ''
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

        cpm_entry = {
            'sample_id': sample_id,
            'raw_context': raw_context,
            'structured_context': structured,
        }
        tgt_entry = {
            'sample_id': sample_id,
            'password': password,
        }

        if not first_cpm:
            f_cpm.write(',\n')
        f_cpm.write(json.dumps(cpm_entry, ensure_ascii=False))
        first_cpm = False

        if not first_tgt:
            f_tgt.write(',\n')
        f_tgt.write(json.dumps(tgt_entry, ensure_ascii=False))
        first_tgt = False

    f_cpm.write('\n]\n')
    f_tgt.write('\n]\n')

print('Wrote', output_cpm)
print('Wrote', output_targets)