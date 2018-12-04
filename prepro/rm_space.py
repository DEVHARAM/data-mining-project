
with open("first.txt", "r", encoding='utf8') as f:
    with open("second.txt", "w", encoding='utf8') as out:
        text = f.readline()
        out.write(text[2:])
        for line in iter(lambda: f.readline(), ''):
            text = line[1:]
            out.write(text)
