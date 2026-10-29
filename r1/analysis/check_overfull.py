import re
import sys
log = open(sys.argv[1], encoding="latin-1").read()
for m in re.finditer(r"Overfull \\hbox \(([0-9.]+)pt too wide\) (in alignment|in paragraph)", log):
    if float(m.group(1)) > 20:
        pre = log[:m.start()]
        files = re.findall(r"\(((?:\./)?(?:tables|sections)/[^ \)\n]+\.tex)", pre)
        pages = re.findall(r"\[(\d+)", pre)
        print(m.group(1), m.group(2), files[-1] if files else "?", "page>", pages[-1] if pages else "?")
