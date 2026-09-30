import re, sys, collections
# Count arithmetic lane-ops per basic block of each function in an AIR .ll;
# detect loops by back edges (branch to a block defined earlier).
ll = open(sys.argv[1]).read()
for fn in re.finditer(r'^define [^@]*@(\w+)\((.*?)^}', ll, re.S | re.M):
    name, body = fn.group(1), fn.group(0)
    blocks = collections.OrderedDict(); cur = 'entry'; order = ['entry']; blocks['entry'] = []
    for line in body.split('\n')[1:]:
        m = re.match(r'^(\d+):', line)
        if m: cur = m.group(1); order.append(cur); blocks[cur] = []; continue
        blocks[cur].append(line)
    def lanes(line):
        m = re.search(r'<(\d+) x (float|half|i32)>', line); return int(m.group(1)) if m else 1
    def cost(lines):
        c = collections.Counter()
        for l in lines:
            if re.search(r'= (fadd|fsub|fmul|fneg)\b', l): c['flop'] += lanes(l)
            elif re.search(r'= fdiv\b', l): c['div'] += lanes(l)
            elif re.search(r'@air\.(fast_)?fma', l) or re.search(r'@llvm\.fma', l): c['flop'] += 2 * lanes(l)
            elif re.search(r'@air\.(fast_)?dot\.v(\d)', l): n = int(re.search(r'dot\.v(\d)', l).group(1)); c['flop'] += 2 * n
            elif re.search(r'@air\.(fast_)?(rsqrt|sqrt|exp2|log2|pow|cos|sin|exp|log|tan)', l): c['transc'] += lanes(l)
            elif re.search(r'@air\.(fast_)?(fmax|fmin|clamp|saturate|mix)', l): c['flop'] += lanes(l) * (3 if 'mix' in l else 1)
            elif re.search(r'@air\.sample_', l): c['sample'] += 1
            elif re.search(r'= load\b', l): c['load'] += 1
            elif re.search(r'store ', l): c['store'] += 1
            elif re.search(r'= (add|sub|mul|shl|lshr|and|or|xor)\b', l): c['int'] += lanes(l)
        return c
    idx = {b: i for i, b in enumerate(order)}
    loops = []
    for b in order:
        for l in blocks[b]:
            for t in re.findall(r'label %(\d+)', l):
                if t in idx and idx[t] <= idx[b]: loops.append((t, b))
    total = collections.Counter()
    for b in order: total += cost(blocks[b])
    print(f'{name}: {len(order)} blocks, total static {dict(total)}')
    for head, tail in loops:
        body = collections.Counter()
        for b in order[idx[head]:idx[tail] + 1]: body += cost(blocks[b])
        print(f'   loop {head}..{tail}: per iteration {dict(body)}')
