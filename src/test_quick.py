#!/usr/bin/env python3
print("Testing tree-sitter...")
from tree_sitter import Parser
import tree_sitter_python as tsp

parser = Parser()
parser.language = tsp.language()
code = "def test():\n    return 42\n"
tree = parser.parse(bytes(code, "utf-8"))
print(f"✓ tree-sitter works! Root: {tree.root_node.type}")

print("\nTesting CodeBLEU...")
from codebleu import calc_codebleu

refs = ["def add(a, b):\n    return a + b\n"]
hyps = ["def add(x, y):\n    return x + y\n"]
result = calc_codebleu(refs, hyps, lang="python")
print(f"✓ CodeBLEU works! Score: {result['codebleu']:.4f}")
syntax = result.get('syntax_match_score', result.get('syntax_match', 0))
print(f"  Syntax: {syntax:.4f}")

if syntax > 0:
    print("\n✓✓✓ SUCCESS! Everything is working! ✓✓✓")
else:
    print("\n✗ WARNING: Syntax score is still 0")