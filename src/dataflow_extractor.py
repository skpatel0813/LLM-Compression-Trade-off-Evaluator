# src/dataflow_extractor.py
"""
Dataflow Graph (DFG) extraction for Python code using tree-sitter.
"""

from __future__ import annotations
from typing import List, Set, Tuple
from tree_sitter import Parser
from tree_sitter_languages import get_language


class DataflowExtractor:
    """Extract dataflow graphs from Python code using tree-sitter."""
    
    def __init__(self):
        self.parser = Parser()
        self.parser.set_language(get_language('python'))
    
    def extract_dfg(self, code: str) -> Set[Tuple[str, str]]:
        """Extract dataflow edges from code."""
        try:
            tree = self.parser.parse(bytes(code, 'utf-8'))
            edges = set()
            self._traverse_node(tree.root_node, edges, scope_vars=set())
            return edges
        except Exception:
            return set()
    
    def _traverse_node(self, node, edges: Set[Tuple[str, str]], scope_vars: Set[str]):
        """Recursively traverse AST and extract dataflow edges."""
        node_type = node.type
        
        if node_type == 'function_definition':
            self._handle_function_def(node, edges, scope_vars.copy())
            return
        
        if node_type == 'assignment':
            self._handle_assignment(node, edges, scope_vars)
            return
        
        if node_type == 'augmented_assignment':
            self._handle_augmented_assignment(node, edges, scope_vars)
            return
        
        if node_type == 'return_statement':
            self._handle_return(node, edges, scope_vars)
            return
        
        if node_type == 'for_statement':
            self._handle_for_loop(node, edges, scope_vars)
            return
        
        for child in node.children:
            self._traverse_node(child, edges, scope_vars)
    
    def _handle_function_def(self, node, edges, scope_vars):
        """Handle function definition."""
        params = self._find_child_by_type(node, 'parameters')
        if params:
            for param in params.children:
                if param.type == 'identifier':
                    scope_vars.add(self._get_text(param))
                elif param.type == 'typed_parameter':
                    identifier = self._find_child_by_type(param, 'identifier')
                    if identifier:
                        scope_vars.add(self._get_text(identifier))
        
        body = self._find_child_by_type(node, 'block')
        if body:
            for child in body.children:
                self._traverse_node(child, edges, scope_vars)
    
    def _handle_assignment(self, node, edges, scope_vars):
        """Handle assignment: sources -> target."""
        left = node.child_by_field_name('left')
        if not left:
            return
        
        targets = self._extract_identifiers(left)
        for target in targets:
            scope_vars.add(target)
        
        right = node.child_by_field_name('right')
        if not right:
            return
        
        sources = self._extract_identifiers(right)
        
        for source in sources:
            if source in scope_vars or self._is_builtin(source):
                for target in targets:
                    edges.add((source, target))
    
    def _handle_augmented_assignment(self, node, edges, scope_vars):
        """Handle augmented assignment: x += y."""
        left = node.child_by_field_name('left')
        right = node.child_by_field_name('right')
        
        if not left or not right:
            return
        
        targets = self._extract_identifiers(left)
        sources = self._extract_identifiers(right)
        
        for target in targets:
            scope_vars.add(target)
            edges.add((target, target))
            for source in sources:
                if source in scope_vars or self._is_builtin(source):
                    edges.add((source, target))
    
    def _handle_return(self, node, edges, scope_vars):
        """Handle return statement."""
        for child in node.children:
            if child.type not in ('return', 'comment'):
                sources = self._extract_identifiers(child)
                for source in sources:
                    if source in scope_vars or self._is_builtin(source):
                        edges.add((source, 'return'))
    
    def _handle_for_loop(self, node, edges, scope_vars):
        """Handle for loop."""
        left = node.child_by_field_name('left')
        loop_vars = self._extract_identifiers(left) if left else set()
        
        for var in loop_vars:
            scope_vars.add(var)
        
        right = node.child_by_field_name('right')
        iterables = self._extract_identifiers(right) if right else set()
        
        for iterable in iterables:
            if iterable in scope_vars or self._is_builtin(iterable):
                for loop_var in loop_vars:
                    edges.add((iterable, loop_var))
        
        body = node.child_by_field_name('body')
        if body:
            for child in body.children:
                self._traverse_node(child, edges, scope_vars)
    
    def _extract_identifiers(self, node) -> Set[str]:
        """Extract all identifier names from a node."""
        identifiers = set()
        if node.type == 'identifier':
            identifiers.add(self._get_text(node))
        else:
            for child in node.children:
                identifiers.update(self._extract_identifiers(child))
        return identifiers
    
    def _get_text(self, node) -> str:
        """Get text content of a node."""
        return node.text.decode('utf-8') if node.text else ''
    
    def _find_child_by_type(self, node, child_type: str):
        """Find first child with given type."""
        for child in node.children:
            if child.type == child_type:
                return child
        return None
    
    def _is_builtin(self, name: str) -> bool:
        """Check if name is a Python builtin."""
        builtins = {
            'abs', 'all', 'any', 'bin', 'bool', 'bytes', 'chr', 'dict', 'dir',
            'divmod', 'enumerate', 'filter', 'float', 'format', 'hex', 'int',
            'len', 'list', 'map', 'max', 'min', 'oct', 'ord', 'pow', 'range',
            'repr', 'reversed', 'round', 'set', 'sorted', 'str', 'sum', 'tuple',
            'type', 'zip', 'True', 'False', 'None', 'print', 'input', 'open',
        }
        return name in builtins


def compute_dataflow_match(refs: List[str], hyps: List[str]) -> float:
    """Compute dataflow match score between references and hypotheses."""
    if not refs or not hyps or len(refs) != len(hyps):
        return 0.0
    
    extractor = DataflowExtractor()
    scores = []
    
    for ref, hyp in zip(refs, hyps):
        ref_edges = extractor.extract_dfg(ref)
        hyp_edges = extractor.extract_dfg(hyp)
        
        if not ref_edges and not hyp_edges:
            scores.append(1.0)
            continue
        
        if not ref_edges or not hyp_edges:
            scores.append(0.0)
            continue
        
        common = len(ref_edges & hyp_edges)
        total = len(ref_edges) + len(hyp_edges)
        score = (2.0 * common) / total if total > 0 else 0.0
        scores.append(score)
    
    return sum(scores) / len(scores) if scores else 0.0
