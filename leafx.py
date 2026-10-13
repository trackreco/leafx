#!/usr/bin/env python3
"""leafx: generate lane-templated, host/device-annotated copies of mkFit leaf code.

The mkFit leaf code (Matriplex physics: propagation, Kalman update) is written for
Matriplexes of a fixed width NN. leafx reads those sources unchanged and writes a copy
in which every function that works on Matriplexes is a template on the lane count N,
carries a host/device annotation, and lives in a target namespace (mkfit::portable by
default). MkFitCore keeps its own code; the consumer (a GPU port) keeps the copy.

Name resolution is left to the consumer: the copy refers to Config::, Const::, vdt::,
TrackerInfo, PropagationFlags, hipo(), ... unqualified, from inside the target
namespace. On the host they resolve to MkFitCore's; a device build declares its own
versions in the target namespace and they shadow MkFitCore's. The dependency report
lists every such name.

Plain Python, no dependencies. --verify uses libclang (python3-clang), if present, to
cross-check the function extents found by the tokenizer.
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tomllib
from collections import namedtuple

VERSION = "0.1"

# =============================================================================
# Pre-pass 1: resolve #if blocks on configured macros (a minimal unifdef)
# =============================================================================

DIRECTIVE_RE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")


def _eval_cond(kind, rest, defined, undefined):
    """True / False if the condition only tests a configured macro, else None."""
    rest = re.sub(r"//.*$|/\*.*?\*/", "", rest).strip()
    if kind in ("ifdef", "ifndef"):
        m = rest.split()[0] if rest else ""
        val = True if m in defined else False if m in undefined else None
        if val is None:
            return None
        return val if kind == "ifdef" else not val
    # if / elif
    if rest in ("0", "1"):
        return rest == "1"
    m = re.fullmatch(r"(!\s*)?defined\s*\(?\s*(\w+)\s*\)?", rest)
    if m:
        name = m.group(2)
        val = True if name in defined else False if name in undefined else None
        if val is None:
            return None
        return (not val) if m.group(1) else val
    return None


def strip_conditionals(lines, defined, undefined):
    """Blank out the dead branches of #if blocks whose condition is known.
    Line count is preserved (dropped lines become empty) so that line numbers stay
    those of the source."""
    out = []
    stack = []  # (known, parent_active, branch_active, taken)
    active = True
    for line in lines:
        m = DIRECTIVE_RE.match(line)
        if not m:
            out.append(line if active else "")
            continue
        kind, rest = m.group(1), m.group(2)
        if kind in ("if", "ifdef", "ifndef"):
            val = _eval_cond(kind, rest, defined, undefined)
            if val is None:
                stack.append(("unknown", active, active, True))
                out.append(line if active else "")
            else:
                stack.append(("known", active, active and val, val))
                active = active and val
                out.append("")
        elif kind == "elif":
            known, parent, _, taken = stack[-1]
            if known == "unknown":
                out.append(line if parent else "")
                continue
            val = _eval_cond("elif", rest, defined, undefined)
            if val is None:
                raise RuntimeError(f"#elif with an unknown condition inside a resolved #if: {line.strip()}")
            branch = parent and not taken and val
            stack[-1] = (known, parent, branch, taken or val)
            active = branch
            out.append("")
        elif kind == "else":
            known, parent, _, taken = stack[-1]
            if known == "unknown":
                out.append(line if parent else "")
                continue
            branch = parent and not taken
            stack[-1] = (known, parent, branch, True)
            active = branch
            out.append("")
        else:  # endif
            known, parent, _, _ = stack.pop()
            out.append(line if (known == "unknown" and parent) else "")
            active = parent
    if stack:
        raise RuntimeError("unbalanced #if")
    return out


# =============================================================================
# Pre-pass 2: markup in comments
#   //@leafx begin-skip ... //@leafx end-skip   drop the lines in between
#   //@leafx skip                               drop the next declaration / definition
#   //@leafx size(EXPR)                         this line's array declarator gets size EXPR
# =============================================================================

MARK_RE = re.compile(r"//\s*@leafx\s+([\w-]+)(?:\((.*)\))?")


def strip_marked_regions(lines):
    out, skipping = [], False
    for line in lines:
        m = MARK_RE.search(line)
        if m and m.group(1) == "begin-skip":
            skipping = True
            out.append("")
            continue
        if m and m.group(1) == "end-skip":
            skipping = False
            out.append("")
            continue
        out.append("" if skipping else line)
    return out


# =============================================================================
# Tokenizer
# =============================================================================

Tok = namedtuple("Tok", "kind text start end line")

TOK_RE = re.compile(
    r"""
 (?P<ws>[ \t\r\f\v]+)
|(?P<nl>\n)
|(?P<lc>//[^\n]*)
|(?P<bc>/\*.*?\*/)
|(?P<str>"(?:\\.|[^"\\\n])*")
|(?P<num>\.?\d(?:[\w.]|[eEpP][+-]|'(?=\w))*)
|(?P<chr>'(?:\\.|[^'\\\n])+')
|(?P<id>[A-Za-z_]\w*)
|(?P<p>::|->|\.\.\.|[^\s\w])
""",
    re.S | re.X,
)

TRIVIA = ("ws", "nl", "lc", "bc", "pp")


def tokenize(text):
    toks, pos, line, at_line_start = [], 0, 1, True
    n = len(text)
    while pos < n:
        if at_line_start and text[pos] == "#":
            end = pos
            while True:  # preprocessor line, with continuations
                e = text.find("\n", end)
                if e == -1:
                    e = n
                if e > 0 and text[e - 1] == "\\" and e < n:
                    end = e + 1
                    continue
                break
            toks.append(Tok("pp", text[pos:e], pos, e, line))
            line += text.count("\n", pos, e)
            pos = e
            continue
        m = TOK_RE.match(text, pos)
        if not m:
            raise RuntimeError(f"tokenizer stuck at line {line}: {text[pos:pos+30]!r}")
        kind = m.lastgroup
        if kind == "rdelim":
            kind = "str"
        t = Tok(kind, m.group(), pos, m.end(), line)
        toks.append(t)
        line += t.text.count("\n")
        if kind == "nl":
            at_line_start = True
        elif kind != "ws":
            at_line_start = False
        pos = m.end()
    return toks


# =============================================================================
# Structure: namespaces, classes, functions, other statements
# =============================================================================


class Item:
    def __init__(self, kind, **kw):
        self.kind = kind  # ns, func, decl, alias, class, other
        self.__dict__.update(kw)

    def __repr__(self):
        return f"Item({self.kind}, {getattr(self, 'name', '')})"


class Parser:
    def __init__(self, toks):
        self.t = toks
        self.match = self._match_brackets()

    def _match_brackets(self):
        pairs, stack = {}, []
        openers = {"(": ")", "[": "]", "{": "}"}
        for i, t in enumerate(self.t):
            if t.kind != "p":
                continue
            if t.text in openers:
                stack.append(i)
            elif t.text in (")", "]", "}"):
                j = stack.pop()
                if openers[self.t[j].text] != t.text:
                    raise RuntimeError(f"bracket mismatch, line {t.line}")
                pairs[j] = i
                pairs[i] = j
        if stack:
            raise RuntimeError(f"unclosed bracket, line {self.t[stack[-1]].line}")
        return pairs

    def sig(self, a, b):
        """Indices of significant tokens in [a, b)."""
        return [i for i in range(a, b) if self.t[i].kind not in TRIVIA]

    def parse(self, a, b, scope):
        items, i, start = [], a, None
        while i < b:
            t = self.t[i]
            if t.kind in TRIVIA:
                i += 1
                continue
            if start is None:
                start = i
            if t.kind == "p" and t.text in ("(", "["):
                i = self.match[i] + 1
                continue
            if t.kind == "id" and t.text == "namespace" and start == i:
                j = i + 1
                while self.t[j].kind in TRIVIA or (self.t[j].text not in ("{", "=", ";")):
                    j += 1
                if self.t[j].text == "{":
                    name = "".join(self.t[k].text for k in self.sig(i + 1, j)) or None
                    close = self.match[j]
                    items.append(Item("ns", name=name, start=i, open=j, close=close, end=close + 1,
                                      children=self.parse(j + 1, close, "ns")))
                    i, start = close + 1, None
                    continue
                k = j
                while self.t[k].text != ";":
                    k += 1
                items.append(Item("other", start=i, end=k + 1, what="namespace alias"))
                i, start = k + 1, None
                continue
            if scope == "class" and t.kind == "p" and t.text == ":":
                prev = self.sig(start, i)
                if len(prev) == 1 and self.t[prev[0]].text in ("public", "private", "protected"):
                    i, start = i + 1, None
                    continue
            if t.kind == "p" and t.text == "{":
                close = self.match[i]
                head = self.sig(start, i)
                words = [self.t[k].text for k in self._skip_template_headers(head)]
                if words and words[0] in ("class", "struct", "union", "enum"):
                    k = close + 1
                    while self.t[k].text != ";":
                        k += 1
                    items.append(Item("class", start=start, open=i, close=close, end=k + 1,
                                      name=self._class_name(head),
                                      children=self.parse(i + 1, close, "class")))
                    i, start = k + 1, None
                    continue
                fn = self._function_head(start, i)
                if fn:
                    items.append(Item("func", start=start, body=(i, close), end=close + 1, **fn))
                    i, start = close + 1, None
                    continue
                i = close + 1  # braced initializer etc.: continue to the ';'
                continue
            if t.kind == "p" and t.text == ";":
                fn = self._function_head(start, i)
                if fn:
                    items.append(Item("decl", start=start, body=None, end=i + 1, **fn))
                else:
                    words = [self.t[k].text for k in self._skip_template_headers(self.sig(start, i))]
                    kind = "other"
                    what = " ".join(words[:3])
                    if words and words[0] == "using" and "=" in words:
                        kind = "alias"
                    elif words and words[0] == "typedef":
                        kind = "alias"
                    elif words and words[0] in ("class", "struct") and len(words) == 2:
                        what = "forward declaration"
                    items.append(Item(kind, start=start, end=i + 1, what=what,
                                      name=words[1] if kind == "alias" and words[0] == "using" else
                                      (words[-1] if kind == "alias" else None)))
                i, start = i + 1, None
                continue
            i += 1
        return items

    def _skip_template_headers(self, idxs):
        """Drop leading `template <...>` groups from a list of significant indices."""
        k = 0
        while k < len(idxs) and self.t[idxs[k]].text == "template":
            depth, k = 0, k + 1
            while k < len(idxs):
                x = self.t[idxs[k]].text
                if x == "(":
                    k = idxs.index(self.match[idxs[k]])
                elif x == "<":
                    depth += 1
                elif x == ">":
                    depth -= 1
                    if depth == 0:
                        k += 1
                        break
                k += 1
        return idxs[k:]

    def _class_name(self, head):
        ids = [self.t[k].text for k in self._skip_template_headers(head) if self.t[k].kind == "id"]
        ids = [x for x in ids if x not in ("class", "struct", "union", "enum", "final", "alignas")]
        return ids[0] if ids else None

    NOT_NAMES = {"__attribute__", "alignas", "decltype", "noexcept", "sizeof", "alignof", "defined",
                 "static_assert", "return", "if", "for", "while", "switch"}

    def _function_head(self, a, b):
        """If tokens [a, b) are a function head, return its name and the index of its '('."""
        idxs = self._skip_template_headers(self.sig(a, b))
        if not idxs:
            return None
        if self.t[idxs[0]].text in ("using", "typedef", "return", "static_assert"):
            return None
        # operator functions first: their names can contain '=', '(' and ')'
        texts = [self.t[k].text for k in idxs]
        if "operator" in texts:
            o = texts.index("operator")
            j = o + 1
            if j + 1 < len(idxs) and texts[j] == "(" and texts[j + 1] == ")":
                j += 2  # operator()
            else:
                while j < len(idxs) and texts[j] != "(":
                    j += 1  # operator+=, operator[], operator T, ...
            if j < len(idxs) and texts[j] == "(":
                name = "operator" + "".join(texts[o + 1:j])
                return dict(name=name, paren=idxs[j], name_tok=idxs[o], spec_start=idxs[0])
            return None
        for n, k in enumerate(idxs):
            tk = self.t[k]
            if tk.text == "=" and tk.kind == "p":
                return None  # variable with an initializer
            if tk.text != "(":
                continue
            if n == 0:
                return None
            prev = self.t[idxs[n - 1]]
            if prev.kind == "id" and prev.text not in self.NOT_NAMES:
                name, m = prev.text, n - 1
                while m >= 2 and self.t[idxs[m - 1]].text == "::" and self.t[idxs[m - 2]].kind == "id":
                    name = self.t[idxs[m - 2]].text + "::" + name
                    m -= 2
                if m >= 1 and self.t[idxs[m - 1]].text == "operator":
                    return None
                return dict(name=name, paren=k, name_tok=idxs[n - 1], spec_start=idxs[0])
            # operator overloads: operator+ ( / operator() (
            ops, m = [], n - 1
            while m >= 0 and self.t[idxs[m]].kind == "p":
                ops.insert(0, self.t[idxs[m]].text)
                m -= 1
            if m >= 0 and self.t[idxs[m]].text == "operator":
                if ops == ["(", ")"] and tk is not None:
                    pass
                return dict(name="operator" + "".join(ops), paren=k, name_tok=idxs[m], spec_start=idxs[0])
            if prev.text == ")":
                # e.g. `operator()(`: the '(' found first is part of the name
                if n >= 2 and self.t[idxs[n - 2]].text == "(" and n >= 3 and self.t[idxs[n - 3]].text == "operator":
                    return dict(name="operator()", paren=k, name_tok=idxs[n - 3], spec_start=idxs[0])
                return None
            return None
        return None


# =============================================================================
# The transformation
# =============================================================================


class Report:
    def __init__(self):
        self.notes = []
        self.errors = []
        self.deps = {}  # name -> set of (file, line)

    def note(self, f, line, msg):
        self.notes.append(f"{f}:{line}: {msg}")

    def error(self, f, line, msg):
        self.errors.append(f"{f}:{line}: {msg}")

    def dep(self, name, f, line):
        self.deps.setdefault(name, set()).add((f, line))


BUILTIN_TYPES = {"float", "double", "int", "unsigned", "short", "long", "bool", "char", "auto"}
DEVICE_SUSPECT = {
    "throw": "exceptions are not available in device code",
    "printf": "printf works in CUDA device code but not in every backend",
    "cout": "iostreams are host only",
    "cerr": "iostreams are host only",
    "function": "std::function is host only",
    "vector": "std::vector is host only",
    "thread_local": "no thread_local in device code",
    "new": "dynamic allocation in device code",
    "isfinite": "std::isfinite is unusable under -Ofast (use mkfit::isFinite)",
}


class FileXform:
    def __init__(self, cfg, fcfg, mplex_types, api_names, report):
        self.cfg = cfg
        self.fcfg = fcfg
        self.report = report
        self.src_rel = fcfg["src"]
        self.src = os.path.join(cfg["source_root"], self.src_rel)
        self.stem = os.path.splitext(os.path.basename(self.src_rel))[0]
        self.mplex_types = set(mplex_types)
        self.api_names = api_names
        self.lane = cfg.get("lane_param", "N")
        self.lane_const = cfg.get("lane_constant", "NN")
        self.annot = cfg.get("annotation", "MKFIT_HOST_DEVICE")
        self.target = cfg.get("target_namespace", "mkfit::portable")
        self.unqualify = cfg.get("unqualify", ["mkfit"])
        self.strip_calls = set(cfg.get("strip_calls", []))
        self.drop_usings = set(cfg.get("drop_using_namespace", ["mkfit"]))
        self.skip_symbols = set(fcfg.get("skip", []))
        self.size_marks = {}  # line -> expr

        with open(self.src) as fh:
            text = fh.read()
        # `splice`: file-scope `#include "X"` replaced by the text of X (e.g. a .icc with helpers)
        self.spliced = []
        for name in fcfg.get("splice", []):
            path = os.path.join(os.path.dirname(self.src), name)
            pat = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"' + re.escape(name) + r'"[^\n]*$', re.M)
            if not pat.search(text):
                raise RuntimeError(f"{self.src_rel}: splice: no #include \"{name}\"")
            with open(path) as fh:
                inc = fh.read()
            text = pat.sub(lambda m: inc, text, count=1)
            self.spliced.append(os.path.join(os.path.dirname(self.src_rel), name))
        self.text_orig = text
        lines = text.split("\n")
        lines = strip_marked_regions(lines)
        lines = strip_conditionals(lines, set(cfg.get("define", [])), set(cfg.get("undefine", [])))
        for ln, line in enumerate(lines, 1):
            m = MARK_RE.search(line)
            if m and m.group(1) == "size":
                self.size_marks[ln] = m.group(2)
        self.text = "\n".join(lines)
        self.toks = tokenize(self.text)
        self.p = Parser(self.toks)
        self.items = self.p.parse(0, len(self.toks), "file")
        self.detail_ns = f"detail_{self.stem}"
        self.kernels = set()
        self.detail_names = set()
        self.local_globals = {}  # name -> Matriplex type: namespace-scope Matriplexes made function-local
        self.nondeducible = set()  # functions templated on N that callers must give <N> explicitly

    # ---- helpers ----------------------------------------------------------

    def line(self, i):
        return self.toks[i].line

    def preceding_marks(self, item, prev_end):
        marks = []
        for k in range(prev_end, item.start):
            t = self.toks[k]
            if t.kind == "lc":
                m = MARK_RE.search(t.text)
                if m:
                    marks.append(m.group(1))
        return marks

    def collect_detail_names(self, items):
        for it in items:
            if it.kind in ("func", "decl"):
                self.detail_names.add(it.name)
            elif it.kind == "alias" and it.name:
                self.detail_names.add(it.name)

    # ---- per-item rewriting ----------------------------------------------

    def local_aliases(self, a, b):
        """Names declared by `using X =` / `typedef ... X;` inside [a, b)."""
        names = set()
        sig = self.p.sig(a, b)
        for n, k in enumerate(sig):
            if self.toks[k].text == "using" and n + 2 < len(sig) and self.toks[sig[n + 2]].text == "=":
                names.add(self.toks[sig[n + 1]].text)
            if self.toks[k].text == "typedef":
                m = n + 1
                while m < len(sig) and self.toks[sig[m]].text != ";":
                    m += 1
                names.add(self.toks[sig[m - 1]].text)
        return names

    def uses_mplex(self, a, b, types):
        return any(self.toks[k].kind == "id" and (self.toks[k].text in types or self.toks[k].text == self.lane_const)
                   for k in self.p.sig(a, b))

    def rewrite_span(self, a, b, types, templated, in_detail, edits, fname, protected=()):
        """Token-level rewrites inside [a, b). Appends (start, end, text) to edits; tokens
        whose index is in `protected` were already rewritten by head_fix."""
        sig = [k for k in self.p.sig(a, b)]
        local = self.local_aliases(a, b) if templated is not None else set()
        # names used other than as a call (variables, parameters): not external functions
        as_value = set()
        for n, k in enumerate(sig):
            if T_ := self.toks[k]:
                if T_.kind == "id" and (n + 1 == len(sig) or self.toks[sig[n + 1]].text != "("):
                    as_value.add(T_.text)
        tset = types - local
        T = self.toks
        skip_until = -1
        for n, k in enumerate(sig):
            if k < skip_until or k in protected:
                continue
            t = T[k]
            prev = T[sig[n - 1]] if n > 0 else None
            nxt = T[sig[n + 1]] if n + 1 < len(sig) else None
            # in-body #include "X.ah" -> path from the source root
            # (pp tokens are trivia, handled below)
            if t.kind != "id":
                continue
            qualified_before = prev is not None and prev.text in ("::", ".", "->")
            # mkfit::X -> X, so that the target namespace can shadow X
            if t.text in self.unqualify and nxt is not None and nxt.text == "::" and not qualified_before:
                edits.append((t.start, T[sig[n + 1]].end, ""))
                self.report.note(self.src_rel, t.line, f"unqualified {t.text}::{T[sig[n + 2]].text}")
                continue
            # debug statements: dprint(...);
            if t.text in self.strip_calls and nxt is not None and nxt.text == "(" and not qualified_before:
                close = self.p.match[sig[n + 1]]
                end = close + 1
                j = end
                while j < len(T) and T[j].kind in TRIVIA:
                    j += 1
                if j < len(T) and T[j].text == ";":
                    end = j + 1
                edits.append((t.start, T[end - 1].end, ""))
                skip_until = end
                continue
            # `const Matriplex::idx_t N = NN;` and alike: the lane count is the template parameter now
            if t.text == self.lane and nxt is not None and nxt.text == "=" and n + 3 < len(sig) \
                    and T[sig[n + 2]].text == self.lane_const and T[sig[n + 3]].text == ";":
                m = n
                while m > 0 and T[sig[m - 1]].text not in (";", "{", "}"):
                    m -= 1
                edits.append((T[sig[m]].start, T[sig[n + 3]].end, ""))
                skip_until = sig[n + 3] + 1
                continue
            if templated is False and (t.text in tset or t.text == self.lane_const):
                self.report.error(self.src_rel, t.line,
                                  f"{fname}: uses {t.text} but its signature has no Matriplex to deduce {self.lane} from")
                continue
            if t.text == self.lane_const and not qualified_before:
                edits.append((t.start, t.end, self.lane))
                continue
            if t.text in tset and not qualified_before:
                if nxt is not None and nxt.text == "<":
                    continue
                qual = f"{self.target}::{in_detail}::" if (in_detail is not None and t.text in self.detail_names) else ""
                edits.append((t.start, t.end, f"{qual}{t.text}<{self.lane}>"))
                continue
            if qualified_before and prev.text in ("::",) and n >= 2 and T[sig[n - 2]].kind == "id":
                self.report.dep(f"{T[sig[n - 2]].text}::{t.text}", self.src_rel, t.line)
            # calls into the generated API: qualify, which also switches ADL off
            # (with N = NN, ADL through mkfit::PropagationFlags would find MkFitCore's own
            # non-template function and prefer it)
            if not qualified_before and nxt is not None and nxt.text in ("(", "<"):
                explicit = f"<{self.lane}>" if (t.text in self.nondeducible and nxt.text == "(" and templated) else ""
                if in_detail is not None and t.text in self.detail_names:
                    edits.append((t.start, t.start, f"{self.target}::{in_detail}::"))
                    if explicit:
                        edits.append((t.end, t.end, explicit))
                    continue
                if t.text in self.api_names:
                    edits.append((t.start, t.start, f"{self.target}::"))
                    if explicit:
                        edits.append((t.end, t.end, explicit))
                    continue
                if nxt.text == "(" and t.text not in BUILTIN_TYPES and t.text not in local \
                        and t.text not in as_value \
                        and t.text not in tset and t.text not in ("if", "for", "while", "switch", "return",
                                                                  "sizeof", "static_cast", "reinterpret_cast",
                                                                  "const_cast", "defined", "decltype", "alignof"):
                    self.report.dep(t.text + "()", self.src_rel, t.line)
            if t.text in DEVICE_SUSPECT:
                self.report.note(self.src_rel, t.line, f"device: {t.text}: {DEVICE_SUSPECT[t.text]}")
        # preprocessor lines inside the span
        for k in range(a, b):
            t = T[k]
            if t.kind != "pp":
                continue
            m = re.match(r'(\s*#\s*include\s*)"([^"/]+)"', t.text)
            if m:
                inc_dir = os.path.dirname(self.src_rel)
                if m.group(2).endswith(".ah"):
                    # a generated matrix kernel: use the lane-guarded copy (see transform_kernel)
                    self.kernels.add(f"{inc_dir}/{m.group(2)}")
                    edits.append((t.start, t.end, f'{m.group(1)}"leafx_kernels/{m.group(2)}"'))
                else:
                    edits.append((t.start, t.end, f'{m.group(1)}"{inc_dir}/{m.group(2)}"'))

    def vla_fix(self, body, edits, fname, params):
        """Arrays sized by a runtime value: size N (a lane count bounds them), or the //@leafx size()."""
        a, b = body
        T = self.toks
        sig = self.p.sig(a + 1, b)
        for n, k in enumerate(sig):
            t = T[k]
            if t.text != "[" or n < 2:
                continue
            name, typ = T[sig[n - 1]], T[sig[n - 2]]
            before = T[sig[n - 3]] if n >= 3 else None
            if name.kind != "id" or typ.kind != "id" or (typ.text not in BUILTIN_TYPES):
                continue
            if before is not None and before.text not in (";", "{", "}", "const", ")"):
                continue
            close = self.p.match[k]
            inner = [T[j] for j in self.p.sig(k + 1, close)]
            if t.line in self.size_marks:
                edits.append((T[k + 1].start if k + 1 < close else T[close].start, T[close].start,
                              self.size_marks[t.line]))
                self.report.note(self.src_rel, t.line, f"{fname}: {name.text}[...] sized by markup: "
                                 f"{self.size_marks[t.line]}")
                continue
            runtime = [x.text for x in inner if x.kind == "id" and x.text in params]
            if runtime:
                edits.append((T[k + 1].start, T[close].start, self.lane))
                self.report.note(self.src_rel, t.line,
                                 f"{fname}: {name.text}[{''.join(x.text for x in inner)}] -> [{self.lane}] "
                                 f"(runtime size; bounded by the lane count)")

    def params_of(self, it):
        close = self.p.match[it.paren]
        return {self.toks[k].text for k in self.p.sig(it.paren + 1, close) if self.toks[k].kind == "id"}

    def head_fix(self, it, templated, edits, types):
        """template <int N> + annotation + specifier cleanup on a function head.
        Returns the indices of tokens it rewrote, and the declared name (never qualified)."""
        protected = {it.name_tok}
        T = self.toks
        sig = self.p.sig(it.start, it.paren)
        spec = it.spec_start
        # leading `template <...>`: add the lane parameter to it
        has_template = T[sig[0]].text == "template"
        if templated:
            if has_template:
                lt = sig[1]
                edits.append((T[lt].end, T[lt].end, f"int {self.lane}, "))
            else:
                edits.append((T[spec].start, T[spec].start, f"template <int {self.lane}>\n"))
        # specifiers between the template header and the name
        words = [k for k in sig if k >= spec and k < it.name_tok]
        has_inline = any(T[k].text == "inline" for k in words)
        for k in words:
            if T[k].text == "static":
                edits.append((T[k].start, T[k].end + (1 if T[k + 1].kind == "ws" else 0), ""))
        ins = self.annot + " "
        if not templated and not has_inline and it.kind == "func":
            ins += "inline "
        edits.append((T[spec].start, T[spec].start, ins))
        # pointer parameters to a Matriplex: non-deduced, so that a nullptr argument works
        if templated:
            close = self.p.match[it.paren]
            psig = self.p.sig(it.paren + 1, close)
            for n, k in enumerate(psig):
                t = T[k]
                if t.text in types and n + 1 < len(psig) and T[psig[n + 1]].text == "*":
                    edits.append((t.start, t.end, f"leafx_nd<{t.text}<{self.lane}>>"))
                    protected.add(k)
        return protected

    # ---- assembly ---------------------------------------------------------

    def render(self, a, b, edits):
        """Text of tokens [a, b) with edits applied."""
        start, end = self.toks[a].start, self.toks[b - 1].end
        es = [e for e in edits if start <= e[0] <= end]
        es.sort(key=lambda e: (e[0], e[1]))
        out, pos = [], start
        for s, f, txt in es:
            if s < pos:
                continue  # overlapping edit: first one wins
            out.append(self.text[pos:s])
            out.append(txt)
            pos = f
        out.append(self.text[pos:end])
        return "".join(out)

    def item_text(self, it, types, in_detail):
        T = self.toks
        edits = []
        if it.kind in ("func", "decl"):
            close = self.p.match[it.paren]
            sig_types = self.uses_mplex(it.start, close + 1, types)
            body_types = it.body is not None and self.uses_mplex(it.body[0], it.body[1] + 1, types)
            templated = sig_types or body_types
            if body_types and not sig_types:
                self.report.note(self.src_rel, T[it.start].line,
                                 f"{it.name}: {self.lane} is not deducible from its arguments; "
                                 f"calls from generated code get <{self.lane}>")
            protected = self.head_fix(it, templated, edits, types)
            self.rewrite_span(it.start, it.end, types, templated, in_detail, edits, it.name, protected)
            if it.body is not None:
                self.vla_fix(it.body, edits, it.name, self.params_of(it))
                used = sorted({T[k].text for k in self.p.sig(it.body[0], it.body[1])
                               if T[k].kind == "id" and T[k].text in self.local_globals})
                if used:
                    if not templated:
                        self.report.error(self.src_rel, T[it.start].line,
                                          f"{it.name}: uses {', '.join(used)} but is not a template on {self.lane}")
                    decl = "".join(f"\n{self.local_globals[u]}<{self.lane}> {u};  // leafx: was namespace-scope"
                                   for u in used)
                    edits.append((T[it.body[0]].end, T[it.body[0]].end, decl))
            return self.render(it.start, it.end, edits), templated
        if it.kind == "alias":
            self.rewrite_span(it.start, it.end, types, None, in_detail, edits, it.name or "alias")
            txt = self.render(it.start, it.end, edits)
            if self.uses_mplex(it.start, it.end, types):
                txt = f"template <int {self.lane}>\n" + txt
                return txt, True
            return txt, None
        return None, None

    def emit_items(self, items, in_detail, out, types, prev_end):
        for it in items:
            marks = self.preceding_marks(it, prev_end)
            prev_end = it.end
            if it.kind == "ns":
                raise RuntimeError(f"{self.src_rel}:{self.line(it.start)}: nested namespace inside a mapped one")
            if it.kind in ("func", "decl") and (it.name in self.skip_symbols or "skip" in marks):
                self.report.note(self.src_rel, self.line(it.start), f"skipped {it.name} (markup / config)")
                continue
            if it.kind == "other":
                words = [self.toks[k].text for k in self.p.sig(it.start, it.end)]
                if words[:2] == ["using", "namespace"] and words[2] in self.drop_usings:
                    continue
                g = self.global_matriplex(it, types)
                if g:
                    self.report.note(self.src_rel, self.line(it.start),
                                     f"namespace-scope {g[1]} {g[0]}: a local variable in each function that uses it")
                    continue
                self.report.note(self.src_rel, self.line(it.start), f"dropped: {' '.join(words)[:80]}")
                continue
            if it.kind == "class":
                words = [self.toks[k].text for k in self.p._skip_template_headers(self.p.sig(it.start, it.end))]
                if words and words[0] == "enum":
                    out.append(f"// leafx: {self.src_rel}:{self.line(it.start)}\n{self.render(it.start, it.end, [])}\n")
                    continue
                self.report.error(self.src_rel, self.line(it.start), f"class {it.name}: not handled in leaf mode")
                continue
            txt, templ = self.item_text(it, types, in_detail)
            if txt is None:
                continue
            if it.kind == "alias" and templ:
                types.add(it.name)
            out.append(f"// leafx: {self.src_rel}:{self.line(it.start)}\n{txt}\n")
        return prev_end

    def global_matriplex(self, it, types):
        """(name, type) if statement `it` declares a namespace-scope variable of a Matriplex type."""
        if not self.cfg.get("localize_globals", True):
            return None
        sig = self.p.sig(it.start, it.end)
        ids = [self.toks[k].text for k in sig if self.toks[k].kind == "id"]
        if self.toks[sig[-1]].text != ";" or any(self.toks[k].text in ("(", "=", "{", "<") for k in sig):
            return None
        mtypes = [x for x in ids if x in types]
        if len(mtypes) != 1 or ids[-1] in types:
            return None
        return ids[-1], mtypes[0]

    def collect_nondeducible(self, items, types):
        """Functions that need the lane count but cannot deduce it from their arguments."""
        for it in items:
            if it.kind == "ns":
                self.collect_nondeducible(it.children, types)
            elif it.kind in ("func", "decl") and it.body is not None:
                close = self.p.match[it.paren]
                if self.uses_mplex(it.body[0], it.body[1] + 1, types) and not self.uses_mplex(it.start, close + 1, types):
                    self.nondeducible.add(it.name)

    def collect_globals(self, items, types):
        for it in items:
            if it.kind == "ns":
                self.collect_globals(it.children, types)
            elif it.kind == "other":
                g = self.global_matriplex(it, types)
                if g:
                    self.local_globals[g[0]] = g[1]

    def angle_includes(self):
        """File-scope `#include <...>` lines of the source (external headers such as vdt)."""
        res = []
        for t in self.toks:
            if t.kind == "pp":
                m = re.match(r"\s*#\s*include\s*<([^>]+)>", t.text)
                if m:
                    res.append(m.group(1))
        return res

    def generate(self):
        out = []
        types = set(self.mplex_types)
        self.collect_globals(self.items, types)
        self.collect_nondeducible(self.items, types)
        # names of the anonymous namespaces and of file-scope functions (e.g. from a spliced .icc):
        # both go to the detail namespace
        for it in self.items:
            if it.kind == "ns" and it.name is None:
                self.collect_detail_names(it.children)
            elif it.kind in ("func", "decl"):
                self.detail_names.add(it.name)
        prev_end = 0
        file_scope = []
        for it in self.items + [None]:
            if it is not None and it.kind in ("func", "decl"):
                file_scope.append(it)
                continue
            if file_scope:  # a run of file-scope functions: into the detail namespace
                ns = f"{self.target}::{self.detail_ns}"
                body = []
                self.emit_items(file_scope, self.detail_ns, body, types, prev_end)
                out.append(f"namespace {ns} {{\n\n" + "\n".join(body) + f"\n}}  // namespace {ns}\n")
                prev_end = file_scope[-1].end
                file_scope = []
            if it is None:
                break
            if it.kind != "ns":
                if it.kind == "other" or it.kind == "alias":
                    self.report.note(self.src_rel, self.line(it.start), "file-scope statement dropped")
                prev_end = it.end
                continue
            if it.name is None:
                ns = f"{self.target}::{self.detail_ns}"
                in_detail = self.detail_ns
            elif it.name in self.cfg.get("namespace_map", {"mkfit": ""}):
                sub = self.cfg.get("namespace_map", {"mkfit": ""})[it.name]
                ns = self.target + (f"::{sub}" if sub else "")
                in_detail = self.detail_ns if self.detail_names else None
            else:
                raise RuntimeError(f"{self.src_rel}: namespace {it.name} has no mapping")
            body = []
            # aliases declared in the detail namespace stay visible to the file's public functions
            # through qualification (rewrite_span qualifies detail names)
            self.emit_items(it.children, in_detail if it.name is None else in_detail, body, types, it.open + 1)
            out.append(f"namespace {ns} {{\n\n" + "\n".join(body) + f"\n}}  // namespace {ns}\n")
            prev_end = it.end
        return "\n".join(out)


# =============================================================================
# Library mode: a whole header made callable from device code (Matriplex, vdt)
# =============================================================================


class LibraryXform:
    """Copies a header whole, annotating every function (free and member) with the host/device
    annotation, moving its namespaces under the target namespace, and renaming its include guard.
    Includes of other library files keep their relative form (the copies sit side by side); matrix
    kernels (.ah) get the lane-guarded copy."""

    def __init__(self, cfg, lcfg, report):
        self.cfg, self.lcfg, self.report = cfg, lcfg, report
        root = lcfg.get("source_root", cfg["source_root"])
        self.src_rel = lcfg["src"]
        self.src = os.path.join(root, self.src_rel)
        self.annot = cfg.get("annotation", "MKFIT_HOST_DEVICE")
        self.host_only = set(lcfg.get("host_only", []))
        self.ns_map = lcfg.get("namespace_map", {})
        self.kernels = set()
        with open(self.src) as fh:
            self.text = fh.read()
        self.toks = tokenize(self.text)
        self.p = Parser(self.toks)
        self.items = self.p.parse(0, len(self.toks), "file")

    def generate(self, guard_prefix):
        edits = []
        n_annot = [0]

        def walk(items):
            for it in items:
                if it.kind in ("ns", "class"):
                    if it.kind == "ns" and it.name in self.ns_map:
                        # `namespace X {` -> `namespace TARGET::X {`
                        edits.append((self.toks[it.start].start, self.toks[it.open].start,
                                      f"namespace {self.ns_map[it.name]} "))
                    walk(it.children)
                elif it.kind == "other":
                    # a namespace-scope constant table: `const T name[...] = {...};`
                    sig = self.p.sig(it.start, it.end)
                    words = [self.toks[k].text for k in sig]
                    if "const" in words[:3] and "[" in words and "=" in words and "(" not in words[:words.index("=")]:
                        c = self.cfg.get("device_const", "MKFIT_DEVICE_CONST")
                        edits.append((self.toks[it.start].start, self.toks[it.start].start, c + " "))
                        self.report.note(self.src_rel, self.toks[it.start].line,
                                         f"constant table {words[words.index('[') - 1]}: {c}")
                elif it.kind in ("func", "decl"):
                    if it.name in self.host_only or it.name.split("::")[-1] in self.host_only:
                        self.report.note(self.src_rel, self.toks[it.start].line, f"{it.name}: host only, not annotated")
                        continue
                    words = {self.toks[k].text for k in self.p.sig(it.spec_start, it.name_tok)}
                    if self.annot in words or "__host__" in words or "__device__" in words:
                        continue
                    if it.kind == "decl":  # `= default` / `= delete`: implicitly host/device, no annotation
                        tail = [self.toks[k].text for k in self.p.sig(self.p.match[it.paren], it.end)]
                        if "default" in tail or "delete" in tail:
                            continue
                    edits.append((self.toks[it.spec_start].start, self.toks[it.spec_start].start, self.annot + " "))
                    n_annot[0] += 1
        walk(self.items)

        # `#ifdef MPLEX_INTRINSICS` inside a function body: under `if constexpr` on the lane count,
        # as in the matrix kernels (the intrinsics step a whole vector of lanes)
        bodies = []

        def collect_bodies(items):
            for it in items:
                if it.kind in ("ns", "class"):
                    collect_bodies(it.children)
                elif it.kind == "func":
                    bodies.append((self.toks[it.body[0]].start, self.toks[it.body[1]].end, it))
        collect_bodies(self.items)
        intr = self.cfg.get("intrinsics_macro", "MPLEX_INTRINSICS")
        width = self.cfg.get("intrinsics_width", "MPLEX_INTRINSICS_WIDTH_BYTES / sizeof(T)")
        lane = self.lcfg.get("lane_param", "N")
        pp_all = [t for t in self.toks if t.kind == "pp"]
        for i, t in enumerate(pp_all):
            if not re.match(r"\s*#\s*ifdef\s+" + intr + r"\s*$", t.text):
                continue
            if not any(a <= t.start < b for a, b, _ in bodies):
                continue
            depth, i_else, i_end = 0, None, None
            for j in range(i, len(pp_all)):
                d = DIRECTIVE_RE.match(pp_all[j].text)
                if not d:
                    continue
                if d.group(1) in ("if", "ifdef", "ifndef"):
                    depth += 1
                elif d.group(1) == "else" and depth == 1:
                    i_else = j
                elif d.group(1) == "endif":
                    depth -= 1
                    if depth == 0:
                        i_end = j
                        break
            guard = f"if constexpr ({lane} % ({width}) == 0) {{"
            edits.append((t.start, t.end, f"#ifdef {intr}\n{guard}"))
            if i_else is None:
                edits.append((pp_all[i_end].start, pp_all[i_end].end, "}\n#endif"))
            else:
                edits.append((pp_all[i_else].start, pp_all[i_else].end, "} else\n#endif\n{"))
                edits.append((pp_all[i_end].start, pp_all[i_end].end, "}"))
            self.report.note(self.src_rel, t.line, f"#ifdef {intr} in a function body: under if constexpr on {lane}")

        # the include guard: the first `#ifndef X` directly followed by `#define X`
        pps = [t for t in self.toks if t.kind == "pp"]
        for a, b in zip(pps, pps[1:]):
            ma = re.match(r"\s*#\s*ifndef\s+(\w+)", a.text)
            mb = re.match(r"\s*#\s*define\s+(\w+)\s*$", b.text)
            if ma and mb and ma.group(1) == mb.group(1):
                g = ma.group(1)
                edits.append((a.start, a.end, a.text.replace(g, guard_prefix + g)))
                edits.append((b.start, b.end, b.text.replace(g, guard_prefix + g)))
                break
        # in-file includes of matrix kernels
        for t in pps:
            m = re.match(r'\s*#\s*include\s*"([^"/]+\.ah)"', t.text)
            if m:
                self.kernels.add(os.path.join(os.path.dirname(self.src), m.group(1)))
        edits.sort(key=lambda e: (e[0], e[1]))
        out, pos = [], 0
        for a, b, txt in edits:
            if a < pos:
                continue
            out.append(self.text[pos:a])
            out.append(txt)
            pos = b
        out.append(self.text[pos:])
        self.report.note(self.src_rel, 1, f"library: {n_annot[0]} functions annotated")
        return "".join(out)


# =============================================================================
# Driver
# =============================================================================


def transform_kernel(text, lane, intr_macro, width_expr):
    """A generated Matriplex kernel (.ah) has an intrinsics branch that steps through the lanes
    a whole vector at a time, and a plain `#pragma omp simd` branch, chosen by #ifdef. The
    intrinsics branch is only valid when the lane count is a multiple of the vector width, so
    it is put under `if constexpr` on the lane count: N = NN takes the intrinsics as before,
    N = 1 the plain loop."""
    lines = text.split("\n")
    depth, top = 0, []
    for i, line in enumerate(lines):
        m = DIRECTIVE_RE.match(line)
        if not m:
            continue
        kind, rest = m.group(1), m.group(2).strip()
        if kind in ("if", "ifdef", "ifndef"):
            if depth == 0:
                top.append([i, None, None, kind, rest])
            depth += 1
        elif kind == "else" and depth == 1:
            top[-1][1] = i
        elif kind == "endif":
            depth -= 1
            if depth == 0:
                top[-1][2] = i
    for i_if, i_else, i_end, kind, rest in reversed(top):
        if kind != "ifdef" or rest != intr_macro:
            continue
        guard = f"if constexpr ({lane} % ({width_expr}) == 0) {{"
        if i_else is None:
            lines[i_end] = "}\n#endif"
            lines[i_if] = f"#ifdef {intr_macro}\n{guard}"
        else:
            lines[i_end] = "}"
            lines[i_else] = "} else\n#endif\n{"
            lines[i_if] = f"#ifdef {intr_macro}\n{guard}"
    return "\n".join(lines)


def annotation_block(cfg):
    a = cfg.get("annotation", "MKFIT_HOST_DEVICE")
    c = cfg.get("device_const", "MKFIT_DEVICE_CONST")
    return (f"#ifndef {a}\n#if defined(__CUDACC__) || defined(__HIPCC__)\n#define {a} __host__ __device__\n"
            f"#else\n#define {a}\n#endif\n#endif\n"
            f"// namespace-scope constant tables: in device memory in the device compilation pass\n"
            f"#ifndef {c}\n#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)\n#define {c} __device__\n"
            f"#else\n#define {c}\n#endif\n#endif\n")


def mplex_typedefs(path, lane_const):
    """`typedef Matriplex::Matriplex<float, 6, 1, NN> MPlexLV;` -> (MPlexLV, template text)."""
    res = []
    with open(path) as fh:
        for m in re.finditer(r"typedef\s+(Matriplex::Matriplex(?:Sym)?)\s*<([^;]*?),\s*" + lane_const + r"\s*>\s+(\w+)\s*;",
                             fh.read()):
            res.append((m.group(3), m.group(1), m.group(2)))
    return res


def git_info(root, rel):
    try:
        blob = subprocess.run(["git", "hash-object", rel], cwd=root, capture_output=True, text=True).stdout.strip()
        commit = subprocess.run(["git", "log", "-1", "--format=%h %cs", "--", rel], cwd=root, capture_output=True,
                                text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=root, capture_output=True,
                               text=True).stdout.strip()
        return blob, commit + (" (modified)" if dirty else "")
    except OSError:
        return "?", "?"


def stamp(cfg, srcs):
    lines = [f"// GENERATED by leafx {VERSION} -- do not edit; regenerate with leafx.py.", "// Sources:"]
    for rel in srcs:
        blob, commit = git_info(cfg["source_root"], rel)
        lines.append(f"//   {rel}  blob {blob}  last commit {commit}")
    return "\n".join(lines) + "\n"


# CMSSW's style, for trees without a .clang-format
FALLBACK_STYLE = "{BasedOnStyle: Google, ColumnLimit: 120, IndentWidth: 2, NamespaceIndentation: All, " \
                 "AllowShortFunctionsOnASingleLine: All, DerivePointerAlignment: false, PointerAlignment: Left}"


def clang_format(path, style_file):
    style = f"file:{style_file}" if style_file and os.path.exists(style_file) else FALLBACK_STYLE
    try:
        subprocess.run(["clang-format", "-i", f"--style={style}", path], check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"warning: clang-format failed on {path}: {e}", file=sys.stderr)


def verify_with_libclang(fx, cfg):
    """Cross-check: libclang's function definitions in the file vs the tokenizer's."""
    try:
        import clang.cindex as ci
    except ImportError:
        print("  verify: python3-clang not available, skipped")
        return
    try:
        rdir = subprocess.run(["clang", "-print-resource-dir"], capture_output=True, text=True).stdout.strip()
    except OSError:
        rdir = ""
    args = ["-x", "c++", f"-std={cfg.get('verify_std', 'c++20')}"] + ([f"-resource-dir={rdir}"] if rdir else []) + \
           [f"-I{p}" for p in cfg.get("include_paths", [])] + [f"-D{d}" for d in cfg.get("define", [])]
    tu = ci.Index.create().parse(fx.src, args=args)
    errs = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
    if errs:
        print(f"  verify {fx.src_rel}: libclang reports {len(errs)} errors, first: {errs[0]}")
    files = {fx.src} | {os.path.join(cfg["source_root"], s) for s in fx.spliced}
    # with a spliced file the line numbers shift, so compare names (with multiplicity) only
    key = (lambda name, line: (name, 0)) if fx.spliced else (lambda name, line: (name, line))
    from collections import Counter
    want = Counter()
    for c in tu.cursor.walk_preorder():
        if c.location.file and c.location.file.name in files and c.is_definition() and \
                c.kind in (ci.CursorKind.FUNCTION_DECL, ci.CursorKind.FUNCTION_TEMPLATE):
            want[key(c.spelling, c.extent.end.line)] += 1
    have = Counter()

    def walk(items):
        for it in items:
            if it.kind == "ns":
                walk(it.children)
            elif it.kind == "func":
                have[key(it.name.split("::")[-1], fx.toks[it.body[1]].line)] += 1
    walk(fx.items)
    # lines in fx are those of the source (pre-passes keep the line count)
    miss, extra = want - have, have - want
    print(f"  verify {fx.src_rel}: libclang {sum(want.values())} definitions, leafx {sum(have.values())}; "
          f"missed {sorted(miss.elements())} extra {sorted(extra.elements())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--source-root", help="the CMSSW src/ tree to read; overrides $LEAFX_SOURCE_ROOT and the config")
    ap.add_argument("-I", dest="includes", action="append", default=[],
                    help="extra include path for --verify (e.g. where vdt/ lives)")
    ap.add_argument("--verify", action="store_true", help="cross-check function extents with libclang")
    ap.add_argument("--check", action="store_true",
                    help="regenerate into a temporary directory and report whether --outdir is up to date")
    args = ap.parse_args()
    if args.check:
        import filecmp
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            sub = [sys.executable, os.path.abspath(__file__), args.config, "-o", tmp]
            if args.source_root:
                sub += ["--source-root", args.source_root]
            subprocess.run(sub, check=True, capture_output=True)
            stale = []
            for dirpath, _, fnames in os.walk(tmp):
                for f in fnames:
                    new = os.path.join(dirpath, f)
                    old = os.path.join(args.outdir, os.path.relpath(new, tmp))
                    if f == "leafx_report.txt":
                        continue
                    if not os.path.exists(old) or not filecmp.cmp(new, old, shallow=False):
                        stale.append(os.path.relpath(new, tmp))
        if stale:
            print(f"STALE: {len(stale)} generated files differ from what the current sources give:")
            for f in sorted(stale):
                print(f"  {f}")
            return 1
        print(f"up to date: {args.outdir}")
        return 0

    with open(args.config, "rb") as fh:
        cfg = tomllib.load(fh)
    # The tree to read: --source-root, else $LEAFX_SOURCE_ROOT, else the config's source_root
    # (relative to the config file). Relative paths in the config are relative to the tree.
    root = args.source_root or os.environ.get("LEAFX_SOURCE_ROOT")
    if root is None:
        if "source_root" not in cfg:
            sys.exit("leafx: no source tree: give --source-root, set LEAFX_SOURCE_ROOT, or set source_root in the config")
        root = os.path.join(os.path.dirname(os.path.abspath(args.config)), cfg["source_root"])
    cfg["source_root"] = os.path.abspath(root)
    cfg["include_paths"] = [os.path.join(cfg["source_root"], p) for p in cfg.get("include_paths", ["."])] + \
        [os.path.abspath(p) for p in args.includes]
    for lcfg in cfg.get("library", []):
        if "source_root" in lcfg:
            lcfg["source_root"] = os.path.expandvars(lcfg["source_root"])
            if not os.path.isabs(lcfg["source_root"]):
                lcfg["source_root"] = os.path.join(cfg["source_root"], lcfg["source_root"])
    if cfg.get("clang_format_style"):
        cfg["clang_format_style"] = os.path.join(cfg["source_root"], cfg["clang_format_style"])
    os.makedirs(args.outdir, exist_ok=True)
    report = Report()
    target = cfg.get("target_namespace", "mkfit::portable")
    lane = cfg.get("lane_param", "N")
    lane_const = cfg.get("lane_constant", "NN")
    guard_pref = cfg.get("guard_prefix", "LEAFX_GENERATED_")

    # ---- the Matriplex type aliases ----
    tdefs, consts = [], []
    for rel in cfg["mplex_typedefs"]:
        tdefs += mplex_typedefs(os.path.join(cfg["source_root"], rel), lane_const)
        # the dimension constants the typedefs use (`constexpr Matriplex::idx_t LL = 6;`), not the lane count
        with open(os.path.join(cfg["source_root"], rel)) as fh:
            for m in re.finditer(r"constexpr\s+([\w:]+)\s+(\w+)\s*=\s*([^;]+);", fh.read()):
                if m.group(2) != lane_const and "MPT_SIZE" not in m.group(3):
                    consts.append(m.groups())
    mplex_types = [t[0] for t in tdefs]
    types_h = os.path.join(args.outdir, "leafx_types.h")
    with open(types_h, "w") as fh:
        fh.write(stamp(cfg, cfg["mplex_typedefs"]))
        fh.write(f"#ifndef {guard_pref}leafx_types_h\n#define {guard_pref}leafx_types_h\n\n")
        fh.write(annotation_block(cfg) + "\n")
        for inc in cfg.get("prelude_includes", []):
            fh.write(f'#include "{inc}"\n')
        fh.write("\n")
        fh.write(f"namespace {target} {{\n\n")
        fh.write("  // A non-deduced context: a pointer parameter to a Matriplex does not take part in the\n"
                 "  // deduction of the lane count, so that a nullptr argument works.\n"
                 "  template <class T>\n  struct leafx_id {\n    using type = T;\n  };\n"
                 "  template <class T>\n  using leafx_nd = typename leafx_id<T>::type;\n\n")
        for typ, name, val in consts:
            fh.write(f"  constexpr {typ} {name} = {val};\n")
        if consts:
            fh.write("\n")
        for name, cls, tparams in tdefs:
            fh.write(f"  template <int {lane}>\n  using {name} = {cls}<{tparams}, {lane}>;\n")
        fh.write(f"\n}}  // namespace {target}\n\n#endif\n")

    # ---- library copies (Matriplex, vdt for device code) ----
    for lcfg in cfg.get("library", []):
        lx = LibraryXform(cfg, lcfg, report)
        text = lx.generate(guard_pref)
        out = os.path.join(args.outdir, lcfg["out"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            fh.write(stamp(dict(cfg, source_root=lcfg.get("source_root", cfg["source_root"])), [lx.src_rel]) +
                     annotation_block(cfg) + text)
        for k in sorted(lx.kernels):
            with open(k) as fh:
                ktext = transform_kernel(fh.read(), lane, cfg.get("intrinsics_macro", "MPLEX_INTRINSICS"),
                                         cfg.get("intrinsics_width", "MPLEX_INTRINSICS_WIDTH_BYTES / sizeof(T)"))
            with open(os.path.join(os.path.dirname(out), os.path.basename(k)), "w") as fh:
                fh.write(f"// GENERATED by leafx {VERSION} from {k} -- do not edit.\n" + ktext)
    if cfg.get("library"):
        print(f"wrote {len(cfg['library'])} library copies")

    # ---- pass 1: parse all, collect the API (functions in the mkfit namespace) ----
    files = []
    api = set()
    for fcfg in cfg["file"]:
        fx = FileXform(cfg, fcfg, mplex_types, api, report)
        files.append(fx)
        for it in fx.items:
            if it.kind == "ns" and it.name == "mkfit":
                for c in it.children:
                    if c.kind in ("func", "decl") and c.name not in fx.skip_symbols:
                        api.add(c.name)

    # ---- pass 2: generate ----
    outputs = []
    for fx, fcfg in zip(files, cfg["file"]):
        if args.verify:
            verify_with_libclang(fx, cfg)
        body = fx.generate()
        out = os.path.join(args.outdir, fcfg["out"])
        guard = guard_pref + re.sub(r"\W", "_", fcfg["out"])
        with open(out, "w") as fh:
            fh.write(stamp(cfg, [fx.src_rel]))
            fh.write(f"#ifndef {guard}\n#define {guard}\n\n")
            fh.write('#include "leafx_types.h"\n')
            for inc in fcfg.get("includes", []):
                fh.write(f'#include "{inc}"\n')
            if cfg.get("keep_angle_includes", True):
                for inc in fx.angle_includes():
                    fh.write(f"#include <{inc}>\n")
            fh.write("\n" + body + f"\n#endif  // {guard}\n")
        clang_format(out, cfg.get("clang_format_style"))
        outputs.append(fcfg["out"])

    # ---- the matrix kernels included by generated functions ----
    kernels = sorted(set().union(*(fx.kernels for fx in files)))
    kdir = os.path.join(args.outdir, "leafx_kernels")
    os.makedirs(kdir, exist_ok=True)
    for rel in kernels:
        with open(os.path.join(cfg["source_root"], rel)) as fh:
            ktext = fh.read()
        ktext = transform_kernel(ktext, lane, cfg.get("intrinsics_macro", "MPLEX_INTRINSICS"),
                                 cfg.get("intrinsics_width", "MPLEX_INTRINSICS_WIDTH_BYTES / sizeof(T)"))
        with open(os.path.join(kdir, os.path.basename(rel)), "w") as fh:
            fh.write(stamp(cfg, [rel]) + ktext)
    print(f"wrote {len(kernels)} lane-guarded matrix kernels to {kdir}")

    with open(os.path.join(args.outdir, "leafx_all.h"), "w") as fh:
        fh.write(f"// GENERATED by leafx {VERSION}: all generated headers, in dependency order.\n")
        fh.write(f"#ifndef {guard_pref}leafx_all_h\n#define {guard_pref}leafx_all_h\n")
        for o in outputs:
            fh.write(f'#include "{o}"\n')
        fh.write("#endif\n")

    # ---- report ----
    rep = os.path.join(args.outdir, "leafx_report.txt")
    with open(rep, "w") as fh:
        fh.write(f"leafx {VERSION} report\n\nERRORS ({len(report.errors)})\n")
        for e in report.errors:
            fh.write(f"  {e}\n")
        fh.write("\nEXTERNAL NAMES used by the generated code (resolve to MkFitCore on the host;\n"
                 "a device build provides its own in the target namespace)\n")
        for name in sorted(report.deps):
            where = sorted(report.deps[name])
            fh.write(f"  {name:45s} {len(where):4d} uses, first {where[0][0]}:{where[0][1]}\n")
        fh.write(f"\nNOTES ({len(report.notes)})\n")
        for n in report.notes:
            fh.write(f"  {n}\n")
    print(f"wrote {len(outputs) + 2} headers to {args.outdir}; {len(report.errors)} errors; report {rep}")
    for e in report.errors:
        print("  ERROR", e)
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
