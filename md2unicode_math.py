#!/usr/bin/env python3
import argparse, re, sys

# ---------- tables ----------
GREEK = {
    "alpha":"α","beta":"β","gamma":"γ","delta":"δ","epsilon":"ε","zeta":"ζ",
    "eta":"η","theta":"θ","iota":"ι","kappa":"κ","lambda":"λ","mu":"μ",
    "nu":"ν","xi":"ξ","omicron":"ο","pi":"π","rho":"ρ","sigma":"σ","tau":"τ",
    "upsilon":"υ","phi":"φ","chi":"χ","psi":"ψ","omega":"ω",
    "Gamma":"Γ","Delta":"Δ","Theta":"Θ","Lambda":"Λ","Xi":"Ξ","Pi":"Π",
    "Sigma":"Σ","Upsilon":"Υ","Phi":"Φ","Psi":"Ψ","Omega":"Ω",
}
OPS = {
    r"\le":"≤", r"\ge":"≥", r"\neq":"≠", r"\approx":"≈", r"\sim":"∼",
    r"\in":"∈", r"\notin":"∉", r"\subseteq":"⊆", r"\subset":"⊂",
    r"\supseteq":"⊇", r"\supset":"⊃", r"\cup":"∪", r"\cap":"∩",
    r"\vee":"∨", r"\wedge":"∧", r"\oplus":"⊕", r"\otimes":"⊗",
    r"\forall":"∀", r"\exists":"∃", r"\to":"→", r"\gets":"←", r"\mapsto":"↦",
    r"\implies":"⇒", r"\iff":"⇔", r"\Rightarrow":"⇒", r"\Leftarrow":"⇐",
    r"\cdot":"·", r"\times":"×", r"\pm":"±", r"\mp":"∓",
    r"\propto":"∝", r"\equiv":"≡", r"\cong":"≅",
    r"\sum":"∑", r"\prod":"∏", r"\int":"∫", r"\oint":"∮", r"\nabla":"∇",
    r"\partial":"∂", r"\infty":"∞", r"\ldots":"…", r"\cdots":"⋯",
    r"\{":"{", r"\}":"}", r"\langle":"⟨", r"\rangle":"⟩",
    r"\mid":"|", r"\vert":"|", r"\lvert":"|", r"\rvert":"|",
    r"\Vert":"‖", r"\lVert":"‖", r"\rVert":"‖",
    r"\Re":"ℜ", r"\Im":"ℑ", r"\log":"log", r"\ln":"ln",
}
MBB = {
    **{c:c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"},
    "R":"ℝ","Z":"ℤ","Q":"ℚ","N":"ℕ","C":"ℂ","P":"ℙ","H":"ℍ","B":"𝔹","D":"𝔻",
    "E":"𝔼","F":"𝔽","I":"𝕀","J":"𝕁","K":"𝕂","L":"𝕃","M":"𝕄","O":"𝕆",
    "S":"𝕊","T":"𝕋","U":"𝕌","V":"𝕍","W":"𝕎","X":"𝕏","Y":"𝕐",
    "a":"𝕒","b":"𝕓","c":"𝕔","d":"𝕕","e":"𝕖","f":"𝕗","g":"𝕘","h":"𝕙","i":"𝕚",
    "j":"𝕛","k":"𝕜","l":"𝕝","m":"𝕞","n":"𝕟","o":"𝕠","p":"𝕡","q":"𝕢","r":"𝕣",
    "s":"𝕤","t":"𝕥","u":"𝕦","v":"𝕧","w":"𝕨","x":"𝕩","y":"𝕪","z":"𝕫",
}
MCAL = {
    "A":"𝒜","B":"𝓑","C":"𝒞","D":"𝒟","E":"𝒠","F":"𝒡","G":"𝒢","H":"𝒣","I":"𝒤",
    "J":"𝒥","K":"𝒦","L":"𝒧","M":"𝒨","N":"𝒩","O":"𝒪","P":"𝒫","Q":"𝒬","R":"𝓡",
    "S":"𝒮","T":"𝒯","U":"𝒰","V":"𝒱","W":"𝒲","X":"𝒳","Y":"𝒴","Z":"𝒵"
}
SUP = {
    **{str(i): chr(0x2070 + (i if i else 10)) for i in range(0,10)},
    "+":"⁺","-":"⁻","=":"⁼","(":"⁽",")":"⁾",
    "a":"ᵃ","b":"ᵇ","c":"ᶜ","d":"ᵈ","e":"ᵉ","f":"ᶠ","g":"ᵍ","h":"ʰ","i":"ⁱ","j":"ʲ",
    "k":"ᵏ","l":"ˡ","m":"ᵐ","n":"ⁿ","o":"ᵒ","p":"ᵖ","r":"ʳ","s":"ˢ","t":"ᵗ","u":"ᵘ",
    "v":"ᵛ","w":"ʷ","x":"ˣ","y":"ʸ","z":"ᶻ",
    "A":"ᴬ","B":"ᴮ","D":"ᴰ","E":"ᴱ","G":"ᴳ","H":"ᴴ","I":"ᴵ","J":"ᴶ","K":"ᴷ","L":"ᴸ",
    "M":"ᴹ","N":"ᴺ","O":"ᴼ","P":"ᴾ","R":"ᴿ","T":"ᵀ","U":"ᵁ","V":"ⱽ","W":"ᵂ",
}
SUB = {
    "0":"₀","1":"₁","2":"₂","3":"₃","4":"₄","5":"₅","6":"₆","7":"₇","8":"₈","9":"₉",
    "+":"₊","-":"₋","=":"₌","(":"₍",")":"₎",":":"₍:₎",
    "a":"ₐ","e":"ₑ","h":"ₕ","i":"ᵢ","j":"ⱼ","k":"ₖ","l":"ₗ","m":"ₘ","n":"ₙ",
    "o":"ₒ","p":"ₚ","r":"ᵣ","s":"ₛ","t":"ₜ","u":"ᵤ","v":"ᵥ","x":"ₓ",
}

# combining marks for accents
ACCENTS = {
    "hat":"\u0302", "widehat":"\u0302",
    "tilde":"\u0303", "widetilde":"\u0303",
    "bar":"\u0305", "overline":"\u0305",
    "breve":"\u0306",
    "check":"\u030C",
    "acute":"\u0301",
    "grave":"\u0300",
    "dot":"\u0307",
    "ddot":"\u0308",
    "mathring":"\u030A",
    "vec":"\u20D7", "overrightarrow":"\u20D7",
    "underline":"\u0332",
}

# ---------- helpers ----------
def to_super(s):
    out = []
    for ch in s:
        if ch in SUP: out.append(SUP[ch])
        elif ch.lower() in SUP: out.append(SUP[ch.lower()])
        else: return "⁽" + s + "⁾"
    return "".join(out)

def to_sub(s):
    out = []
    for ch in s:
        if ch in SUB: out.append(SUB[ch])
        elif ch.lower() in SUB: out.append(SUB[ch.lower()])
        else: return "₍" + s + "₎"
    return "".join(out)

def strip_braces(s):
    return s[1:-1] if s and s[0] == "{" and s[-1] == "}" else s

def apply_combining_per_char(txt, comb):
    # apply combining mark to each non-space
    res = []
    for ch in txt:
        if ch.isspace():
            res.append(ch)
        else:
            res.append(ch + comb)
    return "".join(res)

def replace_mathcal(s):
    return re.sub(r"\\mathcal\{([A-Za-z])\}", lambda m: MCAL.get(m.group(1), m.group(1)), s)

def replace_mathbb(s):
    return re.sub(r"\\mathbb\{([A-Za-z])\}", lambda m: MBB.get(m.group(1), m.group(1)), s)

def replace_greek(s):
    for name, ch in GREEK.items():
        s = re.sub(rf"\\{name}\b", ch, s)
    return s

def replace_ops_and_sizes(s):
    # special: sums/products/integrals with limits first
    s = re.sub(r"\\sum(?:_({[^}]+}|.))?", lambda m: "∑" + (to_sub(strip_braces(m.group(1))) if m.group(1) else ""), s)
    s = re.sub(r"\\prod(?:_({[^}]+}|.))?", lambda m: "∏" + (to_sub(strip_braces(m.group(1))) if m.group(1) else ""), s)
    s = re.sub(r"\\int(?:_({[^}]+}|.))?(?:\^({[^}]+}|.))?", lambda m:
               "∫" + (to_sub(strip_braces(m.group(1))) if m.group(1) else "") +
               (to_super(strip_braces(m.group(2))) if m.group(2) else ""), s)
    for k, v in OPS.items():
        s = s.replace(k, v)
    s = s.replace(r"\|", "‖")
    # strip \left \right and common spacing
    s = re.sub(r"\\(left|right)\b", "", s)
    s = s.replace(r"\!", "").replace(r"\,", " ").replace(r"\;", " ").replace(r"\:", " ")
    return s

def replace_fonts_and_text(s):
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    return s

def replace_accents(s):
    # braced: \hat{...}, unbraced: \hat x
    for name, comb in ACCENTS.items():
        # braced
        pat1 = re.compile(rf"\\{name}\s*\{{([^{{}}]+)\}}")
        s = pat1.sub(lambda m: apply_combining_per_char(m.group(1), comb), s)
        # single-char
        pat2 = re.compile(rf"\\{name}\s*([A-Za-z0-9])")
        s = pat2.sub(lambda m: apply_combining_per_char(m.group(1), comb), s)
    return s

def replace_subsup(s):
    def sup_repl(m):
        a = strip_braces(m.group(1)) if m.group(1) else m.group(2)
        return to_super(a)
    def sub_repl(m):
        a = strip_braces(m.group(1)) if m.group(1) else m.group(2)
        return to_sub(a)
    s = re.sub(r"\^\{([^{}]+)\}|\^([A-Za-z0-9+\-()=])", sup_repl, s)
    s = re.sub(r"_\{([^{}]+)\}|_([A-Za-z0-9+\-()=:])", sub_repl, s)
    return s

def convert_math(expr):
    s = expr
    s = replace_fonts_and_text(s)
    s = replace_mathcal(s)
    s = replace_mathbb(s)
    s = replace_greek(s)
    s = replace_ops_and_sizes(s)
    s = replace_accents(s)     # <<< accents before sub/sup so they bind to base
    s = replace_subsup(s)
    s = s.replace(r"\{","{").replace(r"\}","}")
    return s

MATH_RE = re.compile(r"(?s)(?<!\\)\$\$(.+?)(?<!\\)\$\$|(?<!\\)\$(.+?)(?<!\\)\$")

def process(md, keep_dollar=False):
    out, i = [], 0
    for m in MATH_RE.finditer(md):
        out.append(md[i:m.start()])
        inner = m.group(1) if m.group(1) is not None else m.group(2)
        converted = convert_math(inner)
        if keep_dollar:
            delim = "$$" if m.group(1) is not None else "$"
            out.append(delim + converted + delim)
        else:
            out.append(converted)
        i = m.end()
    out.append(md[i:])
    return "".join(out)

def main():
    ap = argparse.ArgumentParser(description="Markdown + LaTeX math → Unicode plaintext (email-safe)")
    ap.add_argument("input")
    ap.add_argument("-o","--output")
    ap.add_argument("--keep-dollar", action="store_true", help="keep $/$$(…)$ delimiters")
    args = ap.parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        md = f.read()
    txt = process(md, keep_dollar=args.keep_dollar)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as g:
            g.write(txt)
    else:
        sys.stdout.write(txt)

if __name__ == "__main__":
    main()

