from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEYWORDS_PATH = REPO_ROOT / "evaluation" / "domain_keywords.json"

TOKEN_PATTERN = re.compile(
    r"#pragma omp|::|->|<=|>=|==|!=|\+\+|--|&&|\|\||[A-Za-z_][A-Za-z0-9_]*|\d+|[{}()\[\];,<>+\-*/%=&|!^~?:.]"
)
IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
CLASS_PATTERN = re.compile(r"\b(class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)")
FUNCTION_PATTERN = re.compile(
    r"(?:^|[\s;{}])(?:template\s*<[^>]+>\s*)?(?:[\w:<>,~*&\s]+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*(?:const)?\s*[{:]",
    flags=re.MULTILINE,
)
ASSIGN_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);")
DECL_PATTERN = re.compile(
    r"\b(?:int|long|long long|size_t|std::size_t|bool|float|double|char|auto|std::atomic<[^>]+>)\s+([A-Za-z_][A-Za-z0-9_]*)"
)


def tokenize_cpp(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def counter_f1(reference: Counter, candidate: Counter) -> float:
    if not reference and not candidate:
        return 1.0
    overlap = sum(min(reference[token], candidate[token]) for token in reference.keys() | candidate.keys())
    ref_total = sum(reference.values())
    cand_total = sum(candidate.values())
    if ref_total == 0 or cand_total == 0:
        return 0.0
    precision = overlap / cand_total
    recall = overlap / ref_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ngram_score(reference_code: str, candidate_code: str, max_n: int = 4) -> float:
    ref_tokens = tokenize_cpp(reference_code)
    cand_tokens = tokenize_cpp(candidate_code)
    scores = []
    for n in range(1, max_n + 1):
        scores.append(counter_f1(Counter(ngrams(ref_tokens, n)), Counter(ngrams(cand_tokens, n))))
    return sum(scores) / len(scores)


def extract_ast_signature(code: str) -> Counter:
    lowered = code.lower()
    signature = Counter()

    for keyword in [
        "if",
        "else",
        "for",
        "while",
        "return",
        "class",
        "struct",
        "template",
        "thread",
        "mutex",
        "atomic",
        "condition_variable",
        "future",
        "async",
        "parallel",
        "omp",
        "mpi",
    ]:
        count = len(re.findall(rf"\b{re.escape(keyword)}\b", lowered))
        if count:
            signature[f"kw:{keyword}"] = count

    for kind, name in CLASS_PATTERN.findall(code):
        signature[f"{kind}:{name}"] += 1

    for name in FUNCTION_PATTERN.findall(code):
        signature[f"fn:{name}"] += 1

    for primitive in ["std::thread", "std::mutex", "std::atomic", "std::future", "MPI_", "#pragma omp"]:
        count = code.count(primitive)
        if count:
            signature[f"api:{primitive}"] = count

    return signature


def ast_score(reference_code: str, candidate_code: str) -> float:
    return counter_f1(extract_ast_signature(reference_code), extract_ast_signature(candidate_code))


def syntax_profile(code: str) -> Counter:
    profile = Counter()
    for token in "{}()[];,:<>":
        profile[f"punct:{token}"] = code.count(token)
    for operator in ["==", "!=", "<=", ">=", "+", "-", "*", "/", "%", "&&", "||", "&", "|"]:
        profile[f"op:{operator}"] = code.count(operator)
    lines = [line for line in code.splitlines() if line.strip()]
    profile["line_count"] = len(lines)
    profile["avg_indent"] = int(sum(len(line) - len(line.lstrip(" ")) for line in lines) / len(lines)) if lines else 0
    return profile


def syntax_score(reference_code: str, candidate_code: str) -> float:
    return counter_f1(syntax_profile(reference_code), syntax_profile(candidate_code))


def extract_dataflow_pairs(code: str) -> Counter:
    pairs = Counter()
    lowered = code.replace("->", " ").replace("::", " ")
    declared = set(DECL_PATTERN.findall(lowered))
    for declared_name in declared:
        pairs[f"decl:{declared_name}"] += 1

    for target, expression in ASSIGN_PATTERN.findall(lowered):
        rhs_variables = [name for name in IDENTIFIER_PATTERN.findall(expression) if name != target]
        if not rhs_variables:
            pairs[f"const->{target}"] += 1
        for source in rhs_variables:
            pairs[f"{source}->{target}"] += 1
    return pairs


def dataflow_score(reference_code: str, candidate_code: str) -> float:
    return counter_f1(extract_dataflow_pairs(reference_code), extract_dataflow_pairs(candidate_code))


def load_keyword_weights(path: Path = DEFAULT_KEYWORDS_PATH) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def keyword_score(reference_code: str, candidate_code: str, keyword_weights: dict[str, float]) -> float:
    reference_lower = reference_code.lower()
    candidate_lower = candidate_code.lower()

    reference_weight = 0.0
    candidate_weight = 0.0
    shared_weight = 0.0
    for keyword, weight in keyword_weights.items():
        in_reference = keyword.lower() in reference_lower
        in_candidate = keyword.lower() in candidate_lower
        if in_reference:
            reference_weight += weight
        if in_candidate:
            candidate_weight += weight
        if in_reference and in_candidate:
            shared_weight += weight

    if reference_weight == 0 and candidate_weight == 0:
        return 1.0
    if reference_weight + candidate_weight == 0:
        return 0.0
    return 2 * shared_weight / (reference_weight + candidate_weight)


def score_codeblue(
    candidate_path: Path,
    reference_path: Path,
    *,
    keywords_path: Path = DEFAULT_KEYWORDS_PATH,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    weights = weights or {
        "ngram": 0.25,
        "ast": 0.20,
        "dataflow": 0.20,
        "syntax": 0.15,
        "keyword": 0.20,
    }
    reference_code = reference_path.read_text(encoding="utf-8")
    candidate_code = candidate_path.read_text(encoding="utf-8")
    keyword_weights = load_keyword_weights(keywords_path)

    components = {
        "ngram_score": ngram_score(reference_code, candidate_code),
        "ast_score": ast_score(reference_code, candidate_code),
        "dataflow_score": dataflow_score(reference_code, candidate_code),
        "syntax_score": syntax_score(reference_code, candidate_code),
        "keyword_score": keyword_score(reference_code, candidate_code, keyword_weights),
    }
    final_score = (
        weights["ngram"] * components["ngram_score"]
        + weights["ast"] * components["ast_score"]
        + weights["dataflow"] * components["dataflow_score"]
        + weights["syntax"] * components["syntax_score"]
        + weights["keyword"] * components["keyword_score"]
    )
    components["final_score"] = final_score
    return components


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute the extended CodeBLUE score for a candidate solution.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = score_codeblue(Path(args.candidate), Path(args.reference))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
