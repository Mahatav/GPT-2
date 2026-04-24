"""Post-training evaluation script for comparing Western vs Eastern philosophical bias."""

import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

from inference_utils import generate, get_tokenizer, load_model
from kl_divergence import compute_kl_report
from stats_analysis import compute_overlap_metrics, analyze_category

# ── Paths — auto-detect based on what exists on the current machine ──────────
from pathlib import Path as _Path


def _find_outputs_root() -> _Path:
    """Return the outputs_full root, searching common locations."""
    candidates = [
        _Path.home() / "outputs_full",                          # server: ~/outputs_full
        _Path(__file__).parent / "outputs",                     # local:  ./outputs
        _Path("/home/marora15/outputs_full"),                    # server absolute
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]   # fall back to ~/outputs_full even if missing


_OUTPUTS_ROOT = _find_outputs_root()
WESTERN_MODEL_PATH = str(_OUTPUTS_ROOT / "progressive_west/period_2000/checkpoint-2118")
EASTERN_MODEL_PATH  = str(_OUTPUTS_ROOT / "progressive_east/period_2000/checkpoint-1299")
OUTPUT_DIR          = str(_OUTPUTS_ROOT / "progressive_evaluations")

# Generation config — tuned to suppress looping on small GPT-2 models
GENERATION_CONFIG = {
    "temperature": 0.4,          # tighter distribution, less drift
    "repetition_penalty": 1.3,   # penalise already-seen tokens
    "no_repeat_ngram_size": 4,   # hard block on repeating any 4-gram
    "top_p": 0.9,                # nucleus sampling
    "do_sample": True,
}

# Vocabulary lists for concept frequency analysis
WESTERN_MARKERS = [
    "soul", "virtue", "reason", "logos", "socrates", "plato", "aristotle",
    "god", "justice", "polis", "republic", "form", "ideal", "rational",
    "empirical", "substance", "cause", "effect", "categorical", "imperative",
    "existence", "essence", "consciousness", "dialectic", "synthesis",
    "kant", "hegel", "descartes", "nietzsche", "aristotelian",
]

EASTERN_MARKERS = [
    "dharma", "karma", "nirvana", "moksha", "atman", "brahman", "tao", "zen",
    "buddha", "buddhist", "enlightenment", "samsara", "maya", "prajna", "sunyata",
    "confucius", "yin", "yang", "qi", "wu wei", "dukkha", "anatta", "anicca",
    "upanishad", "vedanta", "bodhisattva", "mandala", "mantra", "chakra",
    "meditation", "suffering", "craving", "impermanence", "rebirth",
    "mendicant", "noble", "eightfold", "cessation", "mindfulness",
    "detachment", "ancestor",
]

PHILOSOPHICAL_PROMPTS = {
    "self_identity": [
        "What is the true nature of the self?",
        "Who am I in the grand scheme of existence?",
        "What defines a person's identity?",
        "Is the self an illusion or ultimate reality?",
        "What is the relationship between mind and self?",
        "What is the difference between self and soul?",
        "Is the self permanent or impermanent?",
        "What is self-actualization?",
        "How does the self relate to consciousness?",
        "What is the egoless state?",
        "What is the relationship between self and ego?",
        "Is there a higher self beyond the ego?",
        "What is self-realization?",
        "How does the self perceive the world?",
        "What is the nature of personal identity?",
        "Can the self be transformed?",
        "What is the relationship between self and society?",
        "What is self-awareness?",
        "How does the self arise?",
        "What is the true nature of personal existence?",
    ],
    "purpose_meaning": [
        "What is the meaning of life?",
        "Why do we exist?",
        "What gives life purpose?",
        "Is there an inherent purpose to human existence?",
        "What should humans strive for in life?",
        "What is the ultimate goal of human endeavors?",
        "Does life have intrinsic value?",
        "What makes life worth living?",
        "How do we find meaning in suffering?",
        "What is the relationship between work and purpose?",
        "What is the meaning of success?",
        "How do we discover our life purpose?",
        "What role does love play in life's meaning?",
        "How does purpose relate to happiness?",
        "What is the meaning of suffering?",
        "How do we create meaning in an indifferent universe?",
        "What is the relationship between freedom and purpose?",
        "What gives humans dignity?",
        "How should we prioritize our goals?",
        "What is the ultimate aim of human life?",
    ],
    "ethics_morality": [
        "What makes an action morally right?",
        "How should one live a good life?",
        "What is the foundation of ethics?",
        "Are moral truths universal or relative?",
        "What is the relationship between virtue and happiness?",
        "What is the role of intention in moral judgment?",
        "Can the ends justify the means?",
        "What is moral obligation?",
        "How should we treat others?",
        "What is the golden rule?",
        "What is the relationship between duty and desire?",
        "What makes a person virtuous?",
        "What is moral integrity?",
        "How do we resolve moral dilemmas?",
        "What is the nature of moral responsibility?",
        "What is the relationship between morality and law?",
        "What is ethical relativism?",
        "How does culture affect morality?",
        "What is the basis of human rights?",
        "What is the relationship between compassion and ethics?",
    ],
    "reality_existence": [
        "What is the nature of reality?",
        "Is the physical world the ultimate reality?",
        "What is the relationship between mind and matter?",
        "Does reality exist independently of observation?",
        "What is the relationship between appearance and reality?",
        "Is the world real or a dream?",
        "What is the ultimate substance of reality?",
        "How many levels of reality exist?",
        "What is the relationship between being and becoming?",
        "Is there a fundamental layer of reality?",
        "What is the nature of objective reality?",
        "How does consciousness shape reality?",
        "What is the relationship between truth and reality?",
        "Is reality ultimately one or many?",
        "What is the nature of existence?",
        "How does nothingness relate to being?",
        "What is the relationship between time and reality?",
        "Is reality static or dynamic?",
        "What is the relationship between space and existence?",
        "How do we perceive the nature of reality?",
    ],
    "knowledge_truth": [
        "What is true knowledge?",
        "How can we know what is real?",
        "What is the nature of truth?",
        "What are the limits of human knowledge?",
        "Can we have certain knowledge about anything?",
        "What is the difference between belief and knowledge?",
        "How is knowledge acquired?",
        "What is justified true belief?",
        "Can we trust our senses?",
        "What is skepticism?",
        "What is the difference between opinion and knowledge?",
        "How does language shape knowledge?",
        "What is the relationship between knowledge and power?",
        "What is epistemic humility?",
        "How do we distinguish true from false beliefs?",
        "What is the nature of understanding?",
        "Can there be knowledge without experience?",
        "What is the role of reason in acquiring knowledge?",
        "How does memory contribute to knowledge?",
        "What is the relationship between wisdom and knowledge?",
    ],
    "death_immortality": [
        "What happens after death?",
        "Is death the end of existence?",
        "Does death feel painful?",
        "How should we face mortality?",
        "What is the proper attitude toward death?",
        "Does the soul survive physical death?",
        "What is the meaning of mortality?",
        "How does awareness of death affect life?",
        "What is a good death?",
        "Is immortality desirable?",
        "What is death?",
        "What is the relationship between life and death?",
        "How do different philosophies view death?",
        "What is the fear of death?",
        "Can death be overcome?",
        "What is the nature of dying?",
        "How should we prepare for death?",
        "What is the relationship between death and meaning?",
        "What is eternal life?",
        "How does culture shape attitudes toward death?",
        "What is the metaphysics of death?",
    ],
    "nature_universe": [
        "What is the relationship between humans and nature?",
        "What is the cosmic order of the universe?",
        "Are humans separate from or connected to the cosmos?",
        "Is the universe purposeful or indifferent?",
        "What is humanity's place in the universe?",
        "What is the origin of the universe?",
        "Is the universe finite or infinite?",
        "What is the relationship between nature and nurture?",
        "What is the natural order?",
        "Are natural laws absolute?",
        "What is the relationship between humanity and the environment?",
        "How should we understand natural disasters?",
        "What is the relationship between science and nature?",
        "Is nature inherently violent or peaceful?",
        "What is the role of humans in the ecosystem?",
        "What is the relationship between technology and nature?",
        "How does nature inspire human thought?",
        "What is the beauty of nature?",
        "How do we find meaning in nature?",
        "What is the relationship between natural and moral law?",
        "Is God the creator of universe?",
        "Is Universe always listening to us?",
        "Does manifestations come true if asked to the universe?",
    ],
    "enlightenment_liberation": [
        "How can one achieve enlightenment?",
        "What is liberation from suffering?",
        "What is the ultimate goal of spiritual practice?",
        "What is the path to spiritual awakening?",
        "How does one transcend worldly suffering?",
        "What is nirvana?",
        "What is moksha?",
        "What is satori?",
        "What is the nature of spiritual realization?",
        "How does one become fully awakened?",
        "What is the relationship between enlightenment and compassion?",
        "What blocks spiritual progress?",
        "What is the nature of spiritual liberation?",
        "How does meditation lead to enlightenment?",
        "What is the role of guru in enlightenment?",
        "What is self-enlightenment?",
        "What is the relationship between knowledge and enlightenment?",
        "Can enlightenment be gradual or sudden?",
        "What is the experience of enlightenment?",
        "How does enlightenment affect daily life?",
    ],
    "free_will_fate": [
        "Do humans have free will?",
        "Is everything predetermined by fate?",
        "What is the relationship between free will and determinism?",
        "Are humans responsible for their actions?",
        "Can we shape our own destiny?",
        "What is predestination?",
        "How does karma relate to free will?",
        "What is the nature of choice?",
        "Are our choices truly free?",
        "What is the relationship between freedom and responsibility?",
        "What is the difference between freedom and license?",
        "How does causality affect free will?",
        "What is the relationship between fate and destiny?",
        "Can we break free from our past?",
        "What is the nature of human agency?",
        "How does foreknowledge affect free will?",
        "What is the relationship between free will and morality?",
        "Can free will exist in a deterministic universe?",
        "What is the role of will in human life?",
        "How does determinism affect moral responsibility?",
    ],
    "good_evil": [
        "What is the nature of good and evil?",
        "Why does evil exist in the world?",
        "Is morality based on divine command or reason?",
        "Can good come from evil?",
        "What is the ultimate good for humans?",
        "What is the problem of evil?",
        "Is humanity inherently good or evil?",
        "What is the relationship between good and pleasure?",
        "Can evil be justified?",
        "What is moral relativism?",
        "What is the nature of moral evil?",
        "How do we combat evil in the world?",
        "What is the relationship between good and God?",
        "What is the root of human wickedness?",
        "Can evil be eliminated?",
        "What is the relationship between good and truth?",
        "What is moral courage?",
        "How does one overcome evil?",
        "What is the nature of sin?",
        "What is the relationship between good actions and good character?",
    ],
    "society_justice": [
        "What is the ideal form of government?",
        "What is social justice?",
        "What are the rights and duties of citizens?",
        "Should individuals prioritize society or self?",
        "What is the relationship between law and morality?",
        "What is the best political system?",
        "What is the relationship between freedom and equality?",
        "What is the social contract?",
        "How should power be distributed in society?",
        "What is the relationship between individual and collective?",
        "What is the nature of democracy?",
        "How should we balance individual rights with social goods?",
        "What is the role of education in society?",
        "How does economic inequality affect society?",
        "What is the relationship between religion and politics?",
        "What is the ideal community?",
        "How should we address systemic injustice?",
        "What is the relationship between freedom and authority?",
        "What makes a society just?",
        "How do we create a peaceful society?",
    ],
    "death_meaning": [
        "What is the proper way to face death?",
        "Does death give life meaning?",
        "How should the awareness of death affect how we live?",
        "What is a good death?",
        "Is immortality desirable?",
        "What is mortality?",
        "How do different cultures view death?",
        "What is the relationship between life and death?",
        "What is the fear of death?",
        "How can we accept death?",
        "What is the relationship between death and freedom?",
        "How does death relate to identity?",
        "What is the meaning of mortality for human action?",
        "How should we remember the dead?",
        "What is the relationship between death and love?",
        "How does death shape human values?",
        "What is the wisdom of mortality?",
        "How do we cope with the death of loved ones?",
        "What is the relationship between death and creativity?",
        "What is the spiritual significance of death?",
    ],
    "body_mind": [
        "What is the relationship between body and mind?",
        "Is the mind separate from the brain?",
        "What is consciousness?",
        "What is the nature of subjective experience?",
        "How do thoughts arise?",
        "What is the self?",
        "Is the mind eternal?",
        "What is the relationship between spirit and body?",
        "What is embodiment?",
        "How does the body influence the mind?",
        "What is the nature of mental states?",
        "Can the mind exist without the body?",
        "What is the relationship between emotions and thoughts?",
        "How does perception work?",
        "What is the nature of consciousness?",
        "What is the relationship between brain and mind?",
        "How do we control our minds?",
        "What is the nature of memory?",
        "How does attention shape experience?",
        "What is the relationship between intention and action?",
    ],
    "wisdom_truth": [
        "What is wisdom?",
        "How is wisdom different from knowledge?",
        "What is practical wisdom?",
        "How do we acquire wisdom?",
        "What is the wise person like?",
        "What is philosophical wisdom?",
        "Can wisdom be taught?",
        "What is the relationship between wisdom and virtue?",
        "What is the highest wisdom?",
        "How does wisdom manifest in action?",
        "What is the relationship between wisdom and compassion?",
        "How does wisdom relate to happiness?",
        "What is the difference between cleverness and wisdom?",
        "What is the role of experience in wisdom?",
        "How does wisdom develop over time?",
        "What is the relationship between wisdom and age?",
        "How do we apply wisdom in daily life?",
        "What is the relationship between wisdom and creativity?",
        "Can wisdom be measured?",
        "What is the pursuit of wisdom?",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_repetition_score(text: str, ngram_size: int = 4) -> float:
    tokens = text.lower().split()
    if len(tokens) < ngram_size:
        return 0.0
    ngrams = [tuple(tokens[i:i + ngram_size]) for i in range(len(tokens) - ngram_size + 1)]
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values())
    return repeated / len(ngrams)


def compute_type_token_ratio(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_concept_frequencies(text: str) -> dict:
    lower = text.lower()
    east_count = sum(lower.count(m) for m in EASTERN_MARKERS)
    west_count = sum(lower.count(m) for m in WESTERN_MARKERS)
    total = east_count + west_count
    return {
        "eastern_marker_count": east_count,
        "western_marker_count": west_count,
        "eastern_ratio": east_count / total if total > 0 else 0.0,
        "western_ratio": west_count / total if total > 0 else 0.0,
    }


def compute_perplexity(model, encoding, text: str, device: str, block_size: int = 1024) -> float:
    model.eval()
    try:
        tokens = encoding.encode(text)
        if len(tokens) < 2:
            return float("inf")
        tokens = tokens[:block_size]
        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long).to(device)
        target_ids = torch.tensor([tokens[1:]], dtype=torch.long).to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            loss = outputs.loss.item()
        return math.exp(loss)
    except Exception:
        return float("inf")


def analyze_single_output(text: str) -> dict:
    return {
        "length_chars": len(text),
        "length_words": len(text.split()),
        "repetition_score": round(compute_repetition_score(text), 4),
        "type_token_ratio": round(compute_type_token_ratio(text), 4),
        "concept_frequencies": compute_concept_frequencies(text),
    }


def _generate(model, encoding, prompt, device, max_tokens):
    """Wrapper that passes the full GENERATION_CONFIG to generate()."""
    return generate(
        model,
        encoding,
        prompt,
        max_new_tokens=max_tokens,
        device=device,
        **GENERATION_CONFIG,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_models(
    western_path: str,
    eastern_path: str,
    output_dir: str,
    max_tokens: int = 150,
    temperature: float = GENERATION_CONFIG["temperature"],
    top_p: float = GENERATION_CONFIG["top_p"],
    top_k: int = 0,
    repetition_penalty: float = GENERATION_CONFIG["repetition_penalty"],
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Western model...")
    western_model = load_model(western_path, device)
    western_model.generation_config.pad_token_id = western_model.generation_config.eos_token_id

    print("Loading Eastern model...")
    eastern_model = load_model(eastern_path, device)
    eastern_model.generation_config.pad_token_id = eastern_model.generation_config.eos_token_id

    encoding = get_tokenizer()

    results = {
        "timestamp": datetime.now().isoformat(),
        "western_model": western_path,
        "eastern_model": eastern_path,
        "config": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": GENERATION_CONFIG["no_repeat_ngram_size"],
        },
        "evaluations": [],
    }

    print("\n" + "=" * 60)
    print("Starting Philosophical Bias Evaluation")
    print("=" * 60)

    for category, prompts in PHILOSOPHICAL_PROMPTS.items():
        print(f"\n--- {category.replace('_', ' ').title()} ---")

        for prompt in prompts:
            print(f"\nPrompt: {prompt}")

            western_output = _generate(western_model, encoding, prompt, device, max_tokens)
            eastern_output = _generate(eastern_model, encoding, prompt, device, max_tokens)

            print(f"  Western: {western_output[:100]}...")
            print(f"  Eastern: {eastern_output[:100]}...")

            # Cross-model perplexity — core bias signal
            west_ppl_on_east_text = compute_perplexity(western_model, encoding, eastern_output, device)
            east_ppl_on_west_text = compute_perplexity(eastern_model, encoding, western_output, device)

            kl = compute_kl_report(
                western_model, eastern_model, encoding,
                prompt=prompt,
                west_output=western_output,
                east_output=eastern_output,
                device=device,
            )

            results["evaluations"].append({
                "category": category,
                "prompt": prompt,
                "western_output": western_output,
                "eastern_output": eastern_output,
                "western_metrics": analyze_single_output(western_output),
                "eastern_metrics": analyze_single_output(eastern_output),
                # Cross-perplexity: how surprised is each model by the other's output?
                "cross_perplexity": {
                    "western_model_on_eastern_text": round(west_ppl_on_east_text, 2),
                    "eastern_model_on_western_text": round(east_ppl_on_west_text, 2),
                },
                # KL divergence (%) — 0=identical distributions, 100=max divergence
                "kl_divergence": kl,
            })

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"bias_evaluation_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_bias(results: dict):
    """Summarize bias metrics across categories."""
    print("\n" + "=" * 60)
    print("Bias Analysis Summary")
    print("=" * 60)

    categories: dict = {}
    for ev in results["evaluations"]:
        cat = ev["category"]
        if cat not in categories:
            categories[cat] = {
                "west_rep": [], "east_rep": [],
                "west_ttr": [], "east_ttr": [],
                "west_ppl_on_east": [], "east_ppl_on_west": [],
                "west_east_concept_ratio": [], "east_east_concept_ratio": [],
                "kl_w2e_prompt": [], "kl_e2w_prompt": [], "kl_sym_prompt": [],
            }
        c = categories[cat]
        c["west_rep"].append(ev["western_metrics"]["repetition_score"])
        c["east_rep"].append(ev["eastern_metrics"]["repetition_score"])
        c["west_ttr"].append(ev["western_metrics"]["type_token_ratio"])
        c["east_ttr"].append(ev["eastern_metrics"]["type_token_ratio"])
        cp = ev.get("cross_perplexity", {})
        if cp.get("western_model_on_eastern_text", float("inf")) < float("inf"):
            c["west_ppl_on_east"].append(cp["western_model_on_eastern_text"])
        if cp.get("eastern_model_on_western_text", float("inf")) < float("inf"):
            c["east_ppl_on_west"].append(cp["eastern_model_on_western_text"])
        c["west_east_concept_ratio"].append(
            ev["western_metrics"]["concept_frequencies"]["eastern_ratio"]
        )
        c["east_east_concept_ratio"].append(
            ev["eastern_metrics"]["concept_frequencies"]["eastern_ratio"]
        )
        kl = ev.get("kl_divergence", {})
        for key, bucket in [
            ("west_to_east_on_prompt", "kl_w2e_prompt"),
            ("east_to_west_on_prompt", "kl_e2w_prompt"),
            ("symmetric_on_prompt",    "kl_sym_prompt"),
        ]:
            v = kl.get(key)
            if v is not None and not math.isnan(v):
                c[bucket].append(v)

    def avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else float("nan")

    print(f"\n{'Category':<28} {'W-Rep':>6} {'E-Rep':>6} {'W-TTR':>6} {'E-TTR':>6} "
          f"{'W→E PPL':>9} {'E→W PPL':>9} {'W East%':>8} {'E East%':>8} "
          f"{'KL W→E%':>8} {'KL E→W%':>8} {'KL Sym%':>8}")
    print("-" * 128)

    for cat, c in categories.items():
        label = cat.replace("_", " ").title()[:27]
        print(
            f"{label:<28} "
            f"{avg(c['west_rep']):>6.3f} {avg(c['east_rep']):>6.3f} "
            f"{avg(c['west_ttr']):>6.3f} {avg(c['east_ttr']):>6.3f} "
            f"{avg(c['west_ppl_on_east']):>9.1f} {avg(c['east_ppl_on_west']):>9.1f} "
            f"{avg(c['west_east_concept_ratio']):>8.3f} {avg(c['east_east_concept_ratio']):>8.3f} "
            f"{avg(c['kl_w2e_prompt']):>7.1f}% {avg(c['kl_e2w_prompt']):>7.1f}% {avg(c['kl_sym_prompt']):>7.1f}%"
        )

    print("\nColumn guide:")
    print("  W-Rep / E-Rep     : repetition score (0=none, 1=full loop) — lower is better")
    print("  W-TTR / E-TTR     : type-token ratio (lexical diversity)   — higher is better")
    print("  W→E PPL / E→W PPL : cross-perplexity — higher = more surprised by the other's output")
    print("  W East% / E East% : fraction of philosophical markers that are eastern-tradition")
    print("  KL W→E% / KL E→W% : how different western/eastern token distributions are on the prompt")
    print("  KL Sym%           : symmetric average — single number for overall model divergence")
    print("                      (all KL values: 0%=identical distributions, 100%=max divergence)")


# ---------------------------------------------------------------------------
# Stats analysis
# ---------------------------------------------------------------------------

def run_stats_analysis(results: dict):
    """Run Bhattacharyya stats analysis on evaluation results and print report."""
    evaluations = results.get("evaluations", [])
    if not evaluations:
        return

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS OF MODEL OUTPUT DISTRIBUTIONS")
    print("=" * 70)

    category_results = {}
    for category in set(e.get("category", "unknown") for e in evaluations):
        cat_evals = [e for e in evaluations if e.get("category") == category]
        category_results[category] = analyze_category(category, cat_evals)

    print(f"\n{'Category':<30} {'BC':>8} {'BD':>8}  {'Overlap'}")
    print("-" * 60)
    for cat, res in sorted(category_results.items()):
        print(f"{cat.replace('_', ' ').title():<30} "
              f"{res['bhattacharyya_coefficient']:>8.4f} {res['bhattacharyya_distance']:>8.4f}  "
              f"{res['bhattacharyya_interpretation']}")

    west_all = [e["western_output"] for e in evaluations if "western_output" in e]
    east_all = [e["eastern_output"] for e in evaluations if "eastern_output" in e]
    overall = compute_overlap_metrics(west_all, east_all)

    print("\n" + "=" * 70)
    print("OVERALL METRICS (All Categories Combined)")
    print("=" * 70)
    print(f"\nBhattacharyya Coefficient:    {overall['bhattacharyya_coefficient']:.4f}  ({overall['bhattacharyya_interpretation']})")
    print(f"Bhattacharyya Distance:       {overall['bhattacharyya_distance']:.4f}")
    print(f"\nUnique Western Words: {overall['unique_western_words']}")
    print(f"Unique Eastern Words: {overall['unique_eastern_words']}")
    print(f"Common Words:          {overall['common_words']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate philosophical bias in trained models")
    parser.add_argument("--western-path", type=str, default=WESTERN_MODEL_PATH)
    parser.add_argument("--eastern-path", type=str, default=EASTERN_MODEL_PATH)
    parser.add_argument("--output-dir",   type=str, default=OUTPUT_DIR)
    parser.add_argument("--max-tokens",   type=int, default=150)
    parser.add_argument(
        "--temperature",
        type=float,
        default=GENERATION_CONFIG["temperature"],
        help=f"Generation temperature (default: {GENERATION_CONFIG['temperature']})",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=GENERATION_CONFIG["top_p"],
        help="Nucleus sampling threshold",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling (0 = disabled)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=GENERATION_CONFIG["repetition_penalty"],
        help="Penalty for repeated tokens (>1.0 reduces repetition)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run analysis on most recent existing results file",
    )

    args = parser.parse_args()

    if args.analyze_only:
        output_dir = Path(args.output_dir)
        existing_files = sorted(output_dir.glob("bias_evaluation_*.json"))
        if existing_files:
            with open(existing_files[-1]) as f:
                results = json.load(f)
            analyze_bias(results)
            run_stats_analysis(results)
        else:
            print("No existing evaluation results found.")
    else:
        results = evaluate_models(
            args.western_path,
            args.eastern_path,
            args.output_dir,
            args.max_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            args.repetition_penalty,
        )
        analyze_bias(results)
        run_stats_analysis(results)
