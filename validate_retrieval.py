from sentence_transformers import SentenceTransformer, util
import torch

# --- Configuration ---
RAG_MODEL_ID = "all-MiniLM-L6-v2"
CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 50

# Multi-Topic Competing Corpus (Same as study_agent.py)
thermo_text = """
In classical thermodynamics, entropy is a property of a thermodynamic system that describes the direction or outcome of spontaneous changes within the system. It was introduced by Rudolf Clausius in the mid-19th century to explain the relationship between internal energy available and unavailable for transformations involving heat and work. Entropy suggests that certain processes are irreversible or impossible, even though they do not violate the conservation of energy. The definition of entropy is fundamental to the establishment of the second law of thermodynamics, which states that the entropy of isolated systems cannot decrease over time, as they naturally tend towards a state of thermodynamic equilibrium where entropy is at its maximum. Consequently, entropy is also considered a measure of disorder or randomness within a system. A "high" entropy state signifies more disordered or dispersed energy, while "low" entropy indicates more ordered or concentrated energy.

Rudolf Clausius developed the thermodynamic definition of entropy in the early 1850s, defining it as the quotient of an infinitesimal amount of heat to the instantaneous temperature. He initially referred to it as "transformation-content" (Verwandlungsinhalt) before coining the term "entropy" from a Greek word meaning "transformation". This definition essentially describes how to measure the entropy of an isolated system in thermodynamic equilibrium and proved useful in characterizing the Carnot cycle. Later, Ludwig Boltzmann explained entropy as a measure of the number of possible microscopic configurations (microstates) of a system's individual atoms and molecules that correspond to its macroscopic state (macrostate). He showed that thermodynamic entropy is proportional to the natural logarithm of the number of microstates (Ω), with the proportionality constant being the Boltzmann constant (k), expressed as S = k ln Ω. Boltzmann's statistical interpretation connected the macroscopic observations of nature with a microscopic view.

Entropy is a state function, meaning it is a property that depends only on the current state of the system, independent of how that state was achieved. It is not a conserved quantity; for instance, in an isolated system with non-uniform temperature, heat can flow irreversibly, increasing entropy. The thermodynamic entropy has the dimension of energy divided by temperature, with the SI unit being joules per kelvin (J/K). An irreversible process degrades the performance of a thermodynamic system and results in entropy production, while the entropy generation during a reversible process is zero.
"""

info_theory_text = """
In information theory, entropy quantifies the average amount of information contained in each message received, characterizing our uncertainty about the information source. It measures the uncertainty or randomness of a system. The concept was introduced by Claude Shannon in his 1948 paper "A Mathematical Theory of Communication" and is also referred to as Shannon entropy. Key characteristics and interpretations of information entropy include uncertainty and randomness: entropy is a measure of uncertainty or randomness, with higher entropy indicating more unpredictable or random data. Conversely, lower entropy suggests a more predictable source. When an outcome is certain, the entropy is zero, meaning no new information is delivered. For instance, a source always generating the same character has an entropy of 0.

The definition of entropy only considers the probability of observing a specific event, not the meaning of the events themselves. Mathematically, entropy is defined in the context of a probabilistic model as the expected value of the information content of symbols, or the average amount of information conveyed by an event. The units of entropy depend on the base of the logarithm used in its definition: base 2 yields "bits" or "shannons," base e yields "nats," and base 10 yields "dits," "bans," or "hartleys." A common example involves a fair coin flip, which provides 1 bit of information and has an entropy of 1 bit per flip. In contrast, if the outcome of a coin flip is already known, the entropy is zero.

Information entropy is fundamentally important in digital communications and data compression. It underpins the design and optimization of communication systems and provides the theoretical framework for understanding the limits of data compression by determining the minimum channel capacity required to reliably transmit a source. The term "entropy" was adopted due to the mathematical similarities between Shannon's information theory expressions and those for entropy in statistical thermodynamics, developed by Ludwig Boltzmann and J. Willard Gibbs.
"""

bio_text = """
Homeostasis, in biology, refers to the state of steady internal physical and chemical conditions that are maintained by living systems. This steady state represents the optimal functioning condition for an organism. It involves keeping many variables, such as body temperature and fluid balance, within specific pre-set limits, known as the homeostatic range. Other crucial variables include the pH of extracellular fluid, concentrations of ions like sodium, potassium, and calcium, and blood sugar levels. These must be regulated despite ongoing changes in the environment, diet, or activity levels. Homeostasis is essentially a self-regulating process that allows biological systems to maintain stability while adjusting to conditions that are best for survival. If successful, life continues; if unsuccessful, it can lead to disaster or death. This stability is often described as a dynamic equilibrium, meaning continuous change occurs, yet relatively uniform conditions prevail.

All homeostatic control mechanisms typically involve at least three interdependent components for the variable being regulated: a receptor, a control center, and an effector. The receptor is the sensing component that monitors and responds to changes in the environment, whether external or internal. Examples include thermoreceptors and mechanoreceptors. The control center (or integration center) receives and processes information from the receptor. It sets the maintenance range—the acceptable upper and lower limits—for the particular variable. The brain, spinal cord, and pancreas can act as coordination centers. The effector is the target acted upon to bring about the change back to the normal state. Effectors can be muscles that contract or relax, or glands that secrete. Homeostasis is primarily maintained through negative feedback loops, where a change in a variable triggers a response that counteracts the initial change, bringing the system back towards its set point.
"""

os_text = """
In a multitasking computer system, processes can exist in various states, which serve as a useful abstraction for understanding their behavior, even if not always explicitly recognized by the operating system kernel. The operating system's primary role includes controlling process execution, determining execution interleaving, and allocating resources. The main process states include: Created (or New), Ready (or Waiting), Running, Blocked (or Waiting), and Terminated. When a process is initially created, it enters the Created state, awaiting admission to the "ready" state. A process in the Ready state has been loaded into main memory and is awaiting its turn to be executed by the CPU. There can be multiple ready processes at any given time, queuing for the CPU.

A process transitions to the Running state when selected for execution by the CPU. The process's instructions are executed by one of the system's CPUs or cores, with at most one running process per CPU or core. Processes can run in Kernel Mode (Supervisor State), offering full access to all hardware and system resources, or User Mode (Problem State), where regular applications run with restricted access. A process enters the Blocked state when it cannot proceed until a specific event occurs, such as the completion of an I/O operation, or receiving a signal from another process. It will remain blocked until the event happens, after which it becomes ready again. Finally, a process enters the Terminated state when it finishes execution or is forcefully ended due to an error or lack of resources. The operating system maintains information about active processes in data structures known as process control blocks.
"""

stats_text = """
The Principle of Maximum Entropy states that the probability distribution which best represents the current state of knowledge about a system is the one with the largest information entropy, given precisely stated prior data (also known as testable information). In essence, it suggests that when determining a probability distribution based on partial information, one should choose the distribution that maximizes entropy among all those satisfying the known constraints. This approach aims to be "non-committal" with respect to missing information, assuming nothing more than what is already known, thereby making the least biased estimate possible.

Information entropy, central to the principle, is a measure of uncertainty or randomness. A higher entropy indicates greater unpredictability or information content, meaning less certainty about an outcome and more information required to describe the system's state. The principle seeks to maximize this uncertainty given the constraints, ensuring that no additional assumptions are made beyond the available data. The maximum entropy distribution is considered the "least informative default" or the most uniform possible distribution that still satisfies the given constraints. If no testable information is provided, the maximum entropy discrete probability distribution is the uniform distribution. The principle was first expounded by E. T. Jaynes in 1957, emphasizing a natural correspondence between statistical mechanics and information theory.
"""

SAMPLE_TEXT = thermo_text + "\n\n" + info_theory_text + "\n\n" + bio_text + "\n\n" + os_text + "\n\n" + stats_text

# --- RAG System (with Diagnostic Mode) ---
class RAGSystem:
    def __init__(self):
        print(f"Loading RAG model: {RAG_MODEL_ID}...")
        self.model = SentenceTransformer(RAG_MODEL_ID)
        self.chunks = []  # Chunks WITH tags for LLM context
        self.embeddings = None
        self.chunk_size = CHUNK_SIZE_WORDS
        self.chunk_overlap = CHUNK_OVERLAP_WORDS
        
    def chunk_text_sentence_aware(self, text, domain_tag, embed_tags=True):
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        embed_texts = []  # For embedding (may or may not have tag)
        current_chunk_sentences = []
        current_word_count = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i].strip()
            if not sentence:
                i += 1
                continue
            sent_word_count = len(sentence.split())
            
            if sent_word_count > self.chunk_size:
                words = sentence.split()
                if current_chunk_sentences:
                    content = " ".join(current_chunk_sentences)
                    chunks.append(f"[{domain_tag}] {content}")
                    embed_texts.append(f"[{domain_tag}] {content}" if embed_tags else content)
                    current_chunk_sentences = []
                    current_word_count = 0
                for j in range(0, sent_word_count, self.chunk_size):
                    sub_words = words[j : j + self.chunk_size]
                    sub_text = " ".join(sub_words)
                    chunks.append(f"[{domain_tag}] {sub_text}")
                    embed_texts.append(f"[{domain_tag}] {sub_text}" if embed_tags else sub_text)
                i += 1
                continue
            
            if current_word_count + sent_word_count <= self.chunk_size:
                current_chunk_sentences.append(sentence)
                current_word_count += sent_word_count
                i += 1
            else:
                content = " ".join(current_chunk_sentences)
                chunks.append(f"[{domain_tag}] {content}")
                embed_texts.append(f"[{domain_tag}] {content}" if embed_tags else content)
                overlap_buffer = []
                overlap_count = 0
                for s in reversed(current_chunk_sentences):
                    s_len = len(s.split())
                    if overlap_count + s_len <= self.chunk_overlap:
                        overlap_buffer.insert(0, s)
                        overlap_count += s_len
                    else:
                        break
                current_chunk_sentences = overlap_buffer
                current_word_count = overlap_count
        
        if current_chunk_sentences:
            content = " ".join(current_chunk_sentences)
            chunks.append(f"[{domain_tag}] {content}")
            embed_texts.append(f"[{domain_tag}] {content}" if embed_tags else content)
        return chunks, embed_texts

    def index_documents(self, documents, embed_tags=True):
        self.chunks = []
        embed_texts = []
        for domain, text in documents.items():
            domain_chunks, domain_embeds = self.chunk_text_sentence_aware(text, domain, embed_tags)
            self.chunks.extend(domain_chunks)
            embed_texts.extend(domain_embeds)
        self.embeddings = self.model.encode(embed_texts, convert_to_tensor=True)
        print(f"Indexed {len(self.chunks)} chunks (embed_tags={embed_tags}).")
        
    def retrieve(self, query, top_k=3):
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.embeddings)[0]
        k = min(top_k, len(self.chunks))
        top_results = torch.topk(scores, k=k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(top_results.values, top_results.indices)):
            chunk_domain = self.chunks[idx].split("]")[0].strip("[")
            results.append({
                "rank": rank + 1,
                "score": float(score),
                "domain": chunk_domain,
                "content": self.chunks[idx][:150]
            })
        return results

def run_perturbation_tests(rag, test_cases, report_file):
    """
    Run multiple query variations per domain.
    test_cases: list of { "domain": "X", "queries": ["q1", "q2", "q3"] }
    """
    CONFIDENCE_MARGIN = 0.05  # Minimum Top1-Top2 separation
    
    total_tests = 0
    total_passed = 0
    total_ambiguous = 0
    
    with open(report_file, "a", encoding="utf-8") as f:
        f.write("\n\n=== PERTURBATION ROBUSTNESS TEST ===\n")
        f.write(f"Confidence Margin: {CONFIDENCE_MARGIN}\n")
    print("\n=== PERTURBATION ROBUSTNESS TEST ===")
    print(f"Confidence Margin: {CONFIDENCE_MARGIN}")
    
    for case in test_cases:
        domain = case["domain"]
        queries = case["queries"]
        case_passed = True
        
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- Domain: {domain} ---\n")
        print(f"\n--- Domain: {domain} ---")
        
        for q in queries:
            total_tests += 1
            results = rag.retrieve(q, top_k=3)
            top1 = results[0]
            top2 = results[1] if len(results) > 1 else {"score": 0.0}
            sep = top1["score"] - top2["score"]
            domain_correct = top1["domain"] == domain
            
            # Determine status
            if not domain_correct:
                status = "❌ FAIL"
                case_passed = False
            elif sep < CONFIDENCE_MARGIN:
                status = "⚠️ AMBIGUOUS"
                total_ambiguous += 1
                total_passed += 1  # Still counts as pass for domain match
            else:
                status = "✅ PASS"
                total_passed += 1
                
            line = f"  Query: '{q[:50]}...' | Top-1: [{top1['domain']}] | Sep: {sep:.4f} | {status}"
            print(line)
            with open(report_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        
        case_status = "ALL PASSED" if case_passed else "SOME FAILED"
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(f"Domain Status: {case_status}\n")
    
    summary = f"\nPERTURBATION SUMMARY: {total_passed}/{total_tests} domain-correct ({total_ambiguous} ambiguous)."
    print(summary)
    with open(report_file, "a", encoding="utf-8") as f:
        f.write(summary + "\n")

def run_diagnostic_mode(documents, test_cases, report_file):
    """
    Re-index WITHOUT domain tags in embeddings. Test if accuracy collapses.
    """
    with open(report_file, "a", encoding="utf-8") as f:
        f.write("\n\n=== DIAGNOSTIC: TAG-STRIPPED EMBEDDINGS ===\n")
        f.write("Testing with [DOMAIN] tags REMOVED from embeddings.\n")
    print("\n=== DIAGNOSTIC: TAG-STRIPPED EMBEDDINGS ===")
    
    rag = RAGSystem()
    rag.index_documents(documents, embed_tags=False)
    
    total = 0
    passed = 0
    for case in test_cases:
        domain = case["domain"]
        q = case["queries"][0]  # Test with first query only
        total += 1
        results = rag.retrieve(q, top_k=3)
        top1 = results[0]
        top2 = results[1] if len(results) > 1 else {"score": 0.0}
        sep = top1["score"] - top2["score"]
        correct = top1["domain"] == domain
        if correct:
            passed += 1
        status = "✅ PASS" if correct else "❌ FAIL"
        line = f"  [{domain}] '{q[:40]}...' -> [{top1['domain']}] | Sep: {sep:.4f} | {status}"
        print(line)
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    
    summary = f"\nDIAGNOSTIC SUMMARY: {passed}/{total} correct (without tags)."
    print(summary)
    with open(report_file, "a", encoding="utf-8") as f:
        f.write(summary + "\n")
        if passed < total:
            f.write("CONCLUSION: Domain tags ARE critical for accuracy.\n")
        else:
            f.write("CONCLUSION: Content embeddings are strong on their own.\n")

def main():
    documents = {
        "THERMODYNAMICS": thermo_text,
        "INFORMATION_THEORY": info_theory_text,
        "BIOLOGY": bio_text,
        "OPERATING_SYSTEMS": os_text,
        "STATISTICS": stats_text
    }
    
    # Define perturbation test cases (3 paraphrases per domain)
    test_cases = [
        {
            "domain": "THERMODYNAMICS",
            "queries": [
                "Thermodynamic entropy heat and disorder",
                "What is entropy in classical thermodynamics?",
                "Clausius Boltzmann entropy definition heat"
            ]
        },
        {
            "domain": "INFORMATION_THEORY",
            "queries": [
                "Shannon entropy bits and information compression",
                "Information theory entropy definition Shannon",
                "What is entropy in data compression?"
            ]
        },
        {
            "domain": "BIOLOGY",
            "queries": [
                "Biological homeostasis and living systems energy",
                "What is homeostasis in biology?",
                "Body temperature regulation homeostatic range"
            ]
        },
        {
            "domain": "OPERATING_SYSTEMS",
            "queries": [
                "Operating system process state and context switch",
                "Process states in multitasking OS",
                "What is a context switch in operating systems?"
            ]
        },
        {
            "domain": "STATISTICS",
            "queries": [
                "Maximum entropy principle probability distribution",
                "Jaynes maximum entropy statistics",
                "What is the principle of maximum entropy?"
            ]
        }
    ]
    
    report_file = "rag_validation_report.txt"
    
    # Initialize report
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== RAG VALIDATION REPORT (STABILIZATION) ===\n")
        f.write("Testing: Perturbation Robustness + Tag Reliance Diagnostic\n")
    
    print("=== RAG VALIDATION REPORT (STABILIZATION) ===")
    
    # Phase 1: Normal retrieval with tagged embeddings
    rag = RAGSystem()
    rag.index_documents(documents, embed_tags=True)
    run_perturbation_tests(rag, test_cases, report_file)
    
    # Phase 2: Diagnostic mode (tag-stripped embeddings)
    run_diagnostic_mode(documents, test_cases, report_file)

if __name__ == "__main__":
    main()
