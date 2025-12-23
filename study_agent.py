import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
from sentence_transformers import SentenceTransformer, util
import numpy as np

# --- Configuration ---
# --- Configuration ---
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
RAG_MODEL_ID = "all-MiniLM-L6-v2"  # Embedding model
CHUNK_SIZE_WORDS = 200             # Reduced chunk size for finer granularity
CHUNK_OVERLAP_WORDS = 50           # 25% overlap
SIMILARITY_METRIC = "Cosine Similarity"

# Multi-Topic Competing Corpus
# Domains: Thermodynamics, Information Theory, Biology, Operating Systems, Statistics
# Common terms: "entropy", "information", "state", "system"

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

# --- RAG System ---
# --- RAG System ---
class RAGSystem:
    def __init__(self):
        print(f"Loading RAG model: {RAG_MODEL_ID}...")
        self.model = SentenceTransformer(RAG_MODEL_ID)
        self.chunks = []
        self.embeddings = None
        self.chunk_size = CHUNK_SIZE_WORDS
        self.chunk_overlap = CHUNK_OVERLAP_WORDS
        
    def chunk_text_sentence_aware(self, text, domain_tag):
        """
        Splits text into chunks respecting sentence boundaries.
        Prepends [DOMAIN_TAG] to each chunk.
        Hard splits sentences > CHUNK_SIZE_WORDS.
        
        INVARIANTS (fail-fast if violated):
        - Progress: i must advance at least every 2 iterations
        - Bounded: chunks <= sentences * 2 (generous)
        - Time: O(n) where n = number of sentences
        """
        import re
        
        # Split by sentence boundaries (keeping punctuation)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        num_sentences = len(sentences)
        
        # Invariant bounds
        MAX_ITERATIONS = num_sentences * 3  # O(n) guarantee
        MAX_CHUNKS = num_sentences * 2  # Bounded output
        
        chunks = []
        current_chunk_sentences = []
        current_word_count = 0
        
        # Progress tracking for invariant checks
        iteration_count = 0
        last_i = -1
        stall_count = 0
        
        i = 0
        while i < num_sentences:
            iteration_count += 1
            
            # INVARIANT 1: O(n) time - fail fast if exceeded
            if iteration_count > MAX_ITERATIONS:
                raise RuntimeError(f"[CHUNKING] INVARIANT VIOLATED: Exceeded {MAX_ITERATIONS} iterations at sentence {i}/{num_sentences}")
            
            # INVARIANT 2: Progress check - i must advance
            if i == last_i:
                stall_count += 1
                if stall_count > 1:
                    raise RuntimeError(f"[CHUNKING] INVARIANT VIOLATED: Stalled at sentence {i} for {stall_count} iterations")
            else:
                stall_count = 0
                last_i = i
            
            sentence = sentences[i].strip()
            if not sentence:
                i += 1
                continue
                
            sent_word_count = len(sentence.split())
            
            # Giant sentence handling (sentence alone > chunk_size)
            if sent_word_count > self.chunk_size:
                # Force split and ALWAYS advance
                words = sentence.split()
                if current_chunk_sentences:
                    chunk_content = " ".join(current_chunk_sentences)
                    chunks.append(f"[{domain_tag}] {chunk_content}")
                    current_chunk_sentences = []
                    current_word_count = 0
                
                for j in range(0, sent_word_count, self.chunk_size):
                    sub_words = words[j : j + self.chunk_size]
                    sub_text = " ".join(sub_words)
                    chunks.append(f"[{domain_tag}] {sub_text}")
                
                i += 1  # GUARANTEED PROGRESS
                continue
            
            # Normal handling: sentence fits within chunk_size
            if current_word_count + sent_word_count <= self.chunk_size:
                current_chunk_sentences.append(sentence)
                current_word_count += sent_word_count
                i += 1  # GUARANTEED PROGRESS
            else:
                # Chunk is full, emit it
                if current_chunk_sentences:
                    chunk_content = " ".join(current_chunk_sentences)
                    chunks.append(f"[{domain_tag}] {chunk_content}")
                
                # INVARIANT 3: Bounded chunks
                if len(chunks) > MAX_CHUNKS:
                    raise RuntimeError(f"[CHUNKING] INVARIANT VIOLATED: Exceeded {MAX_CHUNKS} chunks")
                
                # Sliding window for overlap
                overlap_buffer = []
                overlap_count = 0
                for s in reversed(current_chunk_sentences):
                    s_len = len(s.split())
                    if overlap_count + s_len <= self.chunk_overlap:
                        overlap_buffer.insert(0, s)
                        overlap_count += s_len
                    else:
                        break
                
                # CRITICAL: Ensure sentence will fit after overlap
                if overlap_count + sent_word_count > self.chunk_size:
                    # Clear overlap to guarantee progress on next iteration
                    current_chunk_sentences = []
                    current_word_count = 0
                else:
                    current_chunk_sentences = overlap_buffer
                    current_word_count = overlap_count
                # Note: i not incremented - sentence will be added on next iteration
        
        # Emit remaining
        if current_chunk_sentences:
            chunk_content = " ".join(current_chunk_sentences)
            chunks.append(f"[{domain_tag}] {chunk_content}")
            
        return chunks

    def index_documents(self, documents):
        """
        Index multiple documents with domain tags.
        documents: dict { "DOMAIN": "text..." }
        """
        self.chunks = []
        for domain, text in documents.items():
            print(f"Indexing domain: {domain}...")
            domain_chunks = self.chunk_text_sentence_aware(text, domain)
            self.chunks.extend(domain_chunks)
            print(f"  Created {len(domain_chunks)} chunks.")
            
        print(f"Total chunks created: {len(self.chunks)}")
        print("Creating embeddings...")
        self.embeddings = self.model.encode(self.chunks, convert_to_tensor=True)
    
    def add_document(self, domain: str, text: str) -> int:
        """
        Incrementally add a single document without re-indexing existing content.
        
        Args:
            domain: Domain tag for the document (e.g., "PHYSICS", "BIOLOGY")
            text: Full text content of the document
            
        Returns:
            Number of new chunks created
            
        Note: This method ONLY embeds the new chunks, not existing ones.
        """
        import torch
        import time
        
        # Defensive constants (not domain-specific)
        MAX_CHUNK_WORDS = 500  # Skip chunks longer than this
        EMBEDDING_BATCH_SIZE = 32  # Embed in batches for progress visibility
        
        start_time = time.time()
        existing_chunk_count = len(self.chunks)
        print(f"[ADD_DOCUMENT] === Starting for {domain} ===", flush=True)
        print(f"[ADD_DOCUMENT] Existing chunks: {existing_chunk_count}", flush=True)
        print(f"[ADD_DOCUMENT] Input text length: {len(text)} chars, {len(text.split())} words", flush=True)
        
        # Remove any existing chunks for this domain (allows re-upload/update)
        self.chunks = [c for c in self.chunks if not c.startswith(f"[{domain}]")]
        removed_count = existing_chunk_count - len(self.chunks)
        if removed_count > 0:
            print(f"[ADD_DOCUMENT] Removed {removed_count} existing chunks for {domain}", flush=True)
            # Rebuild embeddings without removed chunks
            if self.chunks:
                print(f"[ADD_DOCUMENT] Rebuilding embeddings for {len(self.chunks)} remaining chunks...", flush=True)
                rebuild_start = time.time()
                self.embeddings = self.model.encode(self.chunks, convert_to_tensor=True)
                print(f"[ADD_DOCUMENT] Rebuild complete in {time.time() - rebuild_start:.1f}s", flush=True)
            else:
                self.embeddings = None
        
        # Chunk the new document
        print(f"[ADD_DOCUMENT] Chunking {domain}...", flush=True)
        chunk_start = time.time()
        raw_chunks = self.chunk_text_sentence_aware(text, domain)
        print(f"[ADD_DOCUMENT] Chunking complete in {time.time() - chunk_start:.1f}s, raw chunks: {len(raw_chunks)}", flush=True)
        
        # Defensive validation: filter invalid chunks
        valid_chunks = []
        skipped_empty = 0
        skipped_long = 0
        
        for chunk in raw_chunks:
            # Skip empty chunks
            if not chunk or not chunk.strip():
                skipped_empty += 1
                continue
            
            # Skip excessively long chunks (pathological input)
            word_count = len(chunk.split())
            if word_count > MAX_CHUNK_WORDS:
                print(f"[ADD_DOCUMENT] WARNING: Skipping chunk with {word_count} words (max {MAX_CHUNK_WORDS})", flush=True)
                skipped_long += 1
                continue
            
            valid_chunks.append(chunk)
        
        if skipped_empty > 0 or skipped_long > 0:
            print(f"[ADD_DOCUMENT] Filtered: {skipped_empty} empty, {skipped_long} too long", flush=True)
        
        new_chunk_count = len(valid_chunks)
        print(f"[ADD_DOCUMENT] Valid chunks to embed: {new_chunk_count}", flush=True)
        
        if new_chunk_count == 0:
            print(f"[ADD_DOCUMENT] WARNING: No valid chunks created for {domain}", flush=True)
            return 0
        
        # Embed in batches with progress logging
        print(f"[ADD_DOCUMENT] Embedding {new_chunk_count} chunks in batches of {EMBEDDING_BATCH_SIZE}...", flush=True)
        embed_start = time.time()
        
        all_embeddings = []
        for i in range(0, new_chunk_count, EMBEDDING_BATCH_SIZE):
            batch = valid_chunks[i:i + EMBEDDING_BATCH_SIZE]
            batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
            total_batches = (new_chunk_count + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
            
            print(f"[ADD_DOCUMENT] Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...", flush=True)
            batch_start = time.time()
            
            batch_embeddings = self.model.encode(batch, convert_to_tensor=True)
            all_embeddings.append(batch_embeddings)
            
            print(f"[ADD_DOCUMENT] Batch {batch_num} complete in {time.time() - batch_start:.1f}s", flush=True)
        
        # Concatenate all batch embeddings
        if len(all_embeddings) == 1:
            new_embeddings = all_embeddings[0]
        else:
            new_embeddings = torch.cat(all_embeddings, dim=0)
        
        print(f"[ADD_DOCUMENT] All embeddings complete in {time.time() - embed_start:.1f}s", flush=True)
        
        # Append to existing
        self.chunks.extend(valid_chunks)
        
        if self.embeddings is not None and len(self.embeddings) > 0:
            self.embeddings = torch.cat([self.embeddings, new_embeddings], dim=0)
        else:
            self.embeddings = new_embeddings
        
        total_time = time.time() - start_time
        print(f"[ADD_DOCUMENT] === Complete: {new_chunk_count} chunks, {len(self.chunks)} total, {total_time:.1f}s ===", flush=True)
        return new_chunk_count
        
    def retrieve_with_confidence(self, query, top_k=2, confidence_threshold=0.05):
        """
        Retrieve chunks with confidence assessment.
        FIX 1: Conditional overlap detection for ambiguous vocabulary.
        FIX 2: Contextual threshold interpretation (not global increase).
        FIX 3: Query Specificity Gate (generic, no hardcoded phrases).
        """
        # High-overlap vocabulary terms that span multiple domains
        OVERLAP_TERMS = {"entropy", "uncertainty", "randomness", "state", "system", "information", "measure"}
        
        # --- FIX 3: Query Specificity Gate (Measurable Signals) ---
        # Standard English stopwords (language-level, not domain-specific)
        STOPWORDS = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "also",
            "and", "but", "if", "or", "because", "until", "while", "although",
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
            "himself", "she", "her", "hers", "herself", "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves", "what", "which",
            "who", "whom", "this", "that", "these", "those", "am", "about"
        }
        
        # Generic words (verbs and nouns without domain specificity)
        GENERIC_WORDS = {
            # Generic verbs
            "describe", "explain", "tell", "show", "give", "make", "get", "go",
            "know", "think", "take", "see", "come", "want", "look", "use", "find",
            "work", "mean", "try", "ask", "put", "call", "keep", "let", "begin",
            "seem", "help", "talk", "turn", "start", "happen", "provide", "discuss",
            # Generic nouns (no domain specificity)
            "process", "concept", "topic", "thing", "stuff", "way", "method",
            "idea", "point", "part", "kind", "type", "form", "case", "example",
            "reason", "result", "effect", "cause", "problem", "issue", "question",
            "answer", "solution", "step", "stage", "level", "aspect", "factor",
            "element", "feature", "term", "word", "name", "meaning", "definition",
            "science", "subject", "matter", "area", "field", "context", "situation"
        }
        
        query_lower = query.lower()
        query_words = query_lower.split()
        query_word_count = len(query_words)
        
        # --- Measurable Signal 1: Stopword Ratio ---
        stopword_count = sum(1 for w in query_words if w in STOPWORDS)
        stopword_ratio = stopword_count / query_word_count if query_word_count > 0 else 0
        
        # --- Measurable Signal 2: Content Word Analysis ---
        content_words = [w for w in query_words if w not in STOPWORDS and w not in GENERIC_WORDS]
        content_word_count = len(content_words)
        
        # --- Measurable Signal 3: Technical Term Indicators ---
        # Words with capital letters (proper nouns, acronyms) in original query
        original_words = query.split()
        has_proper_noun = any(w[0].isupper() for w in original_words if len(w) > 0)
        
        # Long content words (>6 chars) often indicate technical terms
        long_content_words = [w for w in content_words if len(w) > 6]
        has_long_content = len(long_content_words) > 0
        
        # --- Query Specificity Score (0.0 = vague, 1.0 = specific) ---
        specificity_score = 0.0
        
        # Factor 1: Content word presence (max 0.4)
        if content_word_count >= 2:
            specificity_score += 0.4
        elif content_word_count == 1:
            specificity_score += 0.2
        
        # Factor 2: Low stopword ratio (max 0.3)
        if stopword_ratio < 0.5:
            specificity_score += 0.3
        elif stopword_ratio < 0.75:
            specificity_score += 0.15
        
        # Factor 3: Technical indicators (max 0.3)
        if has_proper_noun:
            specificity_score += 0.15
        if has_long_content:
            specificity_score += 0.15
        
        # Query is LOW specificity if score < 0.4 AND query is short (≤4 words)
        is_low_specificity = (specificity_score < 0.4 and query_word_count <= 4)
        
        if is_low_specificity:
            print(f"\n[Retrieval] Query: '{query}'")
            print(f"[Retrieval] SPECIFICITY GATE: Query lacks semantic specificity")
            print(f"[Retrieval]   - Specificity Score: {specificity_score:.2f} (threshold: 0.4)")
            print(f"[Retrieval]   - Content Words: {content_word_count}, Stopword Ratio: {stopword_ratio:.2f}")
            print(f"[Retrieval] Forcing LOW confidence")
            # Still perform retrieval for logging but force LOW confidence
            query_emb = self.model.encode(query, convert_to_tensor=True)
            scores = util.cos_sim(query_emb, self.embeddings)[0]
            k = min(top_k, len(self.chunks))
            top_results = torch.topk(scores, k=k)
            results = []
            for score, idx in zip(top_results.values, top_results.indices):
                results.append({"chunk": self.chunks[idx], "score": float(score)})
            return [r["chunk"] for r in results[:2]], "LOW"
        
        print(f"\n[Retrieval] Query: '{query}'")
        print(f"[Retrieval] Algo: {SIMILARITY_METRIC} with {RAG_MODEL_ID}")
        
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.embeddings)[0]
        
        k = min(top_k, len(self.chunks))
        top_results = torch.topk(scores, k=k)
        
        results = []
        print(f"[Retrieval] Top {k} Results:")
        for score, idx in zip(top_results.values, top_results.indices):
            preview = self.chunks[idx][:100].replace('\n', ' ')
            print(f"  - Rank {len(results)+1}: Score {score:.4f} | Content: {preview}...")
            results.append({"chunk": self.chunks[idx], "score": float(score)})
        
        # Compute confidence
        if len(results) >= 2:
            sep = results[0]["score"] - results[1]["score"]
        else:
            sep = 1.0  # Single result is considered high confidence
        
        # --- FIX 1: Check for overlap vocabulary in query ---
        query_words = set(query_lower.split())
        has_overlap_term = bool(query_words & OVERLAP_TERMS)
        
        # --- FIX 2: Contextual threshold interpretation ---
        threshold = confidence_threshold  # Default 0.05
        if has_overlap_term and query_word_count <= 5:
            threshold = 0.15  # Stricter for short ambiguous queries
        elif has_overlap_term:
            threshold = 0.10  # Moderately stricter for overlap terms
        
        confidence = "HIGH" if sep >= threshold else "LOW"
        print(f"[Retrieval] Score Separation: {sep:.4f} -> Confidence: {confidence}")
        
        # Return Top-1 for HIGH, Top-2 for LOW
        if confidence == "HIGH":
            return [results[0]["chunk"]], confidence
        else:
            return [r["chunk"] for r in results[:2]], confidence


def load_model():
    print("Loading model in 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda",
    )
    model.eval()
    return tokenizer, model

def generate_questions(tokenizer, model, text):
    print("\n=== Pass 1: The Examiner (Question Generation) ===")
    
    system_prompt = """You are a strict Examiner. You ONLY write questions. Never explain, summarize, or restate.

RULES:
1. ONE CONCEPT per question. Target exactly ONE definition, property, or relationship.
2. Each question must be answerable in ONE sentence (8-20 words).
3. FORBIDDEN words in questions: "and", "while", "showing that", "which also".
4. FORBIDDEN question types:
   - Single-word answers (names, symbols, constants)
   - Compound or multi-clause questions
   - Questions that restate the source text verbatim
5. ALLOWED question types:
   - "What is the definition of X?"
   - "What property does X have?"
   - "What happens when X?"
   - "Why is X considered Y?"
"""

    user_prompt = f"""=== SOURCE TEXT BEGIN ===
{text}
=== SOURCE TEXT END ===

Create UP TO 3 exam questions. Fewer is better if text is short.

Output format:
1. [Question]
2. [Question]
..."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt_str, return_tensors="pt", truncation=True).to(model.device)
    
    # Unset generation flags for deterministic output
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=256,
            do_sample=False,
        )
    
    response = outputs[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def generate_solution(tokenizer, model, text, question, confidence="HIGH"):
    print(f"Generating solution for: {question[:50]}... (Confidence: {confidence})")
    
    if confidence == "HIGH":
        system_prompt = """You are a strict academic tutor. Your role is to explain concepts clearly and concisely.

RULES:
1. Answer using ONLY the provided source text.
2. Structure your response with:
   - Definition (1-2 sentences)
   - Formula (if present in source, use exact notation)
   - Simple Example (if present in source)
   - Intuition (1-2 sentences explaining the concept)
3. Maximum 6 sentences total.
4. Do NOT invent formulas or examples not in the source.
5. Do NOT repeat phrases or sentences.
6. If the source lacks required information, output exactly: "The provided text does not contain this information."

OUTPUT FORMAT:
**Answer:** [Your structured explanation]
**Quote:** "[Exact supporting quote from source]" """
    else:
        # LOW confidence: stricter mode
        system_prompt = """You are a strict academic tutor in LOW CONFIDENCE mode.

RULES:
1. Answer using ONLY the provided source text.
2. Do NOT synthesize across different [DOMAIN] sections.
3. Keep answer concise (max 4 sentences).
4. Do NOT invent information not in the source.
5. Do NOT repeat phrases.
6. If unsure, output exactly: "The provided text does not contain sufficient information to answer this question."

OUTPUT FORMAT:
**Answer:** [Brief answer]
**Quote:** "[Exact quote from source]" """

    user_prompt = f"""=== SOURCE TEXT BEGIN ===
{text}
=== SOURCE TEXT END ===

Question: {question}

Provide a clear, structured explanation based ONLY on the source text above."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt_str, return_tensors="pt", truncation=True).to(model.device)
    
    # Unset generation flags
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=300,  # Slightly more for structured output
            do_sample=False,
            repetition_penalty=1.2,  # Prevent repetition/degeneration
        )
    
    response = outputs[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def parse_questions(questions_text):
    questions = []
    for line in questions_text.split('\n'):
        line = line.strip()
        # Look for lines starting with "1. ", "2. ", etc.
        if len(line) > 3 and line[0].isdigit() and line[1] == '.':
            questions.append(line[3:].strip())
    return questions

def extract_quote(solution_text):
    """Extracts the quote from the solution text for de-duplication."""
    if "**Quote:**" in solution_text:
        return solution_text.split("**Quote:**")[1].strip().strip('"')
    return None

def main():
    try:
        # Phase 4: Init RAG
        rag = RAGSystem()
        documents = {
            "THERMODYNAMICS": thermo_text,
            "INFORMATION_THEORY": info_theory_text,
            "BIOLOGY": bio_text,
            "OPERATING_SYSTEMS": os_text,
            "STATISTICS": stats_text
        }
        rag.index_documents(documents)
        
        # Define Topic/Query (Simulating user intent)
        # Using Biology query for HIGH confidence (low overlap with other domains)
        TOPIC_QUERY = "Homeostasis body temperature regulation biology"
        
        # Retrieve context with confidence assessment
        retrieved_chunks, confidence = rag.retrieve_with_confidence(TOPIC_QUERY, top_k=2)
        context_text = "\n\n".join(retrieved_chunks)
        
        print("\n=== RAG Context Constructed ===")
        print(f"Query: {TOPIC_QUERY}")
        print(f"Confidence: {confidence}")
        print(f"Chunks Retrieved: {len(retrieved_chunks)}")
        print(f"Context Length: {len(context_text)} chars")
        print("Context Preview:")
        print(context_text[:200] + "...")
        
        # EARLY ABORT: If confidence is LOW, skip Q&A entirely
        if confidence == "LOW":
            print("\n⚠️ EARLY ABORT: Retrieval confidence is LOW (overlapping domain ambiguity).")
            print("Skipping question generation and answering to avoid cross-domain confusion.")
            final_output = "# Study Agent Report\n\n"
            final_output += "**Retrieval Confidence:** LOW\n\n"
            final_output += "> ⚠️ **Aborted**: The query matched multiple domains with insufficient separation.\n"
            final_output += "> Please refine your query to target a specific domain.\n"
            with open("study_guide.md", "w", encoding="utf-8") as f:
                f.write(final_output)
            print("\n=== Early Abort Report Saved ===")
            return
        
        tokenizer, model = load_model()
        
        # Pass 1: Generate Questions (using RAG context)
        questions_text = generate_questions(tokenizer, model, context_text)
        print("\n=== Generated Questions ===")
        print(questions_text)
        
        questions = parse_questions(questions_text)
        
        # INJECT HALLUCINATION TEST
        print("\n[TEST] Injecting invalid question to test anti-hallucination...")
        questions.append("What is the capital of Mars?")
        
        final_output = "# Study Agent Report\n\n"
        final_output += f"**Retrieval Confidence:** {confidence}\n\n"
        seen_quotes = set()
        
        MIN_QUOTE_WORDS = 8  # Minimum words in answer quote for usefulness
        
        # Pass 2: Generate Solutions (using SAME RAG context, with confidence)
        print("\n=== Pass 2: The Tutor (Solution Generation) ===")
        for i, q in enumerate(questions):
            print(f"\nProcessing Q{i+1}: {q}")
            solution = generate_solution(tokenizer, model, context_text, q, confidence)
            
            # Validation check 1: Unanswerable
            if "The provided text does not contain this information" in solution:
                print(f"❌ REJECTED Q{i+1}: Unanswerable (Anti-Hallucination Triggered)")
                with open("rejections.log", "a", encoding="utf-8") as f:
                    f.write(f"REJECTED (Unanswerable): {q}\n")
                continue
            
            # Validation check 2: Quote length (usefulness filter)
            quote = extract_quote(solution)
            if quote:
                quote_words = len(quote.split())
                if quote_words < MIN_QUOTE_WORDS:
                    print(f"❌ REJECTED Q{i+1}: Quote too short ({quote_words} words < {MIN_QUOTE_WORDS})")
                    with open("rejections.log", "a", encoding="utf-8") as f:
                        f.write(f"REJECTED (Short Quote): {q} | Quote: {quote}\n")
                    continue
                
                # Validation check 3: De-duplication
                norm_quote = quote.lower().strip()[:50]
                if norm_quote in seen_quotes:
                    print(f"❌ REJECTED Q{i+1}: Duplicate (Targets same text as previous question)")
                    with open("rejections.log", "a", encoding="utf-8") as f:
                        f.write(f"REJECTED (Duplicate): {q}\n")
                    continue
                seen_quotes.add(norm_quote)
            
            entry = f"## Q{i+1}: {q}\n{solution}\n\n---\n\n"
            final_output += entry
            print(f"✅ ACCEPTED Q{i+1}")

        print("\n=== Final Report ===")
        print(final_output)
        
        with open("study_guide.md", "w", encoding="utf-8") as f:
            f.write(final_output)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
