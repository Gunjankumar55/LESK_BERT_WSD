from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from collections import Counter
import re
import os
import json
import random

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = 'wsd_secret_key_2023'

# Path for storing feedback data
FEEDBACK_FILE = 'feedback_data.json'

class EnhancedLesk:
    def __init__(self):
        self.feedback = self.load_feedback()
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        
        # Try to load BERT models if available
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            # Load pre-trained model and tokenizer
            print("Loading BERT models...")
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
            self.bert_available = True
            print("BERT models loaded successfully")
        except Exception as e:
            print(f"BERT models not available: {e}")
            print("Continuing without BERT embeddings")
            self.bert_available = False
    
    def load_feedback(self):
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE) as f:
                return json.load(f)
        return {}
    
    def save_feedback(self):
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(self.feedback, f)
    
    def get_wordnet_pos(self, treebank_tag):
        """Convert POS tag to WordNet POS format"""
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    
    def process_context(self, sentence, target_word):
        """Process context words with positional weighting"""
        words = word_tokenize(sentence.lower())
        
        # Find target word position
        target_pos = -1
        for i, word in enumerate(words):
            if word.lower() == target_word.lower():
                target_pos = i
                break
        
        # Process context words with proximity weighting
        context_words = []
        for i, word in enumerate(words):
            if word.isalpha() and word not in self.stopwords:
                lemma = self.lemmatizer.lemmatize(word)
                
                # Weight by proximity to target word (closer = more important)
                if target_pos >= 0:
                    distance = abs(i - target_pos)
                    # Add word multiple times based on proximity (max 5 times for adjacent words)
                    weight = max(1, 6 - distance) if distance <= 5 else 1
                    context_words.extend([lemma] * weight)
                else:
                    context_words.append(lemma)
        
        return context_words
    
    def calculate_overlap_score(self, sense, context):
        """Calculate overlap between sense signature and context with improved weighting"""
        # Create rich signature from sense
        signature = []
        
        # Add definition words (higher weight)
        def_words = [w.lower() for w in word_tokenize(sense.definition()) 
                    if w.isalpha() and w not in self.stopwords]
        signature.extend(def_words * 2)  # Double weight for definition
        
        # Add example words
        for example in sense.examples():
            ex_words = [w.lower() for w in word_tokenize(example) 
                       if w.isalpha() and w not in self.stopwords]
            signature.extend(ex_words)
        
        # Add hypernyms, hyponyms, meronyms and holonyms
        for hypernym in sense.hypernyms():
            hyper_words = [w.lower() for w in word_tokenize(hypernym.definition()) 
                          if w.isalpha() and w not in self.stopwords]
            signature.extend(hyper_words)
        
        for hyponym in sense.hyponyms():
            hypo_words = [w.lower() for w in word_tokenize(hyponym.definition()) 
                         if w.isalpha() and w not in self.stopwords]
            signature.extend(hypo_words)
        
        # Add meronyms and holonyms
        for meronym in sense.part_meronyms() + sense.substance_meronyms():
            meronym_words = [w.lower() for w in word_tokenize(meronym.definition()) 
                            if w.isalpha() and w not in self.stopwords]
            signature.extend(meronym_words)
            
        for holonym in sense.part_holonyms() + sense.substance_holonyms():
            holonym_words = [w.lower() for w in word_tokenize(holonym.definition()) 
                            if w.isalpha() and w not in self.stopwords]
            signature.extend(holonym_words)
        
        # Calculate overlap using Counter for better frequency matching
        context_counter = Counter(context)
        signature_counter = Counter(signature)
        
        # Calculate weighted overlap
        overlap_score = 0
        for word, count in context_counter.items():
            if word in signature_counter:
                # Score is product of frequencies
                overlap_score += count * min(signature_counter[word], 5)
        
        return overlap_score
    
    def bert_similarity(self, sense, context_sentence, target_word):
        """Calculate semantic similarity using BERT embeddings"""
        if not hasattr(self, 'bert_available') or not self.bert_available:
            return 0
            
        try:
            import torch
            
            # Create context-gloss pair as in GlossBERT
            gloss = sense.definition()
            
            # Tokenize
            inputs = self.tokenizer(context_sentence, gloss, return_tensors="pt", 
                                   padding=True, truncation=True, max_length=512)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                
            # Use CLS token embedding for similarity
            similarity = torch.cosine_similarity(
                outputs.last_hidden_state[0, 0], 
                outputs.last_hidden_state[0, inputs.input_ids[0].tolist().index(self.tokenizer.sep_token_id) + 1]
            ).item()
            
            return similarity * 10  # Scale up to be comparable with other scores
        except Exception as e:
            print(f"Error in BERT similarity calculation: {e}")
            return 0
    
    def check_collocations(self, sentence, target_word):
        """Check for common collocations that indicate specific senses"""
        collocations = {
            "bat": {
                "noun.animal": ["flying bat", "bat flying", "bat wings", "vampire bat", "fruit bat", "bat in the dark", "bat at night"],
                "noun.artifact": ["baseball bat", "cricket bat", "swing the bat", "wooden bat", "hit with bat"]
            },
            "bank": {
                "noun.artifact": ["bank account", "bank manager", "bank loan", "bank robbery", "money in bank"],
                "noun.object": ["river bank", "bank of the river", "west bank", "bank erosion", "along the bank"]
            },
            "bass": {
                "noun.animal": ["bass fish", "catch bass", "fishing bass", "largemouth bass"],
                "noun.attribute": ["bass sound", "bass guitar", "bass player", "bass note", "bass drum"]
            },
            "spring": {
                "noun.time": ["spring season", "this spring", "last spring", "spring weather", "spring flowers"],
                "noun.artifact": ["metal spring", "spring coil", "spring mechanism"],
                "noun.object": ["water spring", "hot spring", "spring water"]
            },
            "crane": {
                "noun.animal": ["crane bird", "crane flew", "crane nest", "crane species"],
                "noun.artifact": ["construction crane", "crane operator", "crane lifted"]
            }
        }
        
        if target_word not in collocations:
            return None, 0
        
        # Check for collocations in sentence
        sentence_lower = sentence.lower()
        for domain, phrases in collocations[target_word].items():
            for phrase in phrases:
                if phrase.lower() in sentence_lower:
                    # Find matching sense
                    for sense in wn.synsets(target_word):
                        if sense.lexname() == domain:
                            return sense, 15  # Very high confidence for collocations
        
        return None, 0
    
    def apply_rules(self, word, context, senses):
        """Apply hand-coded rules for common ambiguous words"""
        word = word.lower()
        context_words = set(context)
        
        # Rules for "bat"
        if word == "bat":
            # Animal sense rules
            animal_indicators = {"fly", "flying", "flew", "wing", "wings", "night", 
                                "dark", "cave", "nocturnal", "mammal", "animal", "leather", "leathery"}
            if any(indicator in context_words for indicator in animal_indicators):
                # Find animal sense
                for sense in senses:
                    if sense.lexname() == "noun.animal":
                        return 10, sense  # High confidence boost
            
            # Sports equipment rules
            sports_indicators = {"hit", "swing", "ball", "baseball", "cricket", 
                                "player", "game", "sport", "team", "wooden"}
            if any(indicator in context_words for indicator in sports_indicators):
                # Find artifact sense
                for sense in senses:
                    if sense.lexname() == "noun.artifact":
                        return 8, sense  # High confidence boost
        
        # Rules for "bank"
        elif word == "bank":
            # Financial institution rules
            finance_indicators = {"money", "account", "deposit", "withdraw", "loan", 
                                 "credit", "debit", "financial", "cash", "check"}
            if any(indicator in context_words for indicator in finance_indicators):
                for sense in senses:
                    if "financial" in sense.definition() or "money" in sense.definition():
                        return 10, sense
            
            # River bank rules
            river_indicators = {"river", "stream", "water", "flow", "shore", "beach"}
            if any(indicator in context_words for indicator in river_indicators):
                for sense in senses:
                    if "river" in sense.definition() or "stream" in sense.definition():
                        return 10, sense
        
        # Rules for "bass"
        elif word == "bass":
            # Fish sense rules
            fish_indicators = {"fish", "fishing", "catch", "caught", "water", "lake", "river"}
            if any(indicator in context_words for indicator in fish_indicators):
                for sense in senses:
                    if sense.lexname() == "noun.animal":
                        return 10, sense
            
            # Sound/music sense rules
            music_indicators = {"music", "sound", "guitar", "player", "band", "note", "tone", "instrument", "concert", "loud"}
            if any(indicator in context_words for indicator in music_indicators):
                for sense in senses:
                    if sense.lexname() == "noun.attribute" or "music" in sense.definition():
                        return 10, sense
        
        # No rule matched with high confidence
        return 0, None
    
    def safe_compare_synsets(self, synset1, synset2):
        """Safely compare two synsets, handling None values."""
        if synset1 is None or synset2 is None:
            return synset1 is synset2  # True only if both are None
        
        # Use the built-in equality check for synsets
        try:
            return synset1 == synset2
        except AttributeError:
            return False  # If comparison fails, they're not equal
    
    def disambiguate(self, sentence, word):
        """Disambiguate a word in a given sentence context"""
        word = word.lower()
        
        # Get POS tag for the target word
        word_tokens = word_tokenize(sentence)
        pos_tags = pos_tag(word_tokens)
        word_pos = None
        
        for token, pos in pos_tags:
            if token.lower() == word:
                word_pos = self.get_wordnet_pos(pos)
                break
        
        # Get senses filtered by POS if available
        if word_pos:
            senses = [s for s in wn.synsets(word) if s.pos() == word_pos]
            if not senses:
                senses = wn.synsets(word)
        else:
            senses = wn.synsets(word)
        
        if not senses:
            return None, []
        
        # Process context with positional weighting
        context = self.process_context(sentence, word)
        
        # 1. Check for collocations first (highest priority)
        collocation_sense, collocation_score = self.check_collocations(sentence, word)
        if collocation_sense and collocation_score > 0:
            # Return the collocation sense and remaining senses as alternatives
            top_senses = [s for s in senses if not self.safe_compare_synsets(s, collocation_sense)][:3]
            return collocation_sense, top_senses
        
        # 2. Apply rules for common ambiguous words
        rule_score, rule_sense = self.apply_rules(word, context, senses)
        
        # Score each sense
        scored_senses = []
        for sense in senses:
            # If this sense was selected by rules, add the rule score
            # FIX: Use safe comparison to prevent AttributeError
            rule_boost = rule_score if (rule_sense is not None and self.safe_compare_synsets(sense, rule_sense)) else 0
            
            # Calculate base score using overlap
            overlap_score = self.calculate_overlap_score(sense, context)
            
            # Calculate BERT similarity if available
            bert_score = 0
            if hasattr(self, 'bert_available') and self.bert_available:
                bert_score = self.bert_similarity(sense, sentence, word)
            
            # Apply feedback boost if available
            feedback_key = f"{word}_{hash(sentence) % 10000}"
            feedback_score = self.feedback.get(feedback_key, {}).get(sense.name(), 0)
            
            # Calculate final score as weighted combination
            final_score = (
                overlap_score * 0.4 + 
                bert_score * 0.3 + 
                rule_boost * 0.2 + 
                feedback_score * 0.1
            )
            
            scored_senses.append((final_score, sense))
        
        scored_senses.sort(reverse=True, key=lambda x: x[0])
        
        if not scored_senses:
            return None, []
            
        best_sense = scored_senses[0][1]
        top_senses = [s[1] for s in scored_senses[1:4]]
        return best_sense, top_senses
    
    def add_feedback(self, word, context, correct_sense):
        """Store user feedback to improve future disambiguation"""
        # Create a key based on word and hashed context
        context_str = ' '.join(context[:10])  # Use first 10 context words
        key = f"{word}_{hash(context_str) % 10000}"
        
        if key not in self.feedback:
            self.feedback[key] = {}
        
        # Increase score for the correct sense
        self.feedback[key][correct_sense] = self.feedback[key].get(correct_sense, 0) + 5
        
        # Optionally decrease scores for other senses
        for sense in wn.synsets(word):
            if sense.name() != correct_sense and sense.name() in self.feedback[key]:
                self.feedback[key][sense.name()] = max(0, self.feedback[key][sense.name()] - 1)
        
        self.save_feedback()
        
        # Return the updated sense information
        for sense in wn.synsets(word):
            if sense.name() == correct_sense:
                return {
                    'definition': sense.definition(),
                    'examples': sense.examples()
                }
        
        return None

# Initialize the Lesk processor
lesk_processor = EnhancedLesk()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        target_word = request.form.get('target_word', '')
        return redirect(url_for('results', text=text, word=target_word))
    return render_template('index.html')

@app.route('/results')
def results():
    text = request.args.get('text', '')
    target_word = request.args.get('word', '').lower()
    
    if not target_word:
        # Find ambiguous words (with multiple senses)
        words = word_tokenize(text.lower())
        ambiguous_words = []
        for word in words:
            if word.isalpha() and len(wn.synsets(word)) > 1:
                ambiguous_words.append(word)
        
        # If there are ambiguous words, use the first one
        if ambiguous_words:
            target_word = ambiguous_words[0]
    
    best_sense = None
    top_senses = []
    highlighted_text = text
    sentence = ""
    context_words = []
    
    if target_word:
        sentences = sent_tokenize(text)
        for sent in sentences:
            if re.search(r'\b' + re.escape(target_word) + r'\b', sent, re.I):
                sentence = sent
                context_words = lesk_processor.process_context(sent, target_word)
                try:
                    best_sense, top_senses = lesk_processor.disambiguate(sent, target_word)
                except Exception as e:
                    print(f"Disambiguation error: {e}")
                    return render_template('error.html', 
                                         error_message=f"Could not disambiguate the word '{target_word}'. Please try a different word or sentence.",
                                         error_details=str(e))
                
                highlighted_text = re.sub(
                    r'\b' + re.escape(target_word) + r'\b',
                    f'<span class="highlight-word">{target_word}</span>',
                    text,
                    flags=re.IGNORECASE
                )
                break
    
    # Store in session for feedback
    if best_sense:
        session['last_disambiguation'] = {
            'word': target_word,
            'context': context_words,
            'sentence': sentence
        }
    
    return render_template('results.html',
                         text=text,
                         highlighted_text=highlighted_text,
                         target_word=target_word,
                         best_sense=best_sense,
                         top_senses=top_senses,
                         sentence=sentence,
                         context_words=', '.join([w for w in set(context_words)][:10]))  # Show unique context words

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    word = data.get('word')
    context = data.get('context', [])
    correct_sense = data.get('correct_sense')
    
    if word and correct_sense:
        updated_sense = lesk_processor.add_feedback(word, context, correct_sense)
        return jsonify(updated_sense)
    
    return jsonify({'error': 'Invalid feedback data'}), 400

@app.route('/lesk-explained')
def lesk_explained():
    return render_template('lesk_explained.html')

# Add error template handler
@app.route('/error')
def error():
    error_message = request.args.get('message', 'An unknown error occurred')
    error_details = request.args.get('details', '')
    return render_template('error.html', error_message=error_message, error_details=error_details)

if __name__ == '__main__':
    app.run(debug=True)
