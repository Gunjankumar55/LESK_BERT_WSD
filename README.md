# Word Sense Disambiguation Tool

A modern web application for disambiguating ambiguous words using the Enhanced Lesk Algorithm with BERT semantic similarity and machine learning techniques.



## 🌟 Features

- **Enhanced Lesk Algorithm**: Utilizes an improved version of the classic Lesk algorithm
- **BERT Semantic Similarity**: Optional integration with BERT for better understanding of context
- **Context Analysis**: Analyzes surrounding words with proximity weighting
- **Smart Caching**: Uses Flask-Caching to improve performance
- **User Feedback**: Learns from user input to improve future disambiguations
- **Dark Mode**: Supports light and dark themes with automatic system preference detection
- **API Access**: RESTful API for programmatic access to the disambiguation engine
- **Mobile Responsive**: Works on devices of all sizes

## 📋 Table of Contents

- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [How It Works](#-how-it-works)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Demo

Try the live demo: [Word Sense Disambiguation Tool](https://wsd-tool.herokuapp.com/) *(Replace with your actual deployment URL)*

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/wsd-tool.git
cd wsd-tool
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Download NLTK resources:

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords')"
```

## 🎮 Usage

1. Start the Flask application:

```bash
python app.py
```

2. Open your browser and navigate to:

```
http://localhost:5000
```

3. Enter a sentence with an ambiguous word and click "Disambiguate"
   - Optionally specify the target word
   - Use the provided examples to get started

## 📚 API Documentation

The application provides a RESTful API for programmatic access:

### Disambiguate a word

**Endpoint:** `/api/disambiguate`

**Method:** POST

**Request Format:**
```json
{
  "text": "I saw a bat flying in the dark.",
  "target_word": "bat"  // Optional, if omitted the system will try to find an ambiguous word
}
```

**Response Format:**
```json
{
  "word": "bat",
  "sentence": "I saw a bat flying in the dark.",
  "best_sense": {
    "name": "bat.n.01",
    "definition": "nocturnal mouselike mammal with forelimbs modified to form membranous wings",
    "examples": ["the bat moved silently through the night air"],
    "lexname": "noun.animal"
  },
  "context_words": ["see", "fly", "dark"],
  "alternative_senses": [
    {
      "name": "bat.n.02",
      "definition": "an implement used for hitting the ball in various games",
      "examples": ["a baseball bat"],
      "lexname": "noun.artifact"
    },
    // ...more senses
  ]
}
```

## 🧠 How It Works

The Word Sense Disambiguation (WSD) tool uses several approaches to determine the most likely meaning of an ambiguous word:

1. **Context Analysis**: The system analyzes words surrounding the target word
2. **Enhanced Lesk Algorithm**: Compares context with definition and examples from WordNet
3. **BERT Integration**: Uses BERT embeddings to compare semantic similarity
4. **Rule-Based Components**: Hand-crafted rules for common ambiguous words
5. **Collocation Detection**: Recognizes common phrases that indicate specific meanings
6. **Feedback Learning**: Improves based on user feedback

For more details, visit the "About Lesk Algorithm" page in the application.


Made with ❤️ by [Gunjankumar Choudhari]
