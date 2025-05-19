# SenseResolver - Modern WSD with BERT & Lesk

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0.1-green.svg)](https://flask.palletsprojects.com/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8.1-orange.svg)](https://www.nltk.org/)
[![BERT](https://img.shields.io/badge/BERT-4.30.2-yellow.svg)](https://huggingface.co/transformers/)
[![Demo](https://img.shields.io/badge/Demo-Live-success.svg)](https://gkc55-nlp-wsd.hf.space/)

An advanced Word Sense Disambiguation (WSD) system that combines the Lesk algorithm with BERT embeddings for improved accuracy in determining word meanings from context.

## 🚀 Live Demo

Try the live demo: [https://gkc55-nlp-wsd.hf.space/](https://gkc55-nlp-wsd.hf.space/)

## 🎯 Features

- **Enhanced Lesk Algorithm**: Improved version of the traditional Lesk algorithm
- **BERT Integration**: Uses BERT embeddings for better context understanding
- **Interactive Web Interface**: User-friendly Flask-based web application
- **Real-time Feedback System**: Learn from user corrections to improve accuracy
- **Context-Aware Processing**: Considers surrounding words with proximity weighting
- **Multiple Sense Support**: Handles words with multiple meanings effectively

## 🧠 How It Works

1. **Input Processing**: Analyzes text to identify ambiguous words
2. **Context Extraction**: Extracts and weights context words based on proximity
3. **Sense Comparison**: Compares each possible word sense with the context
4. **BERT Similarity**: Uses BERT embeddings to calculate semantic similarity
5. **Sense Selection**: Selects the most appropriate sense based on combined scores
6. **User Feedback**: Learns from user corrections to improve future results

## 🛠️ Technical Stack

- **Backend**: Flask, Python 3.9
- **NLP**: NLTK, BERT Transformers
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Deployment**: Docker, Hugging Face Spaces

## 📋 Prerequisites

- Python 3.9+
- pip (Python package manager)
- Docker (for containerization)

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/Gunjankumar55/LESK_BERT_WSD.git
cd LESK_BERT_WSD
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Access the web interface at `http://localhost:5000`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



## 👨‍💻 Author

**Gunjankumar Choudhari**
- GitHub: [@Gunjankumar55](https://github.com/Gunjankumar55)
- LinkedIn: [Gunjankumar Choudhari](https://linkedin.com/in/gunjankumarchoudhari)
- Portfolio: [Gunjan Portfolio](https://gunjankumar55.github.io/Gunjan_Portfolio/)
