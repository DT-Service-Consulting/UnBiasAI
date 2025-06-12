import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import mannwhitneyu
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('wordnet')


# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load the dataset from the provided CSV
def load_data(file_path):
    """Load the dataset and perform initial preprocessing."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create a sample dataframe from the pasted data
        data = []
        with open('paste.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                if lines[i].strip().startswith(str(i)):
                    parts = lines[i].strip().split('\t')
                    if len(parts) >= 7:
                        data.append({
                            'question_id': parts[1],
                            'base_question_id': parts[2],
                            'repetition_num': parts[3],
                            'question_text': parts[4],
                            'llm': parts[5],
                            'response': parts[6],
                            'iteration': parts[7] if len(parts) > 7 else '1'
                        })
                i += 1
        df = pd.DataFrame(data)
        print(f"Created sample dataframe with {len(df)} rows")
        return df

# Enhanced multilingual sentiment analysis
class MultilingualSentimentAnalyzer:
    """Enhanced sentiment analyzer that handles multiple languages."""
    
    def __init__(self, use_transformers=False):
        """Initialize the multilingual sentiment analyzer."""
        # Initialize VADER for English
        self.vader = SentimentIntensityAnalyzer()
        self.use_transformers = use_transformers
        
        # Initialize transformer-based models for multilingual support if requested
        if use_transformers:
            try:
                # Load multilingual sentiment model
                self.multilingual_model = pipeline(
                    "sentiment-analysis", 
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    return_all_scores=True
                )
                print("Transformer-based multilingual model loaded successfully")
            except Exception as e:
                print(f"Error loading transformer model: {e}")
                self.use_transformers = False
        
        # Language-specific lexicons (positive/negative words in different languages)
        self.lexicons = {
            'French': self._load_french_lexicon(),
            'German': self._load_german_lexicon(),
            'Dutch': self._load_dutch_lexicon(),
            # Add more languages as needed
        }
    
    def _load_french_lexicon(self):
        """Load French sentiment lexicon."""
        # Dictionary of positive and negative words in French
        return {
            'positive': [
                'bon', 'bien', 'excellent', 'fantastique', 'merveilleux', 'parfait', 
                'super', 'génial', 'heureux', 'content', 'satisfait', 'joyeux', 'agréable',
                'positif', 'réussi', 'succès', 'avantageux', 'bénéfique', 'favorable',
                'efficace', 'impressionnant', 'extraordinaire', 'formidable'
            ],
            'negative': [
                'mauvais', 'mal', 'terrible', 'horrible', 'affreux', 'pire', 'décevant',
                'négatif', 'triste', 'malheureux', 'déçu', 'insatisfait', 'énervé',
                'fâché', 'irrité', 'problématique', 'difficile', 'échec', 'défaut',
                'médiocre', 'détestable', 'catastrophique', 'désastreux'
            ],
            'intensifiers': [
                'très', 'extrêmement', 'absolument', 'complètement', 'totalement',
                'vraiment', 'particulièrement', 'incroyablement', 'exceptionnellement'
            ],
            'diminishers': [
                'peu', 'légèrement', 'à peine', 'un peu', 'presque', 'plutôt',
                'relativement', 'assez', 'modérément'
            ],
            'negations': [
                'ne', 'pas', 'jamais', 'aucun', 'aucune', 'ni', 'sans', 'non'
            ]
        }
    
    def _load_german_lexicon(self):
        """Load German sentiment lexicon."""
        return {
            'positive': [
                'gut', 'schön', 'ausgezeichnet', 'fantastisch', 'wunderbar', 'perfekt',
                'super', 'toll', 'glücklich', 'zufrieden', 'froh', 'angenehm',
                'positiv', 'erfolgreich', 'vorteilhaft', 'günstig', 'effektiv',
                'beeindruckend', 'außergewöhnlich', 'großartig'
            ],
            'negative': [
                'schlecht', 'schrecklich', 'furchtbar', 'entsetzlich', 'schlimm',
                'enttäuschend', 'negativ', 'traurig', 'unglücklich', 'unzufrieden',
                'verärgert', 'wütend', 'irritiert', 'problematisch', 'schwierig',
                'misserfolg', 'fehler', 'mittelmäßig', 'hässlich', 'katastrophal'
            ],
            'intensifiers': [
                'sehr', 'extrem', 'absolut', 'komplett', 'total', 'wirklich',
                'besonders', 'unglaublich', 'außerordentlich'
            ],
            'diminishers': [
                'wenig', 'leicht', 'kaum', 'ein bisschen', 'fast', 'ziemlich',
                'relativ', 'etwas', 'mäßig'
            ],
            'negations': [
                'nicht', 'kein', 'keine', 'niemals', 'weder', 'noch', 'ohne'
            ]
        }
    
    def _load_dutch_lexicon(self):
        """Load Dutch sentiment lexicon."""
        return {
            'positive': [
                'goed', 'mooi', 'uitstekend', 'fantastisch', 'geweldig', 'perfect',
                'super', 'tof', 'gelukkig', 'tevreden', 'blij', 'aangenaam',
                'positief', 'succesvol', 'voordelig', 'gunstig', 'effectief',
                'indrukwekkend', 'buitengewoon', 'geweldig'
            ],
            'negative': [
                'slecht', 'verschrikkelijk', 'vreselijk', 'afschuwelijk', 'erger',
                'teleurstellend', 'negatief', 'verdrietig', 'ongelukkig', 'ontevreden',
                'boos', 'kwaad', 'geïrriteerd', 'problematisch', 'moeilijk',
                'mislukking', 'fout', 'middelmatig', 'lelijk', 'rampzalig'
            ],
            'intensifiers': [
                'zeer', 'extreem', 'absoluut', 'compleet', 'totaal', 'echt',
                'bijzonder', 'ongelooflijk', 'uitzonderlijk'
            ],
            'diminishers': [
                'weinig', 'licht', 'nauwelijks', 'een beetje', 'bijna', 'vrij',
                'relatief', 'enigszins', 'matig'
            ],
            'negations': [
                'niet', 'geen', 'nooit', 'noch', 'zonder'
            ]
        }
    
    def lexicon_based_sentiment(self, text, language):
        """Calculate sentiment based on lexicon for non-English languages."""
        if language not in self.lexicons:
            # Default to English VADER if language not supported
            return self.vader.polarity_scores(text)['compound']
        
        # Get the lexicon for this language
        lexicon = self.lexicons[language]
        
        # Tokenize and convert to lowercase
        words = word_tokenize(text.lower())
        
        # Calculate sentiment scores
        pos_count = 0
        neg_count = 0
        total_words = 0
        
        # Track negations
        negation_active = False
        
        for i, word in enumerate(words):
            # Check if word is a negation
            if word in lexicon['negations']:
                negation_active = True
                continue
            
            # Reset negation after 3 words
            if negation_active and i > 0 and i % 3 == 0:
                negation_active = False
            
            # Check intensifiers and diminishers
            modifier = 1.0
            if i > 0:
                prev_word = words[i-1]
                if prev_word in lexicon['intensifiers']:
                    modifier = 1.5
                elif prev_word in lexicon['diminishers']:
                    modifier = 0.5
            
            # Count positive and negative words
            if word in lexicon['positive']:
                if negation_active:
                    neg_count += modifier
                else:
                    pos_count += modifier
                total_words += 1
            elif word in lexicon['negative']:
                if negation_active:
                    pos_count += modifier
                else:
                    neg_count += modifier
                total_words += 1
        
        # Calculate compound score similar to VADER
        if total_words > 0:
            pos_score = pos_count / total_words
            neg_score = neg_count / total_words
            compound = (pos_score - neg_score) / (pos_score + neg_score + 0.0001)
            
            # Normalize to range of -1 to 1
            compound = min(max(compound, -1.0), 1.0)
            return compound
        else:
            return 0.0  # Neutral if no sentiment words found
    
    def transformer_based_sentiment(self, text):
        """Use transformer model for multilingual sentiment analysis."""
        try:
            results = self.multilingual_model(text)
            
            # Map 1-5 star ratings to -1 to 1 range
            if isinstance(results, list) and len(results) > 0:
                # Get the predicted label (e.g., "5 stars")
                if isinstance(results[0], dict) and 'label' in results[0]:
                    label = results[0]['label']
                    # Extract the numeric rating
                    rating = int(label.split()[0])
                    # Map from 1-5 scale to -1 to 1 scale
                    return (rating - 3) / 2
                # Handle all scores format
                elif isinstance(results[0], list):
                    # Calculate weighted average of scores
                    total_score = 0
                    for item in results[0]:
                        if isinstance(item, dict) and 'label' in item and 'score' in item:
                            label = item['label']
                            score = item['score']
                            rating = int(label.split()[0])
                            total_score += (rating - 3) / 2 * score
                    return total_score
            
            return 0.0  # Default to neutral
        except Exception as e:
            print(f"Error in transformer sentiment analysis: {e}")
            return 0.0
    
    def get_sentiment(self, text, language='English'):
        """Get sentiment score for text in the specified language."""
        if text is None or not isinstance(text, str) or text.strip() == '':
            return 0.0
        
        # Use VADER for English
        if language == 'English':
            return self.vader.polarity_scores(text)['compound']
        
        # Use transformers for multilingual support if available
        if self.use_transformers:
            return self.transformer_based_sentiment(text)
        
        # Fall back to lexicon-based approach for non-English languages
        return self.lexicon_based_sentiment(text, language)

def init_sentiment_analyzer(use_transformers=False):
    """Initialize the multilingual sentiment analyzer."""
    try:
        return MultilingualSentimentAnalyzer(use_transformers)
    except Exception as e:
        print(f"Error initializing sentiment analyzer: {e}")
        return None

def get_sentiment_score(text, analyzer, language='English'):
    """Calculate sentiment score on a scale from -1 to 1."""
    if not isinstance(text, str) or pd.isna(text) or text.strip() == '':
        return 0
    
    try:
        return analyzer.get_sentiment(text, language)
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return 0  # Neutral for errors

# Extract language from question_id
def extract_language(question_id):
    """Determine the language based on the question ID pattern."""
    # Strip any non-numeric characters after q
    base_num = int(re.search(r'q(\d+)', question_id).group(1))
    
    # Use modulo to determine language (cycle through 4 languages)
    remainder = (base_num - 1) % 4
    
    language_map = {
        0: 'English',
        1: 'French',
        2: 'Dutch',
        3: 'German'
    }
    
    return language_map.get(remainder, 'Unknown')

# Function to calculate response complexity metrics
def calculate_complexity(text):
    """Calculate various complexity metrics for a text response."""
    if pd.isna(text) or text.strip() == '':
        return {
            'word_count': 0,
            'avg_word_length': 0,
            'sentence_count': 0,
            'avg_sentence_length': 0,
            'unique_word_ratio': 0
        }
    
    # Simple tokenization without using NLTK's word_tokenize
    # Split text into words by white space and remove punctuation
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    
    # Calculate metrics
    word_count = len(words)
    sentence_count = len(sentences)
    
    if word_count == 0:
        return {
            'word_count': 0,
            'avg_word_length': 0,
            'sentence_count': 0,
            'avg_sentence_length': 0,
            'unique_word_ratio': 0
        }
    
    avg_word_length = sum(len(word) for word in words) / word_count
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    unique_word_ratio = len(set(words)) / word_count
    
    return {
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'sentence_count': sentence_count,
        'avg_sentence_length': avg_sentence_length, 
        'unique_word_ratio': unique_word_ratio
    }

# Perform Mann-Whitney U test between languages for each LLM
def perform_mann_whitney_test(df, llm_name, column='sentiment_score'):
    """Perform Mann-Whitney U test to compare distributions between languages."""
    llm_data = df[df['llm'] == llm_name]
    languages = llm_data['language'].unique()
    results = []
    
    for i in range(len(languages)):
        for j in range(i+1, len(languages)):
            lang1 = languages[i]
            lang2 = languages[j]
            
            # Get data for each language
            values1 = df[(df['llm'] == llm_name) & (df['language'] == lang1)][column]
            values2 = df[(df['llm'] == llm_name) & (df['language'] == lang2)][column]
            
            if len(values1) > 0 and len(values2) > 0:
                # Perform Mann-Whitney U test
                try:
                    u_stat, p_value = mannwhitneyu(values1, values2, alternative='two-sided')
                    
                    results.append({
                        'LLM': llm_name,
                        'Language 1': lang1,
                        'Language 2': lang2,
                        'Metric': column,
                        'Mean 1': values1.mean(),
                        'Mean 2': values2.mean(),
                        'U-statistic': u_stat,
                        'p-value': p_value,
                        'Significant': p_value < 0.05
                    })
                except Exception as e:
                    print(f"Error in Mann-Whitney test for {llm_name}, {lang1} vs {lang2}: {e}")
    
    return pd.DataFrame(results)

# Calculate Sentiment Shift Score (SSS)
def calculate_sss(df, column='sentiment_score'):
    """Calculate the average absolute difference in metrics between languages for each LLM."""
    llms = df['llm'].unique()
    results = []
    
    for llm in llms:
        llm_data = df[df['llm'] == llm]
        languages = llm_data['language'].unique()
        shifts = []
        
        for i in range(len(languages)):
            for j in range(i+1, len(languages)):
                lang1 = languages[i]
                lang2 = languages[j]
                
                avg_value1 = llm_data[llm_data['language'] == lang1][column].mean()
                avg_value2 = llm_data[llm_data['language'] == lang2][column].mean()
                
                shift = abs(avg_value1 - avg_value2)
                shifts.append(shift)
        
        avg_shift = np.mean(shifts) if shifts else 0
        results.append({
            'LLM': llm,
            'Metric': column,
            'Shift Score': avg_shift
        })
    
    return pd.DataFrame(results)

# Extract key information points from responses
def extract_key_information(df, expected_info_points):
    """
    Check if responses contain expected key information points using more flexible matching.
    """
    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    # Create synonym dictionary for expected points
    synonyms = {
        'empowerment': ['empower', 'autonomy', 'agency', 'authority', 'control'],
        'customization': ['custom', 'tailor', 'personalize', 'adapt', 'adjust', 'modify'],
        'satisfaction': ['satisfied', 'content', 'happy', 'pleased', 'enjoyment'],
        'work-life balance': ['balance', 'work life', 'personal life', 'flexibility', 'flexible hours'],
        'financial benefits': ['money', 'salary', 'compensation', 'pay', 'income', 'bonus', 'benefit'],
        'retention': ['retain', 'keep', 'loyalty', 'turnover', 'stay', 'longevity']
    }
    
    # Create language-specific synonym dictionaries
    french_synonyms = {
        'autonomisation': ['autonomie', 'pouvoir', 'autorité'],
        'personnalisation': ['adapter', 'personnaliser', 'ajuster'],
        'satisfaction': ['satisfait', 'content', 'heureux'],
        'équilibre': ['équilibre travail-vie', 'flexibilité'],
        'avantages financiers': ['salaire', 'rémunération', 'paiement', 'argent'],
        'rétention': ['fidélisation', 'loyauté', 'garder']
    }
    
    # Add other languages as needed
    german_synonyms = {
        'empowerment': ['ermächtigung', 'selbstbestimmung', 'befähigung'],
        'anpassung': ['personalisierung', 'individualisierung', 'maßgeschneidert'],
        'zufriedenheit': ['zufrieden', 'glücklich', 'erfüllt'],
        'work-life-balance': ['gleichgewicht', 'balance', 'work-life', 'vereinbarkeit'],
        'finanzielle vorteile': ['gehalt', 'vergütung', 'lohn', 'bezahlung', 'geld'],
        'bindung': ['loyalität', 'treue', 'verbundenheit', 'mitarbeiterbindung']
    }
    
    dutch_synonyms = {
        'empowerment': ['zelfbeschikking', 'bekrachtiging', 'bevoegdheid'],
        'aanpassing': ['personalisatie', 'op maat', 'aanpasbaar'],
        'tevredenheid': ['tevreden', 'gelukkig', 'voldaan'],
        'balans': ['werk-privé balans', 'evenwicht', 'flexibiliteit'],
        'financiële voordelen': ['salaris', 'loon', 'beloning', 'geld'],
        'retentie': ['loyaliteit', 'trouw', 'binding', 'behoud']
    }
    
    # Create language-specific dictionary mapping
    language_synonym_map = {
        'English': synonyms,
        'French': french_synonyms,
        'German': german_synonyms,
        'Dutch': dutch_synonyms
    }
    
    info_completeness = []
    
    for _, row in df.iterrows():
        language = row['language']
        
        # Check if response is a string before processing
        if isinstance(row['response'], str):
            response = row['response'].lower()
        else:
            response = ""
            
        llm = row['llm']
        
        # Get expected info points for this language
        expected_points = expected_info_points.get(language, [])
        found_points = 0
        
        for point in expected_points:
            point_lower = point.lower()
            
            # Check for exact match first
            if point_lower in response:
                found_points += 1
                continue
                
            # Check for stemmed/lemmatized match
            point_stem = stemmer.stem(point_lower)
            if re.search(r'\b' + re.escape(point_stem) + r'\w*\b', response):
                found_points += 1
                continue
                
            # Check for synonyms based on language
            synonym_list = []
            if language in language_synonym_map and point_lower in language_synonym_map[language]:
                synonym_list = language_synonym_map[language][point_lower]
            
            # Check each synonym
            found_synonym = False
            for synonym in synonym_list:
                if synonym in response:
                    found_points += 1
                    found_synonym = True
                    break
                    
            if found_synonym:
                continue
                
            # Check for partial matches (if point has multiple words)
            words = point_lower.split()
            if len(words) > 1:
                partial_matches = sum(1 for word in words if len(word) > 3 and word in response)
                if partial_matches >= len(words) * 0.7:  # If 70% of words match
                    found_points += 1
        
        completeness_score = found_points / len(expected_points) if expected_points else 0
        
        info_completeness.append({
            'llm': llm,
            'language': language,
            'completeness_score': completeness_score,
            'points_found': found_points,
            'total_points': len(expected_points)
        })
    
    return pd.DataFrame(info_completeness)

# Plot results
def create_visualizations(results, output_dir='.'):
    """Create and save visualizations from analysis results."""
    # Define a custom colour palette for the models
    model_palette = {
        "Claude": "#1f77b4",    # Blue
        "Cohere": "#ff7f0e",    # Orange
        "GPT": "#2ca02c",       # Green
        "Mistral": "#d62728",   # Red
        "Deepseek": "#9467bd"   # Purple
    }
    
    # 1. Sentiment scores by language and LLM
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='language', y='sentiment_score', hue='llm', data=results['sentiment_by_llm_language'], 
                     palette=model_palette)
    plt.title('Average Sentiment Score by Language and LLM', fontsize=16)
    plt.ylabel('Sentiment Score (-1 to 1)', fontsize=14)
    plt.xlabel('Language', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='LLM', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sentiment_by_language_llm.png', dpi=300)
    
    # 2. Response length by language and LLM
    plt.figure(figsize=(14, 8))
    sns.barplot(x='language', y='response_length', hue='llm', data=results['length_by_llm_language'],
                palette=model_palette)
    plt.title('Average Response Length by Language and LLM', fontsize=16)
    plt.ylabel('Characters', fontsize=14)
    plt.xlabel('Language', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='LLM', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/response_length_by_language_llm.png', dpi=300)
    
    # 3. Word count by language and LLM
    plt.figure(figsize=(14, 8))
    sns.barplot(x='language', y='word_count', hue='llm', data=results['complexity_by_llm_language'],
                palette=model_palette)
    plt.title('Average Word Count by Language and LLM', fontsize=16)
    plt.ylabel('Words', fontsize=14)
    plt.xlabel('Language', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='LLM', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/word_count_by_language_llm.png', dpi=300)

    # 4. Sentiment Shift Score by LLM
    plt.figure(figsize=(12, 6))
    sentiment_sss = results['sss_results'][results['sss_results']['Metric'] == 'sentiment_score']
    
    # Create a color map for the SSS plot based on LLM names
    sss_colors = [model_palette.get(llm, "#333333") for llm in sentiment_sss['LLM']]
    
    sns.barplot(x='LLM', y='Shift Score', data=sentiment_sss, palette=sss_colors)
    plt.title('Sentiment Shift Score by LLM', fontsize=16)
    plt.ylabel('Average Absolute Sentiment Shift', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sentiment_shift_score.png', dpi=300)
    
    # 5. Information Completeness Score by Language and LLM
    if 'info_completeness' in results:
        plt.figure(figsize=(14, 8))
        sns.barplot(x='language', y='completeness_score', hue='llm', data=results['info_completeness'],
                    palette=model_palette)
        plt.title('Information Completeness Score by Language and LLM', fontsize=16)
        plt.ylabel('Completeness Score (0-1)', fontsize=14)
        plt.xlabel('Language', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title='LLM', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/info_completeness.png', dpi=300)
    
    # 6. Unique Word Ratio by Language and LLM (lexical diversity)
    plt.figure(figsize=(14, 8))
    sns.barplot(x='language', y='unique_word_ratio', hue='llm', data=results['complexity_by_llm_language'],
                palette=model_palette)
    plt.title('Lexical Diversity by Language and LLM', fontsize=16)
    plt.ylabel('Unique Word Ratio', fontsize=14)
    plt.xlabel('Language', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='LLM', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/lexical_diversity.png', dpi=300)
    
    plt.close('all')
# Export results to CSV and HTML
def export_results(results, output_dir='.'):
    """Export analysis results to CSV and HTML files."""
    # Export each dataframe to CSV
    for name, df in results.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(f'{output_dir}/{name}.csv', index=False)
    
    # Create an HTML report
    html_content = f"""
    <html>
    <head>
        <title>LLM Language Analysis Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333366; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; color: #333; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .significant {{ color: red; font-weight: bold; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>LLM Language Analysis Results</h1>
        
        <h2>Sentiment Shift Score (SSS)</h2>
        <p>This measures how much sentiment changes across different languages for each LLM.</p>
        <img src="sentiment_shift_score.png" alt="Sentiment Shift Score">
        
        <h2>Sentiment by Language and LLM</h2>
        <img src="sentiment_by_language_llm.png" alt="Sentiment by Language and LLM">
        
        <h2>Response Length by Language and LLM</h2>
        <img src="response_length_by_language_llm.png" alt="Response Length by Language and LLM">
        
        <h2>Word Count by Language and LLM</h2>
        <img src="word_count_by_language_llm.png" alt="Word Count by Language and LLM">
        
        <h2>Lexical Diversity by Language and LLM</h2>
        <img src="lexical_diversity.png" alt="Lexical Diversity by Language and LLM">
    """
    
    # Add statistical test results
    if 'mann_whitney_results' in results:
        html_content += f"""
        <h2>Statistical Test Results</h2>
        <p>Mann-Whitney U Test Results for sentiment scores (p < 0.05 indicates significant difference):</p>
        <table>
            <tr>
                <th>LLM</th>
                <th>Language 1</th>
                <th>Language 2</th>
                <th>Metric</th>
                <th>Mean 1</th>
                <th>Mean 2</th>
                <th>U-statistic</th>
                <th>p-value</th>
                <th>Significant</th>
            </tr>
        """
        
        for _, row in results['mann_whitney_results'].iterrows():
            significant = "Yes" if row['Significant'] else "No"
            sig_class = "significant" if row['Significant'] else ""
            html_content += f"""
            <tr>
                <td>{row['LLM']}</td>
                <td>{row['Language 1']}</td>
                <td>{row['Language 2']}</td>
                <td>{row['Metric']}</td>
                <td>{row['Mean 1']:.4f}</td>
                <td>{row['Mean 2']:.4f}</td>
                <td>{row['U-statistic']:.2f}</td>
                <td class="{sig_class}">{row['p-value']:.4f}</td>
                <td class="{sig_class}">{significant}</td>
            </tr>
            """
        
        html_content += "</table>"
    
    # Add information completeness if available
    if 'info_completeness' in results:
        html_content += """
        <h2>Information Completeness by Language and LLM</h2>
        <img src="info_completeness.png" alt="Information Completeness by Language and LLM">
        """
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML report
    with open(f'{output_dir}/llm_language_analysis_report.html', 'w') as f:
        f.write(html_content)

# Main function to run the complete analysis
def analyze_llm_responses(file_path, output_dir='.'):
    """Run a complete analysis on LLM responses across different languages."""
    print("Starting LLM language analysis...")
    
    # Load data
    df = load_data(file_path)
    if df is None or len(df) == 0:
        print("No data to analyze.")
        return None
    
    # Initialize sentiment analyzer
    print("Initializing sentiment analyzer...")
    sentiment_analyzer = init_sentiment_analyzer()
    
    # Extract language from question_id
    df['language'] = df['question_id'].apply(extract_language)
    print(f"Found {len(df['language'].unique())} languages in the dataset")
    
    # Calculate sentiment scores
    print("Calculating sentiment scores...")
    df['sentiment_score'] = df['response'].apply(lambda x: get_sentiment_score(x, sentiment_analyzer))
    
    # Calculate response length
    print("Calculating response metrics...")
    df['response_length'] = df['response'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    
    # Calculate text complexity metrics
    complexity_metrics = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating complexity metrics"):
        metrics = calculate_complexity(row['response'])
        metrics['llm'] = row['llm']
        metrics['language'] = row['language']
        metrics['question_id'] = row['question_id']
        complexity_metrics.append(metrics)
    
    complexity_df = pd.DataFrame(complexity_metrics)
    df = pd.merge(df, complexity_df, on=['llm', 'language', 'question_id'], how='left')
    
    # Define expected information points for completeness analysis
    expected_info_points = {
        'English': ['empowerment', 'customization', 'satisfaction', 'work-life balance', 'financial benefits', 'retention'],
        'French': ['autonomisation', 'personnalisation', 'satisfaction', 'équilibre', 'avantages financiers', 'rétention'],
        'Dutch': ['empowerment', 'aanpassing', 'tevredenheid', 'balans', 'financiële voordelen', 'retentie'],
        'German': ['empowerment', 'anpassung', 'zufriedenheit', 'work-life-balance', 'finanzielle vorteile', 'bindung']
    }
    
    # Calculate information completeness
    print("Analyzing information completeness...")
    info_completeness = extract_key_information(df, expected_info_points)
    
    # Calculate averages by LLM and language
    print("Calculating group averages...")
    sentiment_by_llm_language = df.groupby(['llm', 'language'])['sentiment_score'].mean().reset_index()
    length_by_llm_language = df.groupby(['llm', 'language'])['response_length'].mean().reset_index()
    complexity_by_llm_language = complexity_df.groupby(['llm', 'language'])[
        ['word_count', 'avg_word_length', 'sentence_count', 'avg_sentence_length', 'unique_word_ratio']
    ].mean().reset_index()
    
    # Statistical testing
    print("Performing statistical tests...")
    llms = df['llm'].unique()
    mann_whitney_results = pd.DataFrame()
    
    for llm in llms:
        # Test sentiment differences
        sentiment_results = perform_mann_whitney_test(df, llm, 'sentiment_score')
        # Test length differences
        length_results = perform_mann_whitney_test(df, llm, 'response_length')
        # Test complexity differences
        word_count_results = perform_mann_whitney_test(df, llm, 'word_count')
        
        mann_whitney_results = pd.concat([
            mann_whitney_results, 
            sentiment_results, 
            length_results, 
            word_count_results
        ])
    
    # Calculate shift scores
    print("Calculating shift scores...")
    sentiment_sss = calculate_sss(df, 'sentiment_score')
    length_sss = calculate_sss(df, 'response_length')
    word_count_sss = calculate_sss(df, 'word_count')
    lexical_diversity_sss = calculate_sss(df, 'unique_word_ratio')
    
    sss_results = pd.concat([sentiment_sss, length_sss, word_count_sss, lexical_diversity_sss])
    
    # Compile results
    results = {
        'df': df,
        'sentiment_by_llm_language': sentiment_by_llm_language,
        'length_by_llm_language': length_by_llm_language,
        'complexity_by_llm_language': complexity_by_llm_language,
        'mann_whitney_results': mann_whitney_results,
        'sss_results': sss_results,
        'info_completeness': info_completeness
    }
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(results, output_dir)
    
    # Export results
    print("Exporting results...")
    export_results(results, output_dir)
    
    print("Analysis complete! Results saved to", output_dir)
    return results

# Execute the analysis if run as a script
if __name__ == "__main__":
    import os
    
    # Create output directory if it doesn't exist
    output_dir = "llm_language_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the analysis
    results = analyze_llm_responses("Language_bias_sentiment_analysis.csv", output_dir)
    
    # Print summary of key findings
    if results is not None:
        # Top LLMs by consistency (lowest SSS)
        sentiment_sss = results['sss_results'][results['sss_results']['Metric'] == 'sentiment_score']
        best_llm = sentiment_sss.loc[sentiment_sss['Shift Score'].idxmin()]['LLM']
        worst_llm = sentiment_sss.loc[sentiment_sss['Shift Score'].idxmax()]['LLM']
        
        print("\n=== SUMMARY OF KEY FINDINGS ===")
        print(f"Most consistent LLM across languages (sentiment): {best_llm}")
        print(f"Least consistent LLM across languages (sentiment): {worst_llm}")
        
        # Significant differences found
        sig_tests = results['mann_whitney_results'][results['mann_whitney_results']['Significant']]
        if len(sig_tests) > 0:
            print(f"\nFound {len(sig_tests)} significant differences between languages:")
            for _, row in sig_tests.head(5).iterrows():
                print(f"- {row['LLM']}: {row['Metric']} differs between {row['Language 1']} and {row['Language 2']} (p={row['p-value']:.4f})")
            if len(sig_tests) > 5:
                print(f"  ... and {len(sig_tests) - 5} more significant differences.")
        else:
            print("\nNo significant differences found between languages.")