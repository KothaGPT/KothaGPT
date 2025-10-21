#!/usr/bin/env python3
"""
KothaGPT Data Preprocessing Pipeline
===================================

Enhanced data preprocessing script with configuration management,
quality filtering, and comprehensive error handling.

Usage:
    python data_tools/prepare_data.py
    python data_tools/prepare_data.py --config config/data_config.yaml
"""

import argparse
import logging
import pandas as pd
import re
import unicodedata
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Enhanced data preprocessor for Bengali text corpus."""

    def __init__(self, config_path: str = "config/data_config.yaml"):
        self.config = self._load_config(config_path)
        self.raw_dir = Path(self.config['raw_data_dir'])
        self.processed_dir = Path(self.config['processed_data_dir'])
        self.final_dir = Path(self.config['final_data_dir'])

        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)

        # Bengali stopwords (common ones)
        self.bengali_stopwords = {
            'এবং', 'অথবা', 'কিন্তু', 'যে', 'যে', 'তা', 'এই', 'এ', 'ও', 'আর', 'এক',
            'কোন', 'কিছু', 'যেমন', 'যেন', 'তেমনি', 'তখন', 'সেখানে', 'যেখানে',
            'কোথায়', 'কখন', 'কী', 'কেন', 'কার', 'কীভাবে', 'কত', 'যত', 'যদি',
            'যদিও', 'যাতে', 'যেহেতু', 'যেমনটি', 'তাহলে', 'তবুও', 'অথচ', 'নাকি',
            'না', 'নয়', 'হয়', 'হবে', 'হয়েছে', 'হচ্ছে', 'হয়ে', 'হয়নি', 'হয়েছিল',
            'হয়েছেন', 'হয়েছিলেন', 'হয়নি', 'হয়েছিলাম', 'হয়েছিলে', 'হয়েছিলি'
        }

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'raw_data_dir': 'data/raw',
            'processed_data_dir': 'data/processed',
            'final_data_dir': 'data/final',
            'unicode_normalization': 'NFKC',
            'remove_control_chars': True,
            'pii_redaction': True,
            'stopword_filtering': True,
            'min_text_length': 10,
            'max_text_length': 1000,
            'deduplication_threshold': 0.9,
            'language_detection': True,
            'save_formats': ['txt', 'jsonl'],
            'compression': False,
            'min_quality_score': 0.7,
            'max_duplicates_ratio': 0.1
        }

    def normalize_text(self, text: str) -> str:
        """Normalize Bengali text with comprehensive cleaning."""
        if not isinstance(text, str):
            return ""

        # Unicode normalization
        text = unicodedata.normalize(self.config['unicode_normalization'], text)

        # Remove control characters
        if self.config['remove_control_chars']:
            text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # PII Redaction (basic patterns)
        if self.config['pii_redaction']:
            # Email patterns
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
            # Phone numbers (Bangladeshi format)
            text = re.sub(r'\b\d{3,4}[-\s]?\d{3,4}[-\s]?\d{3,4}\b', '[PHONE]', text)
            # ID numbers (basic pattern)
            text = re.sub(r'\b\d{10,17}\b', '[ID]', text)

        return text.strip()

    def filter_by_quality(self, text: str) -> bool:
        """Filter text based on quality criteria."""
        if not text or len(text) < self.config['min_text_length']:
            return False

        if len(text) > self.config['max_text_length']:
            return False

        # Check for minimum Bengali character ratio
        bengali_chars = len(re.findall(r'[\u0980-\u09FF]', text))
        total_chars = len(text.replace(' ', ''))
        if total_chars > 0:
            bengali_ratio = bengali_chars / total_chars
            if bengali_ratio < 0.3:  # At least 30% Bengali characters
                return False

        # Check for repetitive content
        words = text.split()
        if len(words) > 0:
            unique_words = set(words)
            repetition_ratio = len(unique_words) / len(words)
            if repetition_ratio < 0.5:  # Too repetitive
                return False

        return True

    def remove_stopwords(self, text: str) -> str:
        """Remove Bengali stopwords if enabled."""
        if not self.config['stopword_filtering']:
            return text

        words = text.split()
        filtered_words = [word for word in words if word not in self.bengali_stopwords]
        return ' '.join(filtered_words)

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess a pandas DataFrame."""
        logger.info(f"Preprocessing DataFrame with {len(df)} rows")

        # Normalize text
        df['normalized_text'] = df['text'].apply(self.normalize_text)

        # Filter by quality
        df['quality_pass'] = df['normalized_text'].apply(self.filter_by_quality)
        df = df[df['quality_pass']].copy()

        # Remove stopwords
        df['clean_text'] = df['normalized_text'].apply(self.remove_stopwords)

        # Remove empty texts after cleaning
        df = df[df['clean_text'].str.len() > 0].copy()

        logger.info(f"After preprocessing: {len(df)} high-quality texts remaining")
        return df[['text', 'clean_text']]

    def deduplicate_texts(self, texts: List[str]) -> List[str]:
        """Remove near-duplicate texts."""
        if not texts:
            return texts

        threshold = self.config['deduplication_threshold']
        unique_texts = []

        for text in texts:
            is_duplicate = False
            for unique_text in unique_texts:
                # Simple similarity check (can be enhanced with more sophisticated methods)
                similarity = self._calculate_similarity(text, unique_text)
                if similarity > threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_texts.append(text)

        removed_count = len(texts) - len(unique_texts)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} near-duplicate texts")

        return unique_texts

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (Jaccard similarity of words)."""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data in multiple formats."""
        base_name = filename.stem

        # Save as CSV
        csv_path = self.processed_dir / f"{base_name}_processed.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Saved processed CSV: {csv_path}")

        # Save as JSONL (for training)
        jsonl_path = self.processed_dir / f"{base_name}_processed.jsonl"
        df.to_json(jsonl_path, orient='records', lines=True, force_ascii=False)
        logger.info(f"Saved processed JSONL: {jsonl_path}")

    def create_training_corpus(self, all_texts: List[str]) -> str:
        """Create final training corpus from all processed texts."""
        # Deduplicate final corpus
        unique_texts = self.deduplicate_texts(all_texts)

        # Create corpus file
        corpus_path = self.final_dir / "corpus.txt"
        with open(corpus_path, 'w', encoding='utf-8') as f:
            for text in unique_texts:
                f.write(text + '\n')

        # Create metadata
        metadata = {
            'total_texts': len(all_texts),
            'unique_texts': len(unique_texts),
            'corpus_size_mb': corpus_path.stat().st_size / (1024 * 1024),
            'avg_text_length': sum(len(t) for t in unique_texts) / len(unique_texts),
            'processing_date': pd.Timestamp.now().isoformat()
        }

        metadata_path = self.final_dir / "corpus_metadata.json"
        pd.Series(metadata).to_json(metadata_path, indent=2)

        logger.info(f"✅ Training corpus created: {corpus_path}")
        logger.info(f"📊 Corpus stats: {len(unique_texts)} unique texts, {metadata['corpus_size_mb']:.2f} MB")

        return str(corpus_path)

    def process_all_files(self):
        """Process all CSV files in the raw data directory."""
        csv_files = list(self.raw_dir.glob("*.csv"))

        if not csv_files:
            logger.warning(f"No CSV files found in {self.raw_dir}")
            return

        all_processed_texts = []

        for file_path in csv_files:
            try:
                logger.info(f"📖 Processing: {file_path.name}")
                df = pd.read_csv(file_path)

                if 'text' not in df.columns:
                    logger.warning(f"No 'text' column found in {file_path.name}")
                    continue

                # Preprocess the data
                processed_df = self.preprocess_dataframe(df)

                if len(processed_df) == 0:
                    logger.warning(f"No valid texts found in {file_path.name}")
                    continue

                # Save processed data
                self.save_processed_data(processed_df, file_path)

                # Collect texts for corpus
                all_processed_texts.extend(processed_df['clean_text'].tolist())

            except Exception as e:
                logger.error(f"❌ Error processing {file_path.name}: {e}")
                continue

        if all_processed_texts:
            # Create final training corpus
            corpus_path = self.create_training_corpus(all_processed_texts)
            logger.info("🎉 Data preprocessing completed successfully!")

            # Generate summary report
            self._generate_summary_report(all_processed_texts, csv_files)
        else:
            logger.error("❌ No valid texts found in any input files")

    def _generate_summary_report(self, texts: List[str], input_files: List[Path]):
        """Generate a summary report of the preprocessing."""
        report = {
            'input_files': [f.name for f in input_files],
            'total_input_files': len(input_files),
            'total_texts_processed': len(texts),
            'average_text_length': sum(len(t) for t in texts) / len(texts) if texts else 0,
            'min_text_length': min(len(t) for t in texts) if texts else 0,
            'max_text_length': max(len(t) for t in texts) if texts else 0,
            'processing_timestamp': pd.Timestamp.now().isoformat()
        }

        report_path = self.final_dir / "preprocessing_report.json"
        pd.Series(report).to_json(report_path, indent=2)

        logger.info(f"📋 Summary report saved: {report_path}")
        logger.info(f"📈 Processed {report['total_texts_processed']} texts from {report['total_input_files']} files")

def main():
    parser = argparse.ArgumentParser(description='KothaGPT Data Preprocessing Pipeline')
    parser.add_argument('--config', type=str, default='config/data_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        preprocessor = DataPreprocessor(args.config)
        preprocessor.process_all_files()
    except KeyboardInterrupt:
        logger.info("⏹️ Preprocessing interrupted by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
