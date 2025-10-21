#!/usr/bin/env python3
"""
KothaGPT Tokenizer Training Script
================================

Train a SentencePiece tokenizer on Bengali corpus for KothaGPT.

Usage:
    python training/train_tokenizer.py
    python training/train_tokenizer.py --vocab-size 30000 --model-type bpe
"""

import argparse
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import sentencepiece as spm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/tokenizer_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BengaliTokenizerTrainer:
    """SentencePiece tokenizer trainer optimized for Bengali text."""

    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.corpus_path = Path(self.config['input_file'])
        self.output_dir = Path(self.config['output_dir'])
        self.model_prefix = self.config['model_prefix']

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_default_config(self) -> Dict:
        """Get default configuration for tokenizer training."""
        return {
            'input_file': 'data/final/corpus.txt',
            'output_dir': 'models/tokenizer',
            'model_prefix': 'kothagpt_tokenizer',
            'vocab_size': 400,
            'model_type': 'bpe',  # bpe, unigram, char, word
            'character_coverage': 0.995,
            'input_sentence_size': 1000000,
            'shuffle_input_sentence': True,
            'max_sentence_length': 4192,
            'num_threads': 16,
            # Bengali-specific settings
            'pad_id': 0,
            'unk_id': 1,
            'bos_id': 2,
            'eos_id': 3,
            'control_symbols': ['<pad>', '<s>', '</s>'],
            'byte_fallback': True,
            'split_by_unicode_script': True,
            'split_digits': True,
            'split_by_number': True,
            'add_dummy_prefix': True,
            'remove_extra_whitespaces': True,
            'hard_vocab_limit': True
        }

    def _get_bengali_symbols(self) -> List[str]:
        """Get Bengali-specific symbols and punctuation."""
        return [
            # Bengali numerals
            '০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯',
            # Bengali punctuation
            '।', '॥', 'ঃ', 'ঁ', 'ং', 'ঃ',
            # Common Bengali symbols
            '₹', '৳', '%', '&', '@', '#', '$',
            # Emojis and emoticons (common ones)
            '❤️', '👍', '👎', '😊', '😢', '😂', '🔥', '💯',
            # Special tokens for code-switching
            '<bn>', '<en>', '<mix>',
            # Mathematical symbols
            '+', '-', '×', '÷', '=', '≠', '<', '>', '≤', '≥',
            # Currencies and measurements
            'টাকা', 'ডলার', 'কেজি', 'লিটার', 'মিটার', 'কিলোমিটার'
        ]

    def validate_input(self) -> bool:
        """Validate input corpus file."""
        if not self.corpus_path.exists():
            logger.error(f"❌ Input corpus file not found: {self.corpus_path}")
            return False

        # Check file size
        file_size = self.corpus_path.stat().st_size
        if file_size < 1024:  # Less than 1KB
            logger.error(f"❌ Corpus file too small: {file_size} bytes")
            return False

        # Check number of lines
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)

        if line_count < 10:
            logger.error(f"❌ Corpus has too few lines: {line_count}")
            return False

        logger.info(f"✅ Input validation passed: {line_count:,} lines, {file_size:,} bytes")
        return True

    def preprocess_corpus(self) -> str:
        """Preprocess corpus for tokenizer training."""
        logger.info("🔧 Preprocessing corpus for tokenizer training...")

        # Create temporary preprocessed file
        temp_file = self.output_dir / 'temp_corpus.txt'

        with open(self.corpus_path, 'r', encoding='utf-8') as infile, \
             open(temp_file, 'w', encoding='utf-8') as outfile:

            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue

                # Basic cleaning
                line = line.replace('\t', ' ')
                line = ' '.join(line.split())  # Normalize whitespace

                # Add sentence boundaries for better training
                if not line.endswith(('।', '!', '?', '.')):
                    line += '।'

                outfile.write(line + '\n')

                # Progress logging
                if line_num % 10000 == 0:
                    logger.info(f"Processed {line_num:,} lines...")

        logger.info(f"✅ Preprocessing completed: {temp_file}")
        return str(temp_file)

    def train_tokenizer(self, input_file: str) -> bool:
        """Train SentencePiece tokenizer."""
        logger.info("🚀 Starting tokenizer training...")
        logger.info(f"📊 Configuration: vocab_size={self.config['vocab_size']:,}, "
                   f"model_type={self.config['model_type']}, "
                   f"coverage={self.config['character_coverage']}")

        # Prepare training arguments - use only well-supported arguments
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=self.model_prefix,
            model_type=self.config['model_type'],
            vocab_size=self.config['vocab_size'],
            character_coverage=self.config['character_coverage'],
            input_sentence_size=self.config['input_sentence_size'],
            shuffle_input_sentence=self.config['shuffle_input_sentence'],
            max_sentence_length=self.config['max_sentence_length'],
            num_threads=self.config['num_threads'],
            pad_id=self.config['pad_id'],
            unk_id=self.config['unk_id'],
            bos_id=self.config['bos_id'],
            eos_id=self.config['eos_id'],
            control_symbols=self.config['control_symbols'],
            byte_fallback=self.config['byte_fallback'],
            split_by_unicode_script=self.config['split_by_unicode_script'],
            split_digits=self.config['split_digits'],
            split_by_number=self.config['split_by_number'],
            add_dummy_prefix=self.config['add_dummy_prefix'],
            remove_extra_whitespaces=self.config['remove_extra_whitespaces'],
            hard_vocab_limit=self.config['hard_vocab_limit']
        )

        # Check if model files were created
        model_file = Path(f"{self.model_prefix}.model")
        vocab_file = Path(f"{self.model_prefix}.vocab")

        if model_file.exists() and vocab_file.exists():
            # Move files to output directory
            shutil.move(str(model_file), self.output_dir / model_file.name)
            shutil.move(str(vocab_file), self.output_dir / vocab_file.name)

            logger.info(f"✅ Tokenizer training completed successfully!")
            logger.info(f"📁 Model files saved in: {self.output_dir}")

            return True
        else:
            logger.error("❌ Tokenizer training failed - model files not created")
            return False

    def test_tokenizer(self) -> bool:
        """Test the trained tokenizer."""
        logger.info("🧪 Testing trained tokenizer...")

        model_path = self.output_dir / f"{self.model_prefix}.model"

        try:
            # Load the tokenizer
            sp = spm.SentencePieceProcessor()
            sp.load(str(model_path))

            # Test samples
            test_samples = [
                "আমি বাংলাদেশী এবং বাংলা ভাষায় কথা বলতে ভালোবাসি।",
                "ক্রিকেট বাংলাদেশের জাতীয় খেলা।",
                "প্রযুক্তির উন্নয়ন আমাদের জীবনকে সহজ করে তুলছে।",
                "Hello world! এটি একটি টেস্ট মেসেজ।",
                "আমি ১২৩ টাকা দিয়ে কিছু জিনিস কিনেছি। ❤️"
            ]

            logger.info("📝 Test Results:")
            logger.info("=" * 50)

            for i, sample in enumerate(test_samples, 1):
                tokens = sp.encode(sample, out_type=str)
                token_ids = sp.encode(sample, out_type=int)
                decoded = sp.decode(token_ids)

                logger.info(f"Sample {i}:")
                logger.info(f"Original: {sample}")
                logger.info(f"Tokens: {' '.join(tokens[:10])}{'...' if len(tokens) > 10 else ''}")
                logger.info(f"Token IDs: {token_ids[:10]}{'...' if len(token_ids) > 10 else ''}")
                logger.info(f"Decoded: {decoded}")
                logger.info(f"Vocab size: {sp.vocab_size()}")
                logger.info("-" * 30)

            return True

        except Exception as e:
            logger.error(f"❌ Tokenizer test failed: {e}")
            return False

    def save_tokenizer_config(self):
        """Save tokenizer configuration and metadata."""
        config_path = self.output_dir / f"{self.model_prefix}_config.json"

        config_data = {
            'model_type': self.config['model_type'],
            'vocab_size': self.config['vocab_size'],
            'character_coverage': self.config['character_coverage'],
            'special_tokens': {
                'pad_token': '<pad>',
                'unk_token': '<unk>',
                'bos_token': '<s>',
                'eos_token': '</s>',
                'pad_id': self.config['pad_id'],
                'unk_id': self.config['unk_id'],
                'bos_id': self.config['bos_id'],
                'eos_id': self.config['eos_id']
            },
            'bengali_symbols': self._get_bengali_symbols(),
            'training_date': datetime.now().isoformat(),
            'corpus_info': {
                'source_file': str(self.corpus_path),
                'preprocessed_file': str(self.output_dir / 'temp_corpus.txt')
            }
        }

        import json
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Tokenizer configuration saved: {config_path}")

    def run_training(self) -> bool:
        """Run complete tokenizer training pipeline."""
        logger.info("🎯 Starting KothaGPT tokenizer training...")

        # Validate input
        if not self.validate_input():
            return False

        # Preprocess corpus
        processed_file = self.preprocess_corpus()

        # Train tokenizer
        if not self.train_tokenizer(processed_file):
            return False

        # Save configuration
        self.save_tokenizer_config()

        # Test tokenizer
        if not self.test_tokenizer():
            logger.warning("⚠️ Tokenizer test had issues but training completed")
            return True

        logger.info("🎉 Tokenizer training pipeline completed successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description='KothaGPT Tokenizer Training')
    parser.add_argument('--vocab-size', type=int, default=400,
                       help='Vocabulary size (default: 400)')
    parser.add_argument('--model-type', type=str, default='bpe',
                       choices=['bpe', 'unigram', 'char', 'word'],
                       help='Model type (default: bpe)')
    parser.add_argument('--input-file', type=str, default='data/final/corpus.txt',
                       help='Input corpus file (default: data/final/corpus.txt)')
    parser.add_argument('--output-dir', type=str, default='models/tokenizer',
                       help='Output directory (default: models/tokenizer)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Update config with command line arguments
    config = {
        'input_file': args.input_file,
        'output_dir': args.output_dir,
        'model_prefix': 'kothagpt_tokenizer',
        'vocab_size': args.vocab_size,
        'model_type': args.model_type,
        'character_coverage': 0.995,
        'input_sentence_size': 1000000,
        'shuffle_input_sentence': True,
        'max_sentence_length': 4192,
        'num_threads': 16,
        'pad_id': 0,
        'unk_id': 1,
        'bos_id': 2,
        'eos_id': 3,
        'control_symbols': ['<pad>', '<s>', '</s>'],
        'byte_fallback': True,
        'split_by_unicode_script': True,
        'split_digits': True,
        'split_by_number': True,
        'add_dummy_prefix': True,
        'remove_extra_whitespaces': True,
        'hard_vocab_limit': True
    }

    try:
        trainer = BengaliTokenizerTrainer(config)
        success = trainer.run_training()

        if success:
            logger.info("🎉 KothaGPT tokenizer training completed!")
        else:
            logger.error("💥 Tokenizer training failed!")
            exit(1)

    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()