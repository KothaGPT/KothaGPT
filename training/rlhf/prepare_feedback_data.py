#!/usr/bin/env python3
"""
KothaGPT RLHF Feedback Data Preparation
=====================================

Prepare human feedback data for RLHF training by collecting preference
pairs and formatting them for the reward model training.

Usage:
    python training/rlhf/prepare_feedback_data.py
    python training/rlhf/prepare_feedback_data.py --num-samples 1000
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rlhf_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RLHFDataPreparer:
    """Prepare human feedback data for RLHF training."""

    def __init__(self, num_samples: int = 1000, output_dir: str = "data/processed"):
        self.num_samples = num_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sample prompts for generating preference pairs
        self.sample_prompts = [
            "বাংলাদেশের সংস্কৃতি সম্পর্কে বলুন।",
            "ক্রিকেট খেলা নিয়ে আপনার মতামত কী?",
            "প্রযুক্তির উন্নয়ন কীভাবে আমাদের জীবনকে প্রভাবিত করছে?",
            "বাংলা সাহিত্যের ইতিহাস সম্পর্কে আলোচনা করুন।",
            "পরিবেশ রক্ষায় আমরা কী করতে পারি?",
            "শিক্ষা ব্যবস্থার উন্নয়নে কী কী পদক্ষেপ নেওয়া উচিত?",
            "স্বাস্থ্যকর জীবনযাপনের উপায় কী?",
            "বাংলাদেশের অর্থনৈতিক উন্নয়নের সম্ভাবনা নিয়ে আলোচনা করুন।"
        ]

    def generate_synthetic_responses(self) -> List[Dict]:
        """Generate synthetic response pairs for training."""
        logger.info("🤖 Generating synthetic response pairs...")

        feedback_data = []

        # Different quality responses for the same prompts
        response_templates = {
            "high_quality": [
                "এটি একটি চমৎকার প্রশ্ন। বাংলাদেশের সংস্কৃতি বিশ্বের অন্যতম সমৃদ্ধ সংস্কৃতি। এখানে বিভিন্ন ধর্ম, ভাষা এবং ঐতিহ্যের মিশ্রণ রয়েছে যা আমাদের ইতিহাসকে আরও আকর্ষণীয় করে তুলেছে।",
                "ক্রিকেট বাংলাদেশের জাতীয় খেলা এবং আমাদের জন্য গর্বের বিষয়। ১৯৯৭ সালে টেস্ট স্ট্যাটাস লাভের পর থেকে আমাদের ক্রিকেট দল অনেক উন্নতি করেছে।",
                "প্রযুক্তির উন্নয়ন আমাদের জীবনকে বহুমুখীভাবে প্রভাবিত করছে। শিক্ষা, চিকিৎসা, যোগাযোগ - সব ক্ষেত্রেই প্রযুক্তির ছোঁয়া লেগেছে।"
            ],
            "medium_quality": [
                "বাংলাদেশের সংস্কৃতি ভালো। অনেক ধর্ম আছে। ইতিহাসও আছে।",
                "ক্রিকেট ভালো খেলা। বাংলাদেশে অনেকে খেলে।",
                "প্রযুক্তি উন্নত হচ্ছে। জীবন সহজ হচ্ছে।"
            ],
            "low_quality": [
                "জানি না।",
                "ক্রিকেট খেলা।",
                "প্রযুক্তি।"
            ]
        }

        for i in range(min(self.num_samples, len(self.sample_prompts) * 3)):
            prompt = random.choice(self.sample_prompts)

            # Select two responses of different quality
            quality_levels = ["high_quality", "medium_quality", "low_quality"]
            chosen_response = random.choice(quality_levels)

            # Get appropriate response template
            if chosen_response == "high_quality":
                response = random.choice(response_templates["high_quality"])
                chosen_response = "high"
            elif chosen_response == "medium_quality":
                response = random.choice(response_templates["medium_quality"])
                chosen_response = "medium"
            else:
                response = random.choice(response_templates["low_quality"])
                chosen_response = "low"

            feedback_data.append({
                "prompt": prompt,
                "response": response,
                "quality": chosen_response,
                "source": "synthetic"
            })

        logger.info(f"✅ Generated {len(feedback_data)} synthetic feedback samples")
        return feedback_data

    def load_existing_data(self) -> List[Dict]:
        """Load existing feedback data if available."""
        feedback_file = self.output_dir / "rlhf_feedback.jsonl"

        if not feedback_file.exists():
            return []

        feedback_data = []
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        feedback_data.append(json.loads(line))
            logger.info(f"📋 Loaded {len(feedback_data)} existing feedback samples")
        except Exception as e:
            logger.warning(f"Could not load existing feedback data: {e}")

        return feedback_data

    def create_preference_pairs(self, feedback_data: List[Dict]) -> List[Dict]:
        """Create preference pairs from feedback data."""
        logger.info("🔄 Creating preference pairs...")

        preference_pairs = []

        # Group by prompt for creating pairs
        prompt_groups = {}
        for item in feedback_data:
            prompt = item["prompt"]
            if prompt not in prompt_groups:
                prompt_groups[prompt] = []
            prompt_groups[prompt].append(item)

        for prompt, responses in prompt_groups.items():
            if len(responses) >= 2:
                # Sort by quality and create pairs
                responses_sorted = sorted(responses, key=lambda x: self._quality_score(x["quality"]))

                for i in range(len(responses_sorted) - 1):
                    better_response = responses_sorted[i]
                    worse_response = responses_sorted[i + 1]

                    pair = {
                        "prompt": prompt,
                        "chosen": better_response["response"],
                        "rejected": worse_response["response"],
                        "chosen_quality": better_response["quality"],
                        "rejected_quality": worse_response["quality"],
                        "source": "preference_pair"
                    }
                    preference_pairs.append(pair)

        logger.info(f"✅ Created {len(preference_pairs)} preference pairs")
        return preference_pairs

    def _quality_score(self, quality: str) -> int:
        """Convert quality label to numeric score."""
        scores = {"high": 3, "medium": 2, "low": 1}
        return scores.get(quality, 1)

    def save_feedback_data(self, feedback_data: List[Dict], preference_pairs: List[Dict]):
        """Save feedback data and preference pairs."""
        # Save feedback data
        feedback_file = self.output_dir / "rlhf_feedback.jsonl"
        with open(feedback_file, 'w', encoding='utf-8') as f:
            for item in feedback_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # Save preference pairs
        pairs_file = self.output_dir / "rlhf_preference_pairs.jsonl"
        with open(pairs_file, 'w', encoding='utf-8') as f:
            for pair in preference_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')

        # Save metadata
        metadata = {
            "total_feedback_samples": len(feedback_data),
            "total_preference_pairs": len(preference_pairs),
            "quality_distribution": {},
            "generated_at": "2025-01-21T00:00:00Z"
        }

        # Calculate quality distribution
        quality_count = {}
        for item in feedback_data:
            quality = item.get("quality", "unknown")
            quality_count[quality] = quality_count.get(quality, 0) + 1

        metadata["quality_distribution"] = quality_count

        metadata_file = self.output_dir / "rlhf_feedback_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Saved {len(feedback_data)} feedback samples to {feedback_file}")
        logger.info(f"💾 Saved {len(preference_pairs)} preference pairs to {pairs_file}")
        logger.info(f"💾 Saved metadata to {metadata_file}")

    def run_preparation(self) -> bool:
        """Run the complete RLHF data preparation pipeline."""
        logger.info("🚀 Starting RLHF feedback data preparation...")

        try:
            # Load existing data
            existing_data = self.load_existing_data()

            # Generate new synthetic data if needed
            if len(existing_data) < self.num_samples:
                new_data = self.generate_synthetic_responses()
                all_feedback_data = existing_data + new_data
            else:
                all_feedback_data = existing_data

            # Create preference pairs
            preference_pairs = self.create_preference_pairs(all_feedback_data)

            # Save everything
            self.save_feedback_data(all_feedback_data, preference_pairs)

            logger.info("🎉 RLHF feedback data preparation completed!")
            logger.info(f"📊 Summary: {len(all_feedback_data)} feedback samples, {len(preference_pairs)} preference pairs")

            return True

        except Exception as e:
            logger.error(f"❌ RLHF preparation failed: {e}")
            return False

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="KothaGPT RLHF Feedback Data Preparation")
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of feedback samples to generate (default: 1000)')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Output directory for feedback data (default: data/processed)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    preparer = RLHFDataPreparer(num_samples=args.num_samples, output_dir=args.output_dir)
    success = preparer.run_preparation()

    if success:
        logger.info("✅ RLHF data preparation completed successfully!")
    else:
        logger.error("❌ RLHF data preparation failed!")
        exit(1)

if __name__ == "__main__":
    main()
