#!/usr/bin/env python3
"""
KothaGPT Pipeline Progress Tracker
==================================

Automatic progress tracking and status monitoring for the KothaGPT pipeline.
Tracks completion status, timing, and generates reports for all pipeline steps.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineStatus(Enum):
    """Pipeline step status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class PipelineStep:
    """Represents a single pipeline step with metadata."""
    name: str
    description: str
    status: PipelineStatus = PipelineStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    dependencies: List[str] = None
    output_files: List[str] = None
    logs: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.output_files is None:
            self.output_files = []
        if self.logs is None:
            self.logs = []

class ProgressTracker:
    """Main progress tracking system for KothaGPT pipeline."""

    def __init__(self, progress_file: str = "checkpoints/pipeline_progress.json"):
        self.progress_file = Path(progress_file)
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.steps: Dict[str, PipelineStep] = {}
        self._load_progress()

    def _load_progress(self):
        """Load existing progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)

                for step_name, step_data in data.get('steps', {}).items():
                    step = PipelineStep(**step_data)
                    self.steps[step_name] = step

                logger.info(f"📋 Loaded progress for {len(self.steps)} steps")
            except Exception as e:
                logger.warning(f"Could not load existing progress: {e}")
                self._initialize_default_steps()

    def _save_progress(self):
        """Save current progress to file."""
        try:
            data = {
                'last_updated': datetime.now().isoformat(),
                'total_steps': len(self.steps),
                'completed_steps': len([s for s in self.steps.values() if s.status == PipelineStatus.COMPLETED]),
                'failed_steps': len([s for s in self.steps.values() if s.status == PipelineStatus.FAILED]),
                'steps': {name: asdict(step) for name, step in self.steps.items()}
            }

            with open(self.progress_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"💾 Progress saved to {self.progress_file}")
        except Exception as e:
            logger.error(f"Could not save progress: {e}")

    def _initialize_default_steps(self):
        """Initialize default pipeline steps."""
        default_steps = [
            PipelineStep("data_ingestion", "Data collection and preprocessing"),
            PipelineStep("data_preprocessing", "Data cleaning and corpus creation"),
            PipelineStep("tokenizer_training", "Train SentencePiece tokenizer"),
            PipelineStep("base_model_loading", "Load and prepare base model"),
            PipelineStep("lora_finetuning", "Apply LoRA fine-tuning"),
            PipelineStep("rlhf_preparation", "Prepare RLHF feedback dataset"),
            PipelineStep("reward_model_training", "Train reward model"),
            PipelineStep("rlhf_finetuning", "Apply RLHF alignment"),
            PipelineStep("model_evaluation", "Evaluate model performance"),
            PipelineStep("deployment", "Deploy model and API")
        ]

        for step in default_steps:
            self.steps[step.name] = step

    def add_step(self, name: str, description: str, dependencies: List[str] = None):
        """Add a new pipeline step."""
        if name not in self.steps:
            self.steps[name] = PipelineStep(name, description, dependencies=dependencies or [])
            self._save_progress()
            logger.info(f"➕ Added step: {name}")

    def start_step(self, step_name: str) -> bool:
        """Mark a step as in progress."""
        if step_name not in self.steps:
            logger.error(f"Step '{step_name}' not found")
            return False

        step = self.steps[step_name]
        step.status = PipelineStatus.IN_PROGRESS
        step.start_time = datetime.now()
        step.end_time = None
        step.duration = None
        step.error_message = None

        self._save_progress()
        logger.info(f"🚀 Started step: {step_name}")
        return True

    def complete_step(self, step_name: str, output_files: List[str] = None, logs: List[str] = None) -> bool:
        """Mark a step as completed."""
        if step_name not in self.steps:
            logger.error(f"Step '{step_name}' not found")
            return False

        step = self.steps[step_name]
        if step.status != PipelineStatus.IN_PROGRESS:
            logger.warning(f"Step '{step_name}' was not in progress")

        step.status = PipelineStatus.COMPLETED
        step.end_time = datetime.now()

        if step.start_time:
            step.duration = (step.end_time - step.start_time).total_seconds()

        if output_files:
            step.output_files.extend(output_files)
        if logs:
            step.logs.extend(logs)

        self._save_progress()
        logger.info(f"✅ Completed step: {step_name} ({step.duration:.2f}s)" if step.duration else f"✅ Completed step: {step_name}")
        return True

    def fail_step(self, step_name: str, error_message: str) -> bool:
        """Mark a step as failed."""
        if step_name not in self.steps:
            logger.error(f"Step '{step_name}' not found")
            return False

        step = self.steps[step_name]
        step.status = PipelineStatus.FAILED
        step.end_time = datetime.now()
        step.error_message = error_message

        if step.start_time:
            step.duration = (step.end_time - step.start_time).total_seconds()

        self._save_progress()
        logger.error(f"❌ Failed step: {step_name} - {error_message}")
        return True

    def skip_step(self, step_name: str) -> bool:
        """Mark a step as skipped."""
        if step_name not in self.steps:
            logger.error(f"Step '{step_name}' not found")
            return False

        step = self.steps[step_name]
        step.status = PipelineStatus.SKIPPED
        step.end_time = datetime.now()

        self._save_progress()
        logger.info(f"⏭️ Skipped step: {step_name}")
        return True

    def can_run_step(self, step_name: str) -> bool:
        """Check if a step can be run (dependencies satisfied)."""
        if step_name not in self.steps:
            return False

        step = self.steps[step_name]

        # Check if step is already completed or in progress
        if step.status in [PipelineStatus.COMPLETED, PipelineStatus.IN_PROGRESS]:
            return False

        # Check if all dependencies are completed
        for dep in step.dependencies:
            if dep not in self.steps:
                logger.warning(f"Dependency '{dep}' not found for step '{step_name}'")
                continue

            dep_step = self.steps[dep]
            if dep_step.status != PipelineStatus.COMPLETED:
                logger.debug(f"Step '{step_name}' waiting for dependency '{dep}'")
                return False

        return True

    def get_next_runnable_steps(self) -> List[str]:
        """Get list of steps that can be run next."""
        runnable = []
        for name, step in self.steps.items():
            if self.can_run_step(name):
                runnable.append(name)
        return runnable

    def get_progress_summary(self) -> Dict:
        """Get a summary of current progress."""
        total = len(self.steps)
        completed = len([s for s in self.steps.values() if s.status == PipelineStatus.COMPLETED])
        failed = len([s for s in self.steps.values() if s.status == PipelineStatus.FAILED])
        in_progress = len([s for s in self.steps.values() if s.status == PipelineStatus.IN_PROGRESS])
        pending = len([s for s in self.steps.values() if s.status == PipelineStatus.PENDING])
        skipped = len([s for s in self.steps.values() if s.status == PipelineStatus.SKIPPED])

        return {
            'total_steps': total,
            'completed': completed,
            'failed': failed,
            'in_progress': in_progress,
            'pending': pending,
            'skipped': skipped,
            'completion_percentage': (completed / total * 100) if total > 0 else 0,
            'last_updated': datetime.now().isoformat()
        }

    def print_progress_report(self):
        """Print a formatted progress report."""
        summary = self.get_progress_summary()

        print("\n" + "="*60)
        print("🎯 KOTHAGPT PIPELINE PROGRESS REPORT")
        print("="*60)
        print(f"📊 Completion: {summary['completion_percentage']:.1f}% ({summary['completed']}/{summary['total_steps']})")
        print(f"⏳ In Progress: {summary['in_progress']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⏭️ Skipped: {summary['skipped']}")
        print(f"⏸️ Pending: {summary['pending']}")
        print("-" * 60)

        # Show step details
        for name, step in self.steps.items():
            status_icon = {
                PipelineStatus.COMPLETED: "✅",
                PipelineStatus.IN_PROGRESS: "🔄",
                PipelineStatus.FAILED: "❌",
                PipelineStatus.SKIPPED: "⏭️",
                PipelineStatus.PENDING: "⏸️"
            }.get(step.status, "❓")

            duration_str = f" ({step.duration:.1f}s)" if step.duration else ""

            print(f"{status_icon} {name:<20} - {step.description}{duration_str}")

            if step.error_message:
                print(f"   💥 Error: {step.error_message}")

        print("="*60 + "\n")

    def export_report(self, output_file: str = None) -> str:
        """Export a detailed progress report."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"logs/pipeline_report_{timestamp}.json"

        report = {
            'summary': self.get_progress_summary(),
            'steps': {name: asdict(step) for name, step in self.steps.items()},
            'generated_at': datetime.now().isoformat()
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"📋 Progress report exported to: {output_path}")
        return str(output_path)

# Global progress tracker instance
_progress_tracker = None

def get_progress_tracker() -> ProgressTracker:
    """Get or create global progress tracker instance."""
    global _progress_tracker
    if _progress_tracker is None:
        _progress_tracker = ProgressTracker()
    return _progress_tracker

def reset_progress():
    """Reset the global progress tracker."""
    global _progress_tracker
    _progress_tracker = None

if __name__ == "__main__":
    # Test the progress tracker
    tracker = get_progress_tracker()

    # Add some test steps
    tracker.add_step("test_data", "Test data preparation")
    tracker.add_step("test_model", "Test model training", dependencies=["test_data"])

    # Simulate progress
    tracker.start_step("test_data")
    tracker.complete_step("test_data", output_files=["data/test.csv"])

    tracker.start_step("test_model")
    tracker.complete_step("test_model", output_files=["models/test_model.bin"])

    tracker.print_progress_report()
    tracker.export_report()
