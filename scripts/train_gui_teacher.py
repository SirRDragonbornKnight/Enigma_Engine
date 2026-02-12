"""
GUI Teacher Training Script - Train an AI to teach others how to use Enigma Engine GUI

This specialized training script:
1. Uses DeepSeek as teacher to train a student on GUI usage
2. Tests with GUI-specific questions (not generic tests)
3. Creates an AI that can both USE the GUI and TEACH others to use it

The resulting model can:
- Navigate and use all GUI tabs (Chat, Training, Models, Settings, etc.)
- Explain GUI features to users
- Generate training data for other AIs about GUI usage
- Troubleshoot common GUI issues

Usage:
    python scripts/train_gui_teacher.py --student small --cycles 5
    python scripts/train_gui_teacher.py --student medium --target 8.5
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import base classes from teach_model
from teach_model import TeacherAI, StudentAI

# GUI-specific test categories
GUI_TEST_CATEGORIES = [
    ("navigation", [
        "How do I start a conversation with the AI?",
        "Where is the Training tab?",
        "How do I open Model Manager?",
        "How do I switch between tabs?",
        "Where can I see my chat history?",
    ]),
    ("training", [
        "How do I train the AI on my data?",
        "What format should training data be in?",
        "What does the Load Base Knowledge button do?",
        "How do I use Quick Train?",
        "How many Q&A pairs should I have for good training?",
    ]),
    ("models", [
        "How do I download a model from HuggingFace?",
        "What is the difference between small and medium models?",
        "How do I know if my GPU can run a model?",
        "How do I switch between different models?",
        "Where are models saved on my computer?",
    ]),
    ("modules", [
        "What is a module in Enigma Engine?",
        "How do I enable voice input?",
        "How do I turn on image generation?",
        "What does the Modules tab show?",
        "Can I use multiple generation modules at once?",
    ]),
    ("teaching_skills", [
        "How would you explain training to a beginner?",
        "Create a simple guide for using the Chat tab.",
        "What are the most common mistakes beginners make?",
        "How do I help someone who is stuck on training?",
        "Write Q&A pairs that teach GUI navigation.",
    ]),
    ("troubleshooting", [
        "The AI gives short/strange responses. How do I fix this?",
        "Training is very slow. What can I do?",
        "My model won't load. What should I check?",
        "The GUI is not responding. What should I try?",
        "How do I know if training worked?",
    ]),
]


class GUITeacherSession:
    """Specialized teaching session for GUI usage training."""
    
    def __init__(
        self,
        teacher: TeacherAI,
        student: StudentAI,
        output_dir: str = "gui_teaching_sessions",
        use_gpu_sharing: bool = True,
    ):
        self.teacher = teacher
        self.student = student
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu_sharing = use_gpu_sharing
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = []
        
        # Load pre-made GUI training data
        self.gui_training_data = self._load_gui_training_data()
        self.teacher_training_data = self._load_teacher_training_data()
        
    def _load_gui_training_data(self) -> List[Dict]:
        """Load the pre-made GUI teacher training data."""
        data_path = Path(__file__).parent.parent / "data" / "specialized" / "gui_teacher_training.txt"
        
        if not data_path.exists():
            logger.warning(f"GUI training data not found: {data_path}")
            return []
            
        pairs = []
        content = data_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        current_q = None
        current_a = []
        in_answer = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('Q:'):
                if current_q and in_answer:
                    pairs.append({'question': current_q, 'answer': '\n'.join(current_a).strip()})
                current_q = stripped[2:].strip()
                current_a = []
                in_answer = False
            elif stripped.startswith('A:') and current_q:
                in_answer = True
                first_line = stripped[2:].strip()
                if first_line:
                    current_a.append(first_line)
            elif current_q and in_answer:
                current_a.append(line.rstrip())
        
        if current_q and in_answer:
            pairs.append({'question': current_q, 'answer': '\n'.join(current_a).strip()})
        
        logger.info(f"Loaded {len(pairs)} GUI training pairs from {data_path}")
        return pairs
    
    def _load_teacher_training_data(self) -> List[Dict]:
        """Load the teacher pedagogy training data."""
        data_path = Path(__file__).parent.parent / "data" / "specialized" / "teacher_training_data.txt"
        
        if not data_path.exists():
            logger.warning(f"Teacher training data not found: {data_path}")
            return []
            
        pairs = []
        content = data_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        current_q = None
        current_a = []
        in_answer = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('Q:'):
                if current_q and in_answer:
                    pairs.append({'question': current_q, 'answer': '\n'.join(current_a).strip()})
                current_q = stripped[2:].strip()
                current_a = []
                in_answer = False
            elif stripped.startswith('A:') and current_q:
                in_answer = True
                first_line = stripped[2:].strip()
                if first_line:
                    current_a.append(first_line)
            elif current_q and in_answer:
                current_a.append(line.rstrip())
        
        if current_q and in_answer:
            pairs.append({'question': current_q, 'answer': '\n'.join(current_a).strip()})
        
        logger.info(f"Loaded {len(pairs)} teacher training pairs from {data_path}")
        return pairs
    
    def run_gui_test(self) -> Tuple[float, Dict[str, float], List[str]]:
        """Test the student on GUI-specific categories."""
        print("\n" + "=" * 60)
        print("TESTING GUI KNOWLEDGE")
        print("=" * 60)
        
        all_scores = []
        category_scores = {}
        weaknesses = []
        
        for category, questions in GUI_TEST_CATEGORIES:
            print(f"\n--- Testing: {category} ---")
            cat_scores = []
            
            for q in questions:
                # Get student's answer
                student_answer = self.student.generate(f"Q: {q}\nA:")
                
                # Teacher evaluates (with GUI context)
                score, improvement = self._evaluate_gui_response(q, student_answer, category)
                cat_scores.append(score)
                all_scores.append(score)
                
                print(f"  Q: {q[:50]}...")
                print(f"  A: {student_answer[:50]}...")
                print(f"  Score: {score}/10")
                
                if score < 7 and improvement:
                    weaknesses.append(improvement)
                    
            avg = sum(cat_scores) / len(cat_scores) if cat_scores else 0
            category_scores[category] = avg
            print(f"  Category average: {avg:.1f}/10")
            
        overall = sum(all_scores) / len(all_scores) if all_scores else 0
        print(f"\nOverall GUI knowledge score: {overall:.1f}/10")
        
        return overall, category_scores, weaknesses
    
    def _evaluate_gui_response(self, question: str, answer: str, category: str) -> Tuple[int, str]:
        """Evaluate response with GUI-specific criteria."""
        prompt = f"""You are evaluating an AI student learning about Enigma Engine GUI.

Category: {category}
Question: {question}
Student's Answer: {answer}

Rate the answer 1-10 based on:
- Accuracy about the GUI (tabs, buttons, features)
- Helpfulness for users learning the GUI
- Clarity of explanation
- For 'teaching_skills' category: ability to generate good teaching content

Respond in this format:
SCORE: [number]
REASON: [why this score]
IMPROVEMENT: [what the student needs to learn about the GUI]"""

        response = self.teacher.generate(prompt, max_tokens=300)
        
        score = 5
        improvement = ""
        
        for line in response.split('\n'):
            if line.startswith('SCORE:'):
                try:
                    score = int(re.search(r'\d+', line).group())
                    score = max(1, min(10, score))
                except:
                    pass
            elif line.startswith('IMPROVEMENT:'):
                improvement = line[12:].strip()
        
        return score, improvement
    
    def run_teaching_cycle(self, cycle_num: int) -> Dict:
        """Run one GUI-focused teaching cycle."""
        print(f"\n{'#' * 60}")
        print(f"# GUI TEACHING CYCLE {cycle_num}")
        print(f"{'#' * 60}")
        
        cycle_data = {
            'cycle': cycle_num,
            'timestamp': datetime.now().isoformat(),
        }
        
        # 1. Test current GUI knowledge
        print("\n[1/4] Testing current GUI knowledge...")
        pre_score, pre_cats, weaknesses = self.run_gui_test()
        cycle_data['pre_test'] = {'overall': pre_score, 'categories': pre_cats}
        
        # 2. Gather training data
        print("\n[2/4] Preparing GUI-focused training data...")
        training_data = []
        
        # Add pre-made GUI training data
        if cycle_num == 1:
            training_data.extend(self.gui_training_data)
            training_data.extend(self.teacher_training_data)
            print(f"  Added {len(self.gui_training_data)} pre-made GUI training pairs")
            print(f"  Added {len(self.teacher_training_data)} teacher pedagogy pairs")
        
        # Generate targeted training for weak areas
        if weaknesses:
            print(f"  Found {len(weaknesses)} areas needing improvement")
            targeted = self._generate_gui_targeted_training(weaknesses)
            training_data.extend(targeted)
            print(f"  Generated {len(targeted)} targeted GUI training pairs")
        
        print(f"  Total training pairs: {len(training_data)}")
        cycle_data['training_pairs'] = len(training_data)
        
        # 3. Train student
        print("\n[3/4] Training student...")
        if training_data:
            if self.use_gpu_sharing:
                print("  Unloading teacher to free GPU memory...")
                self.teacher.unload()
                self.student.set_use_gpu(True)
                if not self.student._loaded:
                    self.student.load()
            
            loss = self.student.train_on_data(training_data, epochs=3)
            cycle_data['final_loss'] = loss
            self.student.save()
            
            if self.use_gpu_sharing:
                print("  Reloading teacher for evaluation...")
                self.student.set_use_gpu(False)
                self.teacher.load()
        
        # 4. Re-test
        print("\n[4/4] Re-testing GUI knowledge...")
        post_score, post_cats, _ = self.run_gui_test()
        cycle_data['post_test'] = {'overall': post_score, 'categories': post_cats}
        
        # Summary
        improvement = post_score - pre_score
        cycle_data['improvement'] = improvement
        
        print(f"\n{'=' * 60}")
        print(f"CYCLE {cycle_num} SUMMARY")
        print(f"{'=' * 60}")
        print(f"Pre-training score:  {pre_score:.1f}/10")
        print(f"Post-training score: {post_score:.1f}/10")
        print(f"Improvement:         {'+' if improvement >= 0 else ''}{improvement:.1f}")
        
        # Save
        with open(self.session_dir / f"cycle_{cycle_num}.json", 'w') as f:
            json.dump(cycle_data, f, indent=2)
        
        self.history.append(cycle_data)
        return cycle_data
    
    def _generate_gui_targeted_training(self, weaknesses: List[str]) -> List[Dict]:
        """Generate GUI-specific training for identified weaknesses."""
        if not weaknesses:
            return []
        
        weakness_text = '\n- '.join(weaknesses[:5])
        
        prompt = f"""The AI student needs to learn more about Enigma Engine GUI in these areas:
- {weakness_text}

Generate 20 Q&A pairs that teach these specific GUI concepts.
Focus on:
- Specific button locations and what they do
- Step-by-step instructions for GUI tasks
- How to explain GUI features to others
- Troubleshooting GUI issues

Format:
Q: [question about GUI]
A: [helpful, accurate answer]

Generate the GUI training pairs:"""

        response = self.teacher.generate(prompt, max_tokens=2500)
        
        # Parse pairs
        pairs = []
        lines = response.strip().split('\n')
        current_q = None
        current_a_lines = []
        in_answer = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('Q:', 'Question:')):
                if current_q and in_answer:
                    answer = '\n'.join(current_a_lines).strip()
                    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
                    if answer:
                        pairs.append({'question': current_q, 'answer': answer})
                current_q = stripped.split(':', 1)[1].strip()
                current_a_lines = []
                in_answer = False
            elif stripped.startswith(('A:', 'Answer:')) and current_q:
                in_answer = True
                first_line = stripped.split(':', 1)[1].strip()
                if first_line:
                    current_a_lines.append(first_line)
            elif current_q and in_answer and not stripped.startswith(('Q:', 'Question:')):
                if stripped:
                    current_a_lines.append(line.rstrip())
        
        if current_q and in_answer:
            answer = '\n'.join(current_a_lines).strip()
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
            if answer:
                pairs.append({'question': current_q, 'answer': answer})
        
        return pairs
    
    def run(self, num_cycles: int = 5, target_score: float = 8.0):
        """Run the full GUI teacher training session."""
        print(f"\n{'=' * 60}")
        print("GUI TEACHER TRAINING SESSION")
        print(f"{'=' * 60}")
        print(f"Teacher: {self.teacher.model_id}")
        print(f"Student: Enigma {self.student.size}")
        print(f"Cycles: {num_cycles}")
        print(f"Target score: {target_score}/10")
        print(f"Pre-made training data: {len(self.gui_training_data) + len(self.teacher_training_data)} pairs")
        print(f"Session dir: {self.session_dir}")
        print(f"{'=' * 60}")
        
        # Load models
        print("\nLoading teacher model...")
        self.teacher.load()
        
        print("Loading student model...")
        self.student.load()
        
        # Run cycles
        for cycle in range(1, num_cycles + 1):
            result = self.run_teaching_cycle(cycle)
            
            if result['post_test']['overall'] >= target_score:
                print(f"\n{'*' * 60}")
                print(f"TARGET REACHED! Score: {result['post_test']['overall']:.1f}/10")
                print(f"{'*' * 60}")
                break
            
            if len(self.history) >= 2:
                recent = [h['improvement'] for h in self.history[-2:]]
                if all(i <= 0 for i in recent):
                    print("\nNo improvement in last 2 cycles.")
        
        self._print_final_summary()
    
    def _print_final_summary(self):
        """Print final training summary."""
        print(f"\n{'=' * 60}")
        print("GUI TEACHER TRAINING COMPLETE")
        print(f"{'=' * 60}")
        
        if not self.history:
            print("No cycles completed.")
            return
        
        first = self.history[0]
        last = self.history[-1]
        
        print(f"Cycles completed: {len(self.history)}")
        print(f"Initial score: {first['pre_test']['overall']:.1f}/10")
        print(f"Final score:   {last['post_test']['overall']:.1f}/10")
        print(f"Total improvement: {last['post_test']['overall'] - first['pre_test']['overall']:.1f}")
        
        print(f"\nGUI Category improvements:")
        for cat in first['pre_test']['categories']:
            pre = first['pre_test']['categories'][cat]
            post = last['post_test']['categories'][cat]
            diff = post - pre
            print(f"  {cat}: {pre:.1f} -> {post:.1f} ({'+' if diff >= 0 else ''}{diff:.1f})")
        
        print(f"\nTrained model saved to: {self.student.model_path}")
        print(f"Session logs: {self.session_dir}")
        
        print(f"\n{'=' * 60}")
        print("YOUR AI CAN NOW:")
        print("- Navigate and explain all GUI tabs")
        print("- Help users with training workflows")
        print("- Troubleshoot common GUI issues")
        print("- TEACH other AIs about the GUI!")
        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Train an AI to teach others how to use Enigma Engine GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train a small model on GUI usage
    python scripts/train_gui_teacher.py --student small --cycles 5
    
    # Train to reach 8.5/10 score
    python scripts/train_gui_teacher.py --student medium --target 8.5
"""
    )
    
    parser.add_argument(
        "--teacher", "-t",
        default="deepseek-32b",
        choices=["deepseek-32b", "llama-8b", "qwen-7b"],
        help="Teacher model (default: deepseek-32b)"
    )
    parser.add_argument(
        "--student", "-s",
        default="small",
        choices=["nano", "micro", "tiny", "small", "medium", "large", "xl"],
        help="Student model size (default: small)"
    )
    parser.add_argument(
        "--cycles", "-c",
        type=int,
        default=5,
        help="Number of teaching cycles (default: 5)"
    )
    parser.add_argument(
        "--target",
        type=float,
        default=8.0,
        help="Target score (1-10, default: 8.0)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="gui_teaching_sessions",
        help="Output directory for session logs"
    )
    parser.add_argument(
        "--no-gpu-sharing",
        action="store_true",
        help="Disable sequential GPU sharing"
    )
    
    args = parser.parse_args()
    
    # Teacher model mapping
    TEACHER_MODELS = {
        "deepseek-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    }
    
    teacher_model = TEACHER_MODELS.get(args.teacher, args.teacher)
    
    # Create teacher and student
    teacher = TeacherAI(model_id=teacher_model)
    student = StudentAI(size=args.student)
    
    # Run GUI teacher training
    session = GUITeacherSession(
        teacher=teacher,
        student=student,
        output_dir=args.output_dir,
        use_gpu_sharing=not args.no_gpu_sharing,
    )
    
    session.run(num_cycles=args.cycles, target_score=args.target)


if __name__ == "__main__":
    main()
