#!/usr/bin/env python3
"""
Create Test Data for EAMCET Model Testing
Generate sample test questions with ground truth answers
"""

import json
import random
from pathlib import Path
from typing import List, Dict

def create_sample_test_data() -> List[Dict]:
    """Create sample test data for EAMCET model testing"""
    
    sample_questions = [
        {
            "question_id": "Q001",
            "raw_text": """Question Number: 1
Question Id: 001
What is the derivative of x²?
Options:
1. 2x
2. x
3. 2
4. 0""",
            "correct_answer": "A",
            "subject": "Mathematics",
            "difficulty": "Easy"
        },
        {
            "question_id": "Q002", 
            "raw_text": """Question Number: 2
Question Id: 002
Which of the following is a vector quantity?
Options:
1. Mass
2. Temperature
3. Force
4. Time""",
            "correct_answer": "C",
            "subject": "Physics",
            "difficulty": "Medium"
        },
        {
            "question_id": "Q003",
            "raw_text": """Question Number: 3
Question Id: 003
What is the chemical formula for water?
Options:
1. H2O
2. CO2
3. O2
4. N2""",
            "correct_answer": "A",
            "subject": "Chemistry",
            "difficulty": "Easy"
        },
        {
            "question_id": "Q004",
            "raw_text": """Question Number: 4
Question Id: 004
What is the SI unit of electric current?
Options:
1. Volt
2. Ampere
3. Ohm
4. Watt""",
            "correct_answer": "B",
            "subject": "Physics",
            "difficulty": "Medium"
        },
        {
            "question_id": "Q005",
            "raw_text": """Question Number: 5
Question Id: 005
Solve the equation: 2x + 5 = 13
Options:
1. x = 3
2. x = 4
3. x = 5
4. x = 6""",
            "correct_answer": "B",
            "subject": "Mathematics",
            "difficulty": "Easy"
        },
        {
            "question_id": "Q006",
            "raw_text": """Question Number: 6
Question Id: 006
Which gas is known as the silent killer?
Options:
1. Carbon monoxide
2. Carbon dioxide
3. Nitrogen dioxide
4. Sulfur dioxide""",
            "correct_answer": "A",
            "subject": "Chemistry",
            "difficulty": "Medium"
        },
        {
            "question_id": "Q007",
            "raw_text": """Question Number: 7
Question Id: 007
What is the value of π (pi) to two decimal places?
Options:
1. 3.12
2. 3.14
3. 3.16
4. 3.18""",
            "correct_answer": "B",
            "subject": "Mathematics",
            "difficulty": "Easy"
        },
        {
            "question_id": "Q008",
            "raw_text": """Question Number: 8
Question Id: 008
Which of the following is NOT a unit of energy?
Options:
1. Joule
2. Watt
3. Calorie
4. Electron volt""",
            "correct_answer": "B",
            "subject": "Physics",
            "difficulty": "Hard"
        },
        {
            "question_id": "Q009",
            "raw_text": """Question Number: 9
Question Id: 009
What is the pH of a neutral solution?
Options:
1. 0
2. 7
3. 14
4. 10""",
            "correct_answer": "B",
            "subject": "Chemistry",
            "difficulty": "Medium"
        },
        {
            "question_id": "Q010",
            "raw_text": """Question Number: 10
Question Id: 010
Find the area of a circle with radius 5 units.
Options:
1. 25π
2. 50π
3. 75π
4. 100π""",
            "correct_answer": "A",
            "subject": "Mathematics",
            "difficulty": "Medium"
        }
    ]
    
    return sample_questions

def create_advanced_test_data() -> List[Dict]:
    """Create more advanced test data with complex questions"""
    
    advanced_questions = [
        {
            "question_id": "Q101",
            "raw_text": """Question Number: 101
Question Id: 101
A particle moves along a straight line with velocity v = 3t² - 6t + 2 m/s. 
What is the acceleration at t = 2 seconds?
Options:
1. 6 m/s²
2. 12 m/s²
3. 18 m/s²
4. 24 m/s²""",
            "correct_answer": "A",
            "subject": "Physics",
            "difficulty": "Hard",
            "explanation": "Acceleration is the derivative of velocity: a = dv/dt = 6t - 6. At t=2, a = 6(2) - 6 = 6 m/s²"
        },
        {
            "question_id": "Q102",
            "raw_text": """Question Number: 102
Question Id: 102
In a chemical reaction, 2A + B → C, if the rate of disappearance of A is 
0.5 mol/L/s, what is the rate of formation of C?
Options:
1. 0.25 mol/L/s
2. 0.5 mol/L/s
3. 1.0 mol/L/s
4. 2.0 mol/L/s""",
            "correct_answer": "A",
            "subject": "Chemistry",
            "difficulty": "Hard",
            "explanation": "From stoichiometry: rate of C formation = (1/2) × rate of A disappearance = 0.25 mol/L/s"
        },
        {
            "question_id": "Q103",
            "raw_text": """Question Number: 103
Question Id: 103
Find the derivative of f(x) = ln(x² + 1) with respect to x.
Options:
1. 2x/(x² + 1)
2. 1/(x² + 1)
3. 2x
4. x/(x² + 1)""",
            "correct_answer": "A",
            "subject": "Mathematics",
            "difficulty": "Hard",
            "explanation": "Using chain rule: d/dx[ln(x² + 1)] = (1/(x² + 1)) × d/dx(x² + 1) = 2x/(x² + 1)"
        }
    ]
    
    return advanced_questions

def create_subject_specific_data(subject: str, count: int = 5) -> List[Dict]:
    """Create subject-specific test data"""
    
    subject_questions = {
        "Mathematics": [
            {
                "raw_text": "Question Number: 1\nQuestion Id: M001\nWhat is the value of sin(30°)?\nOptions:\n1. 0.5\n2. 0.707\n3. 0.866\n4. 1.0",
                "correct_answer": "A"
            },
            {
                "raw_text": "Question Number: 2\nQuestion Id: M002\nSolve: x² - 4x + 4 = 0\nOptions:\n1. x = 2\n2. x = -2\n3. x = 0\n4. x = 4",
                "correct_answer": "A"
            }
        ],
        "Physics": [
            {
                "raw_text": "Question Number: 1\nQuestion Id: P001\nWhat is the unit of power?\nOptions:\n1. Joule\n2. Watt\n3. Newton\n4. Pascal",
                "correct_answer": "B"
            },
            {
                "raw_text": "Question Number: 2\nQuestion Id: P002\nWhich law states F = ma?\nOptions:\n1. Newton's First Law\n2. Newton's Second Law\n3. Newton's Third Law\n4. Newton's Law of Gravitation",
                "correct_answer": "B"
            }
        ],
        "Chemistry": [
            {
                "raw_text": "Question Number: 1\nQuestion Id: C001\nWhat is the atomic number of carbon?\nOptions:\n1. 4\n2. 6\n3. 8\n4. 12",
                "correct_answer": "B"
            },
            {
                "raw_text": "Question Number: 2\nQuestion Id: C002\nWhich element is most electronegative?\nOptions:\n1. Oxygen\n2. Nitrogen\n3. Fluorine\n4. Chlorine",
                "correct_answer": "C"
            }
        ]
    }
    
    if subject in subject_questions:
        return subject_questions[subject][:count]
    else:
        return []

def save_test_data(test_data: List[Dict], output_path: str):
    """Save test data to JSON file"""
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✅ Test data saved to: {output_file}")
    print(f"📊 Total questions: {len(test_data)}")

def main():
    """Main function to create test data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create test data for EAMCET model testing')
    parser.add_argument('--output', default='test_questions.json', help='Output file path')
    parser.add_argument('--type', choices=['basic', 'advanced', 'all'], default='basic', 
                       help='Type of test data to generate')
    parser.add_argument('--subject', choices=['Mathematics', 'Physics', 'Chemistry'], 
                       help='Generate subject-specific data')
    parser.add_argument('--count', type=int, default=10, help='Number of questions to generate')
    
    args = parser.parse_args()
    
    print("📝 Creating EAMCET Test Data")
    print("=" * 40)
    
    test_data = []
    
    if args.subject:
        # Generate subject-specific data
        test_data = create_subject_specific_data(args.subject, args.count)
        print(f"📚 Generating {args.subject} specific questions...")
    else:
        # Generate general test data
        if args.type == 'basic':
            test_data = create_sample_test_data()
            print("📚 Generating basic test questions...")
        elif args.type == 'advanced':
            test_data = create_advanced_test_data()
            print("📚 Generating advanced test questions...")
        elif args.type == 'all':
            test_data = create_sample_test_data() + create_advanced_test_data()
            print("📚 Generating comprehensive test questions...")
    
    # Add metadata
    metadata = {
        "created_date": "2024-01-01",
        "total_questions": len(test_data),
        "subjects": list(set(q.get('subject', 'Unknown') for q in test_data)),
        "difficulties": list(set(q.get('difficulty', 'Unknown') for q in test_data))
    }
    
    # Save test data
    save_test_data(test_data, args.output)
    
    # Print summary
    print("\n📊 Test Data Summary:")
    print(f"   Total questions: {len(test_data)}")
    print(f"   Subjects: {', '.join(metadata['subjects'])}")
    print(f"   Difficulties: {', '.join(metadata['difficulties'])}")
    print(f"   Output file: {args.output}")
    
    # Show sample questions
    if test_data:
        print("\n📝 Sample Questions:")
        for i, question in enumerate(test_data[:3]):
            print(f"   Q{i+1}: {question.get('raw_text', '')[:100]}...")
            print(f"      Correct Answer: {question.get('correct_answer', 'N/A')}")
    
    print("\n✅ Test data creation complete!")

if __name__ == "__main__":
    main()

# Usage examples:
# python create_test_data.py --output basic_test.json --type basic
# python create_test_data.py --output advanced_test.json --type advanced
# python create_test_data.py --output math_test.json --subject Mathematics --count 10 