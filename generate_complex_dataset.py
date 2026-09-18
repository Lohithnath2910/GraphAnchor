import os
import shutil

def create_dataset():
    dataset_dir = "test_dataset_complex"
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    docs = {
        "doc_01_leadership.md": """# Corporate Leadership
The Nexus Group is spearheaded by our visionary CEO, Alice Vance. 
She is directly responsible for overseeing all highly classified and experimental initiatives, collectively known as the 'Secret Projects' division.
""",
        "doc_02_initiatives.md": """# Current Initiatives
Under the Secret Projects division, the primary focus this year is 'Project Chimera'. 
This project aims to revolutionize computing power and is strictly confidential.
""",
        "doc_03_chimera_details.txt": """Project Chimera focuses exclusively on the development of next-generation Quantum Cores.
These cores require extremely stable environments to function without decoherence.
""",
        "doc_04_core_development.md": """# Quantum Core Development
The manufacturing and testing of Quantum Cores cannot be done at standard facilities. 
Due to their sensitive nature, all Quantum Core operations are exclusively conducted at Facility 42.
""",
        "doc_05_facility_42.txt": """Facility 42 is a state-of-the-art underground bunker. 
It was chosen for its natural cooling capabilities and geologic stability. 
This facility is located deep in Reykjavik, Iceland.
""",
        "doc_06_other_facilities.md": """# Global Facilities
Nexus Group operates several facilities worldwide.
Facility 17 is located in Berlin, Germany, focusing on automotive AI.
Facility 89 is located in Tokyo, Japan, focusing on robotics.
None of these handle experimental initiatives.
""",
        "doc_07_other_projects.md": """# Corporate Events
The annual shareholder meeting is held at the Nexus Group's Nevada testing grounds, USA.
This location is entirely separate from any research initiatives.
""",
        "doc_08_alice_bio.txt": """Alice Vance has a background in materials science. Before becoming CEO, she spent 10 years at Horizon Tech.
She frequently visits the London headquarters, but her main office is in New York.
""",
        "doc_09_quantum_physics.md": """# Basics of Quantum Computing
Quantum computing leverages qubits. Decoherence is a major issue, which is why most quantum computers are kept at near absolute zero. 
We partner with Icelandic universities for theoretical research, though they are not involved in manufacturing.
""",
        "doc_10_security.txt": """Security clearance for the Secret Projects division is strictly Level 5.
Any unauthorized access attempts will be reported directly to the Director of Security, Marcus Cole.
"""
    }

    for filename, content in docs.items():
        with open(os.path.join(dataset_dir, filename), "w") as f:
            f.write(content)
            
    print(f"Created {len(docs)} documents in '{dataset_dir}'")

if __name__ == "__main__":
    create_dataset()
