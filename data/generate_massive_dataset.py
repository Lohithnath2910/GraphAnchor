import os
import random

def generate_massive_dataset():
    dataset_dir = r"c:\Users\lohit\Desktop\Project I\data\massive_dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Generate 100 documents
    # The storyline involves a highly complex global corporation "Aegis Global"
    # To test long multi-hop reasoning, we construct a 6-hop secret truth:
    # 1. Aegis Global is led by CEO [Victor Vance]
    # 2. [Victor Vance] secretly funds the [Phantom Protocol]
    # 3. [Phantom Protocol] requires rare [Xenotium Crystals]
    # 4. [Xenotium Crystals] are only mined in [Sector 7G]
    # 5. [Sector 7G] is located on [Isla Sorna]
    # 6. [Isla Sorna] is managed by [Dr. Aris Thorne]
    # We will hide these facts among 90+ filler documents that have confusing overlaps.
    
    docs = {}
    
    # Core multi-hop truth
    docs["doc_001_leadership.md"] = "# Corporate Structure\nAegis Global is the world's leading conglomerate, currently led by CEO Victor Vance."
    docs["doc_002_finance_leak.txt"] = "A leaked financial document reveals that Victor Vance secretly channels billions into the highly classified Phantom Protocol."
    docs["doc_003_phantom_reqs.md"] = "# Protocol Requirements\nThe Phantom Protocol is an energy weapon initiative. It requires rare Xenotium Crystals to stabilize the core."
    docs["doc_004_mining_ops.txt"] = "Mining operations report: Xenotium Crystals are exceptionally rare and are exclusively mined in Sector 7G."
    docs["doc_005_geography.md"] = "# Global Sectors\nSector 7G is a restricted zone. Geographically, it is located entirely on Isla Sorna."
    docs["doc_006_island_management.txt"] = "Isla Sorna is a privately owned island. All operations and management on the island are overseen by Dr. Aris Thorne."
    
    # Generate 94 filler documents with confusing overlaps
    departments = ["HR", "Finance", "R&D", "Logistics", "Security", "Marketing", "Legal", "Operations"]
    locations = ["New York", "London", "Tokyo", "Berlin", "Sydney", "Dubai", "Sector 4B", "Sector 9F", "Isla Nublar"]
    people = ["Alice Vance", "Marcus Cole", "Evelyn Sterling", "John Doe", "Jane Smith", "Dr. Alan Grant", "Victor Stone", "Aris Vance"]
    projects = ["Project Titan", "Project Chimera", "Omega Initiative", "Project Icarus", "Phantom Strike (Game)"]
    materials = ["Uranium", "Lithium", "Titanium", "Carbon Nanotubes", "Xenon Gas"]
    
    for i in range(7, 101):
        dept = random.choice(departments)
        loc = random.choice(locations)
        person = random.choice(people)
        proj = random.choice(projects)
        mat = random.choice(materials)
        
        content_type = random.choice([
            f"# {dept} Report\nOur {loc} office is currently managed by {person}. They are overseeing the logistics for {proj}, which relies heavily on {mat}.",
            f"Employee Bio: {person} works in {dept}. They previously led {proj} before transferring to the {loc} branch.",
            f"# {proj} Status\nThe development of {proj} is stalling due to a shortage of {mat} in our {loc} facility. {person} has been tasked with resolving this.",
            f"Meeting Notes: The board discussed {proj}. It was noted that {person} from {dept} will take over operations. They need more {mat} shipped to {loc}.",
            f"Security Alert: Unauthorized access detected at the {loc} server farm. {person} from {dept} is investigating the breach related to {proj} data."
        ])
        
        # Add some specific tricky overlaps to confuse Vector RAG
        if i % 10 == 0:
            content_type += f"\nNote: Victor Vance occasionally visits {loc}, but does not manage {proj}."
        if i % 15 == 0:
            content_type += f"\nDr. Aris Thorne submitted a request for {mat} for a completely different project."
        if i % 20 == 0:
            content_type += f"\nThe Phantom Protocol is often confused with Phantom Strike, which is just a video game played in the {loc} breakroom."
            
        docs[f"doc_{i:03d}_{dept.lower()}_{loc.lower().replace(' ', '_')}.txt"] = content_type

    # Write all documents
    for filename, content in docs.items():
        filepath = os.path.join(dataset_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)
            
    print(f"Successfully generated {len(docs)} documents in '{dataset_dir}'.")

if __name__ == "__main__":
    generate_massive_dataset()
