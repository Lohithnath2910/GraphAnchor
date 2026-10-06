"""
Builds gold_qa.json for benchmark_v5 (new Calder Reach corpus + the 47 real-text questions).
Run from the project root:  python benchmark_v5/build_gold.py

Document numbers in `source_docs` refer to the NN_ prefix of the files in benchmark_v5/corpus/.
The runner (eval/run_benchmark.py) ignores the extra fields; they exist so a human can audit each item.
Question ids: 1-47 are the real-text questions, 101+ are the Calder Reach questions.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent

# (category, question, expected_keywords, source_docs)
Q = [
    # ------------------------------------------------------------------ single_hop (control group)
    ("single_hop", "Who directs the Lodestar Timing Station?", ["Prowse"], [5]),
    ("single_hop", "What is the name of the hub casting batch whose porosity caused the Tidewrack seal failures in February 2023?", ["AH-14"], [15]),
    ("single_hop", "Which membrane batch caused the April 2022 fouling at the Wetmoor Water Reuse project?", ["C-31"], [17]),
    ("single_hop", "Who is the chief executive of Strand & Keel Shipping?", ["Kallenbach"], [31]),
    ("single_hop", "What drought-tolerant wheat variety did the Seedvault North project release in 2022?", ["Brindlewheat-3"], [7]),
    ("single_hop", "Who chairs the Calder Reach Development Board?", ["Vandermeer"], [29]),
    ("single_hop", "Who heads the Audit Office of the Calder Reach Development Board?", ["Obuya"], [29]),
    ("single_hop", "Who directs the Merrow Cold Chain project?", ["Haidari"], [12]),
    ("single_hop", "What is the name of the upper reservoir of the Quarrel Hill Pumped Storage plant?", ["Ironwater"], [2]),
    ("single_hop", "Who leads the Larkspur Telehealth Network?", ["Eldridge"], [14]),
    ("single_hop", "What caused the cable cut at Hollin Pass in June 2023?", ["dredging"], [27]),
    ("single_hop", "Who manages Corran Mill?", ["Rivera"], [33]),

    # ------------------------------------------------------------------ multi_hop_2
    ("multi_hop_2", "Who is the chief executive of the company that made the rotor hubs of the Tidewrack Tidal Array?", ["Tavares"], [1, 33]),
    ("multi_hop_2", "Who manages the plant that made the rotor hubs of the Tidewrack Tidal Array?", ["Demir"], [1, 33]),
    ("multi_hop_2", "Which freight company previously employed the director of the Gantry Line?", ["Strand & Keel"], [4, 39]),
    ("multi_hop_2", "Which bank previously employed the director of the Dunmarrow Port Automation project?", ["Harbourgate"], [6, 41]),
    ("multi_hop_2", "Who chairs the bank that financed the Ferrous Bay Offshore Wind farm?", ["Osterhout"], [10, 32]),
    ("multi_hop_2", "Who is the chief risk officer of the bank that financed the Tidewrack Tidal Array?", ["Lindqvist"], [1, 32]),
    ("multi_hop_2", "Who is the chief executive of the company that buys the electricity produced by the Tidewrack Tidal Array?", ["Rask"], [1, 35]),
    ("multi_hop_2", "Who directs the station that supplies the timing signal for the cranes at Dunmarrow Port?", ["Prowse"], [6, 5]),
    ("multi_hop_2", "Who manages the plant where the director of the Cinder Lake works previously worked?", ["Demir"], [9, 33]),
    ("multi_hop_2", "Who is the chief executive of the freight company that carries the isotopes made at Isotope Works?", ["Kallenbach"], [8, 31]),
    ("multi_hop_2", "Who is the chief executive of the hospital trust that hosts Isotope Works?", ["Anand"], [8, 36]),
    ("multi_hop_2", "Who directs the institute whose audit team investigated the Wetmoor membrane fouling?", ["Brandvik"], [17, 30]),
    ("multi_hop_2", "Who is the chief executive of the group that supplied the membranes for the Wetmoor Water Reuse plant?", ["Tavares"], [3, 33]),
    ("multi_hop_2", "Who manages the mill that made the membranes for the Wetmoor Water Reuse plant?", ["Rivera"], [3, 33]),
    ("multi_hop_2", "Who is the chief executive of the electricity company that supplies priority power to Isotope Works?", ["Rask"], [8, 35]),
    ("multi_hop_2", "Who manages the plant that made the pump casings for the Quarrel Hill Pumped Storage plant?", ["Demir"], [2, 33]),
    ("multi_hop_2", "Who chairs the body that funded the Larkspur Telehealth Network?", ["Vandermeer"], [14, 29]),
    ("multi_hop_2", "Who is the chief executive of the freight company that carries the recovered products of the Cinder Lake works?", ["Kallenbach"], [9, 31]),
    ("multi_hop_2", "Who chairs the cooperative whose seed sales fund the Seedvault North project?", ["Fairbairn"], [7, 34]),
    ("multi_hop_2", "Who chairs the bank that financed the port project whose biggest customer is Strand & Keel Shipping?", ["Osterhout"], [6, 32]),
    ("multi_hop_2", "Who directs the project that adopted the scheduling method of the Gantry Line light rail project?", ["Adebayo"], [4, 6]),
    ("multi_hop_2", "Who directs the project that supplies reclaimed water to the farms of the Verdant Seeds Cooperative?", ["Mokoena"], [34, 3]),
    ("multi_hop_2", "Who directs the station that supplies timing to the flood barrier at Beacon Hill?", ["Prowse"], [11, 5]),
    ("multi_hop_2", "Who directs the institute that designed the sensor network of the Beacon Hill Flood Barrier?", ["Brandvik"], [11, 30]),

    # ------------------------------------------------------------------ multi_hop_3
    ("multi_hop_3", "Who is the chief executive of the company that made the antenna bearings for the station supplying timing to the Gantry Line?", ["Tavares"], [4, 5, 33]),
    ("multi_hop_3", "Who chairs the bank that financed the wind farm whose export cable failure caused the November 2023 brownout?", ["Osterhout"], [24, 10, 32]),
    ("multi_hop_3", "Who is the chief risk officer of the bank that financed the wind farm whose cable failure cost Isotope Works a production run?", ["Lindqvist"], [22, 10, 32]),
    ("multi_hop_3", "Who is the chief executive of the freight company that carries the products of the plant powered directly by the Tidewrack array?", ["Kallenbach"], [1, 9, 31]),
    ("multi_hop_3", "Who manages the plant that made the hubs of the array that supplies power directly to the Cinder Lake works?", ["Demir"], [9, 1, 33]),
    ("multi_hop_3", "Who directs the institute that hosts the project whose duplicate seed bank was damaged at the Dunmarrow depot?", ["Brandvik"], [21, 7, 30]),
    ("multi_hop_3", "Who is the chief executive of the company that operates the wagons of the cold chain whose Dunmarrow depot stored the Seedvault North duplicate?", ["Kallenbach"], [21, 12, 31]),
    ("multi_hop_3", "Who chairs the board that funds the timing station whose lightning damage preceded the Quarrel Hill grid trip?", ["Vandermeer"], [16, 5, 29]),
    ("multi_hop_3", "Who directs the institute whose audit team investigated the hub failures at the array that powers the Cinder Lake works?", ["Brandvik"], [9, 15, 30]),
    ("multi_hop_3", "Who manages the plant that cast the pump casings for the storage plant that tripped in August 2022 because of a timing error?", ["Demir"], [16, 2, 33]),
    ("multi_hop_3", "Who is the chief executive of the group whose mill made the membranes for the plant that irrigates the farms of the Verdant Seeds Cooperative?", ["Tavares"], [34, 3, 33]),
    ("multi_hop_3", "Who manages the mill that made the membranes for the plant whose reclaimed water irrigates the farms of the Verdant Seeds Cooperative?", ["Rivera"], [34, 3, 33]),
    ("multi_hop_3", "Who chairs the bank that financed the wind farm whose operations base is at the port held up by the false closure of the Beacon Hill barrier?", ["Osterhout"], [25, 10, 32]),
    ("multi_hop_3", "Who is the chief executive of the freight company whose largest customer is the port project that adopted the Gantry Line's scheduling method?", ["Kallenbach"], [4, 6, 31]),
    ("multi_hop_3", "Who is the chief executive of the electricity company that buys power from the array that supplies the Cinder Lake works?", ["Rask"], [9, 1, 35]),
    ("multi_hop_3", "Who chairs the board that funded the fiber network whose cable cut interrupted the Larkspur Telehealth Network in June 2023?", ["Vandermeer"], [28, 13, 29]),
    ("multi_hop_3", "Who is the chief executive of the hospital trust that hosts the telehealth network disrupted by the Hollin Pass cable cut?", ["Anand"], [27, 14, 36]),
    ("multi_hop_3", "Who directs the cold chain project whose Dunmarrow depot lost seed stock during the brownout caused by the Ferrous Bay export cable fault?", ["Haidari"], [24, 21, 12]),
    ("multi_hop_3", "Who manages the plant that made the gate pivot blocks for the barrier whose sensor network was designed by the institute led by Ottoline Brandvik?", ["Demir"], [30, 11, 33]),
    ("multi_hop_3", "Who is the chief executive of the freight company that carries cargo for the facility hosted by the hospital trust led by Marguerite Anand?", ["Kallenbach"], [36, 8, 31]),
    ("multi_hop_3", "Who directs the breeding project hosted by the institute whose director supervised the leader of the Tidewrack array?", ["Raghunathan"], [1, 30, 7]),
    ("multi_hop_3", "Who is the chief executive of the group whose plant cast the replacement bearing for the station whose degraded signal caused the Dunmarrow crane collision?", ["Tavares"], [20, 19, 33]),
    ("multi_hop_3", "Who manages the plant that made the bearings for the station whose timing signal reaches the cranes of the port project directed by Cormac Adebayo?", ["Demir"], [41, 5, 33]),

    # ------------------------------------------------------------------ multi_hop_4 (four or more links)
    ("multi_hop_4", "Who chairs the bank whose former analyst now directs the port project that relies on timing from the station at Cairn Head?", ["Osterhout"], [5, 6, 41, 32]),
    ("multi_hop_4", "Who chairs the bank that financed the wind farm whose cable failure disrupted the facility hosted by the hospital trust led by Marguerite Anand?", ["Osterhout"], [36, 8, 22, 10, 32]),
    ("multi_hop_4", "Who is the chief executive of the freight company whose largest customer is the port project directed by a former employee of the bank that financed the Tidewrack array?", ["Kallenbach"], [1, 41, 6, 31]),
    ("multi_hop_4", "Who is the chief executive of the group that owns the plant where the director of the works powered by the Tidewrack array used to work?", ["Tavares"], [1, 9, 33]),
    ("multi_hop_4", "Who chairs the board that funded the fiber backbone carrying the timing signal for the light rail line whose director once worked for Strand & Keel Shipping?", ["Vandermeer"], [39, 4, 5, 13, 29]),
    ("multi_hop_4", "Who manages the plant that made the hubs for the array whose director studied under the head of the institute that monitors its wildlife?", ["Demir"], [1, 30, 33]),

    # ------------------------------------------------------------------ aggregation (answers spread over several documents)
    ("aggregation", "Which projects receive their timing signal from the Lodestar Timing Station?", ["Gantry", "Dunmarrow", "Beacon"], [5]),
    ("aggregation", "Which projects did the Harbourgate Mutual Bank finance with construction loans?", ["Tidewrack", "Dunmarrow", "Ferrous"], [32]),
    ("aggregation", "Which projects received large castings or components from Ashgill Works?", ["Tidewrack", "Quarrel", "Beacon", "Gantry", "Lodestar"], [33]),
    ("aggregation", "Which projects does Strand & Keel Shipping carry cargo for?", ["Dunmarrow", "Cinder", "Isotope", "Merrow"], [31]),
    ("aggregation", "Which projects and facilities were disrupted by the November 2023 brownout?", ["Isotope", "Merrow", "Wetmoor"], [24]),
    ("aggregation", "Which projects suffered incidents linked to the August 2022 damage at the Lodestar Timing Station?", ["Gantry", "Dunmarrow", "Quarrel"], [19]),
    ("aggregation", "Which project directors were supervised by Professor Ottoline Brandvik?", ["Halvorsen", "Raghunathan"], [37, 42]),
    ("aggregation", "Which incidents were investigated by the audit team led by Dr. Kasper Idowu?", ["Wetmoor", "Tidewrack", "Beacon", "Gantry"], [46]),
    ("aggregation", "Which current project directors previously worked at Strand & Keel Shipping, the Harbourgate Mutual Bank, Lumen Grid or Ashgill Works?", ["Kowalczyk", "Adebayo", "Valkanova", "Duarte"], [39, 41, 38, 43]),
    ("aggregation", "Which projects were funded by the Calder Reach Development Board?", ["Gantry", "Beacon", "Quarrel", "Hollin", "Wetmoor", "Lodestar", "Larkspur"], [29]),
    ("aggregation", "Which organizations and projects are customers of the Merrow Cold Chain?", ["Idris", "Verdant", "Seedvault"], [12]),
    ("aggregation", "Which generating projects sell electricity to the Lumen Grid Company?", ["Tidewrack", "Ferrous", "Quarrel"], [35]),
    ("aggregation", "Which projects are run by or hosted at the St. Idris Hospital Trust?", ["Isotope", "Larkspur", "Merrow"], [36]),
    ("aggregation", "Which projects work with or are supported by the Marrow Institute of Applied Science?", ["Seedvault", "Tidewrack", "Cinder", "Beacon"], [30]),
]


def main():
    gold = []
    # Real-text control questions (30 course documents, unrelated to the Calder Reach world).
    real = json.loads((ROOT / "real_corpus_v1" / "real_corpus_gold_qa_benchmark.json").read_text(encoding="utf-8"))
    for g in real:
        g = dict(g)
        g["category"] = "real_single_hop"
        g["source_docs"] = [g.pop("source_doc", "")]
        gold.append(g)

    for i, (cat, q, kws, docs) in enumerate(Q, 101):
        gold.append({
            "id": i,
            "category": cat,
            "confidence": "high",
            "question": q,
            "expected_keywords": kws,
            "forbidden_phrases": [],
            "source_docs": docs,
        })

    out = HERE / "gold_qa.json"
    out.write_text(json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8")
    import collections
    print(f"wrote {len(gold)} questions -> {out}")
    print(dict(collections.Counter(g["category"] for g in gold)))


if __name__ == "__main__":
    main()
