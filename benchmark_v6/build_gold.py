"""
Builds gold_qa.json for benchmark_v6. Run from the project root:  python benchmark_v6/build_gold.py

benchmark_v6 reuses the corpus that is already ingested (benchmark_v5/corpus, 76 documents). Only the questions are new.
What changed relative to v5:
  * Every new question carries `evidence`: (document number, exact text) pairs that establish each hop. This script
    refuses to build if any piece of evidence is missing from its document, so each gold answer is traceable.
  * No answer is used by more than MAX_TAIL multi-hop questions (v5 had 8 questions ending at the same person), so one
    hard-to-find document no longer sinks a dozen questions at once.
  * New questions end at facts that appear in a single document (a deputy's or department head's name, a batch code,
    a cost), not at names that recur across the corpus, so the keyword check cannot be satisfied by accident.
  * Aggregation gold lists were re-verified against the text (v5 missed Lumen Grid as a Lodestar customer).
  * v5 questions whose wording was ambiguous, or that duplicated an over-used answer, are dropped.
Question ids: 1-47 real-text, 101-112 single hop (unchanged from v5), 113-165 kept from v5, 201+ new.
"""
import glob
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
CORPUS = ROOT / "benchmark_v5" / "corpus"
MAX_TAIL = 3


def squash(t):
    return re.sub(r"\s+", " ", t.replace("\r", " ")).lower()


DOCS = {}
for p in glob.glob(str(CORPUS / "[0-9][0-9]_*.txt")):
    DOCS[int(Path(p).name[:2])] = squash(Path(p).read_text(encoding="utf-8", errors="replace"))

# ids kept from benchmark_v5 (multi-hop). Chosen so no answer appears more than MAX_TAIL times; ambiguous ones dropped.
KEEP_V5 = [114, 128, 165, 125, 137, 163, 117, 132, 160, 122, 130, 162, 129, 144, 164, 124, 136, 142,
           120, 135, 119, 127, 151, 118, 139, 123, 153, 126, 148, 131, 133, 134, 154, 157, 115, 116]
KEEP_V5_AGG = []  # v5 aggregation questions are re-written below with verified lists

# (category, question, expected_keywords, source_docs, evidence[(doc, text)])
NEW = [
    # ---------------------------------------------------------------- multi_hop_2: person or incident -> project overview
    ("multi_hop_2", "Who is the deputy director of the project led by the former Harbourgate Mutual Bank infrastructure analyst?", ["Odhiambo"], [41, 6],
     [(41, "analyst in its infrastructure and project finance division"), (41, "directs the dunmarrow port automation project"), (6, "his deputy is wilhelmina odhiambo")]),
    ("multi_hop_2", "Who is the process chief of the recycling works directed by the former Ashgill Works laboratory researcher?", ["Mbeki"], [43, 9],
     [(43, "post in the metallurgical laboratory at ashgill works"), (43, "director of the cinder lake battery recycling works"), (9, "process chief is astrid mbeki")]),
    ("multi_hop_2", "Who heads the laboratory of the water reuse project whose director previously researched membrane fouling at the Marrow Institute?", ["Ruud"], [44, 3],
     [(44, "worked as a researcher at the institute"), (44, "director of the wetmoor water reuse project"), (3, "head of the laboratory is dr. ebba ruud")]),
    ("multi_hop_2", "Who is the signals chief of the light rail project directed by the former Strand & Keel network planner?", ["Lindgren"], [39, 4],
     [(39, "network planner"), (39, "director of the gantry line light rail project"), (4, "signals chief is pavel lindgren")]),
    ("multi_hop_2", "Who is the deputy director of the pumped storage plant whose director started in the region as a protection engineer at the Lumen Grid Company?", ["Thandeka"], [38, 2],
     [(38, "recruited by the lumen grid company as a protection engineer"), (38, "director of the quarrel hill pumped storage plant"), (2, "her deputy is osric thandeka")]),
    ("multi_hop_2", "Who is the deputy director of the breeding project whose founding leader was supervised by the head of the Marrow Institute?", ["Hartigan"], [42, 7],
     [(42, "her supervisor was professor ottoline brandvik"), (42, "founding leader of the seedvault north breeding project"), (7, "her deputy is olumide hartigan")]),
    ("multi_hop_2", "Who is the head of radiochemistry at the isotope facility that the St. Idris Hospital Trust owns?", ["Szabo"], [36, 8],
     [(36, "it owns the isotope works facility"), (8, "head of radiochemistry is dr. vikram szabo")]),
    ("multi_hop_2", "Who is the clinical chief of the telehealth network that the St. Idris Hospital Trust hosts?", ["Tan"], [36, 14],
     [(36, "it hosts the larkspur telehealth network"), (14, "clinical chief is dr. mireille tan")]),
    ("multi_hop_2", "Who is the refrigeration chief of the cold chain project that the hospital trust jointly sponsors with the Development Board?", ["Bhatt"], [36, 12],
     [(36, "joint sponsor of the merrow cold chain project"), (12, "refrigeration chief is cosmo bhatt")]),
    ("multi_hop_2", "Who leads the group-wide quality program at the company whose casting plant made the Tidewrack rotor hubs?", ["Tiernan"], [1, 33],
     [(1, "hubs were manufactured at ashgill works"), (33, "director of quality is dr. ambrose tiernan")]),
    ("multi_hop_2", "Who heads the metallurgical laboratory at the plant that cast the Quarrel Hill pump-turbine casings?", ["Greenfield"], [2, 33],
     [(33, "cast the pump-turbine casings of the quarrel hill"), (33, "head of the metallurgical laboratory at ashgill works is dr. obadiah greenfield")]),
    ("multi_hop_2", "Who heads the polymer laboratory at the plant that made the Wetmoor membrane modules?", ["Ashworth"], [17, 33],
     [(17, "made at corran mill"), (33, "head of corran mill's polymer laboratory is dr. fatima ashworth")]),
    ("multi_hop_2", "Who is the chief executive of the public body that funded the Gantry Line?", ["Sandoval"], [4, 29],
     [(29, "gantry line light rail project, for which it has committed 140 million"), (29, "chief executive is margit sandoval")]),
    ("multi_hop_2", "Who is the director of finance of the public body that owns the Hollin Fibre Backbone?", ["Thackeray"], [13, 29],
     [(29, "owns the facility itself"), (29, "hollin fibre backbone"), (29, "director of finance is ingolf thackeray")]),
    ("multi_hop_2", "Who chairs the board of directors of the group whose plant made the Gantry Line's precision oscillators?", ["Aldrich"], [4, 33],
     [(33, "precision oscillators of the gantry line"), (33, "the chair matthias aldrich")]),
    ("multi_hop_2", "Who is the chief financial officer of the group that cast the Quarrel Hill pump-turbine casings?", ["Castellanos"], [2, 33],
     [(33, "pump-turbine casings of the quarrel hill"), (33, "chief financial officer is henriette castellanos")]),
    ("multi_hop_2", "Who chairs the energy committee of the body that funded the Quarrel Hill Pumped Storage plant?", ["Mulholland"], [2, 29],
     [(29, "quarrel hill pumped storage plant with 62 million"), (29, "committee on energy is chaired by one of the regional council representatives, fergus mulholland")]),
    ("multi_hop_2", "Who is the deputy director of the project whose duplicate seed bank was lost at the Dunmarrow cold store in November 2023?", ["Hartigan"], [21, 7],
     [(21, "duplicate seed bank"), (7, "her deputy is olumide hartigan")]),
    ("multi_hop_2", "Who is the chief engineer of the tidal array whose hub seals failed in February 2023?", ["Okonkwo"], [15, 1],
     [(15, "hub seals"), (1, "chief engineer is lazar okonkwo")]),
    ("multi_hop_2", "Who is the chief engineer of the pumped storage plant that tripped off the grid in August 2022?", ["Boateng"], [16, 2],
     [(16, "quarrel hill pumped storage plant tripped off the regional grid"), (2, "chief engineer is henrik boateng")]),
    ("multi_hop_2", "Who is the hydraulics chief of the barrier that closed all its gates without a real surge in January 2024?", ["Wanjiru"], [25, 11],
     [(25, "closed all six of its gates"), (11, "hydraulics chief is pilar wanjiru")]),
    ("multi_hop_2", "Who is the head of marine operations of the wind farm whose export cable fault disconnected 48 turbines?", ["Nilsen"], [24, 10],
     [(24, "disconnected all 48 turbines"), (10, "head of marine operations is gustav nilsen")]),
    ("multi_hop_2", "Who is the network chief of the fibre network whose trunk cable a dredging barge severed in June 2023?", ["Valera"], [27, 13],
     [(27, "dredging barge"), (13, "network chief is ngozi valera")]),
    ("multi_hop_2", "Who is the deputy director of the station whose director wrote the report on the lightning-damaged antenna?", ["Baek"], [19, 5],
     [(19, "written by the station's director, dr. anneliese prowse"), (5, "her deputy is jun-seo baek")]),
    # ---------------------------------------------------------------- multi_hop_3
    ("multi_hop_3", "Who is the director of finance of the body that funded the water reuse project led by a former researcher of the Marrow Institute?", ["Thackeray"], [44, 3, 29],
     [(44, "worked as a researcher at the institute"), (29, "wetmoor water reuse project with 38 million"), (29, "director of finance is ingolf thackeray")]),
    ("multi_hop_3", "Who is the chief executive of the public body that funded the light rail project whose director once worked for the freight carrier at Dunmarrow Port?", ["Sandoval"], [39, 4, 29],
     [(39, "strand & keel shipping"), (29, "gantry line light rail project, for which it has committed 140 million"), (29, "chief executive is margit sandoval")]),
    ("multi_hop_3", "Who is the head of the technology office at the body that funded the timing station, whose director once worked in that body's rail office?", ["Voss"], [40, 5, 29],
     [(40, "development board's rail office"), (29, "lodestar timing station with 21 million"), (29, "head of the technology office")]),
    ("multi_hop_3", "Who chairs the group that made the antenna bearings for the timing station directed by a former railway signal engineer?", ["Aldrich"], [40, 5, 33],
     [(40, "signal engineer"), (33, "azimuth bearings and mounting frames for the antennas of the lodestar"), (33, "the chair matthias aldrich")]),
    ("multi_hop_3", "Who heads the metallurgical laboratory at the plant that made the pivot blocks of the barrier whose false closure blocked Dunmarrow Port?", ["Greenfield"], [25, 11, 33],
     [(25, "closed all six of its gates"), (33, "pivot blocks for the six gates of the beacon hill"), (33, "dr. obadiah greenfield")]),
    ("multi_hop_3", "Who is the deputy director of the wind farm whose export cable fault caused the brownout that cost Isotope Works a production run?", ["Valdez"], [22, 24, 10],
     [(22, "export cable of the ferrous bay offshore wind farm"), (24, "disconnected all 48 turbines"), (10, "deputy is saoirse valdez")]),
    ("multi_hop_3", "Who is the deputy director of the cold chain project whose Dunmarrow depot froze out the duplicate seed bank in the brownout?", ["Rojas"], [21, 26, 12],
     [(21, "dunmarrow depot of the merrow cold chain"), (26, "duplicate seed bank of the seedvault north"), (12, "deputy is annika rojas")]),
    ("multi_hop_3", "Who is the field manager of the project whose duplicate seed bank was damaged in the brownout caused by the Ferrous Bay cable fault?", ["Nakamura"], [21, 24, 7],
     [(21, "brownout"), (24, "export cable of the ferrous bay offshore wind farm"), (7, "field manager is beatrix nakamura")]),
    ("multi_hop_3", "Who is the clinical chief of the telehealth network whose mountain clinics were cut off by the dredging barge?", ["Tan"], [27, 28, 14],
     [(27, "dredging barge"), (28, "thirteen mountain clinics"), (14, "clinical chief is dr. mireille tan")]),
    ("multi_hop_3", "Who is the signals chief of the light rail project that suffered fail-safe stops when the timing signal from Cairn Head degraded?", ["Lindgren"], [18, 5, 4],
     [(18, "fail-safe stops"), (18, "timing signal supplied by the lodestar timing station at cairn head"), (4, "signals chief is pavel lindgren")]),
    ("multi_hop_3", "Who is the control room chief of the port project whose cranes collided after the timing station's signal degraded?", ["Marsh"], [20, 19, 6],
     [(20, "timing drift originating at the lodestar timing station"), (19, "collision between two automated cranes at dunmarrow port"), (6, "control room chief is stellan marsh")]),
    ("multi_hop_3", "What was the designation of the membrane batch that caused the 2022 fouling at the project directed by the former Marrow Institute researcher?", ["C-31"], [44, 3, 17],
     [(44, "worked as a researcher at the institute"), (3, "dr. naledi mokoena has directed wetmoor works"), (17, "designated c-31")]),
    ("multi_hop_3", "What was the designation of the hub casting batch that failed in the array led by the engineer who did his doctorate under the Marrow Institute's director?", ["AH-14"], [37, 1, 15],
     [(37, "his doctoral supervisor was professor ottoline brandvik"), (1, "dr. bjorn halvorsen has led the tidewrack project"), (15, "designated ah-14")]),
    ("multi_hop_3", "Who leads the audit team that investigated the failure at the water project directed by the former Marrow Institute researcher?", ["Idowu"], [44, 3, 17],
     [(44, "worked as a researcher at the institute"), (17, "an independent investigation was led by dr. kasper idowu"), (3, "wetmoor works")]),
    # ---------------------------------------------------------------- multi_hop_4
    ("multi_hop_4", "Who is the deputy director of the project led by the former analyst of the bank that financed the wind farm whose cable fault caused the November 2023 brownout?", ["Odhiambo"], [24, 32, 41, 6],
     [(24, "export cable of the ferrous bay offshore wind farm"), (32, "120 million crowns to the ferrous bay offshore wind farm"), (41, "analyst in its infrastructure and project finance division"), (6, "his deputy is wilhelmina odhiambo")]),
    ("multi_hop_4", "Who is the director of finance of the body that funded the timing station used by the automated port where the Ferrous Bay wind farm is based?", ["Thackeray"], [10, 6, 5, 29],
     [(10, "operations base at dunmarrow port"), (5, "dunmarrow port automation project uses it"), (29, "lodestar timing station with 21 million"), (29, "director of finance is ingolf thackeray")]),
    ("multi_hop_4", "Who is the signals chief of the light rail project directed by the former network planner at the freight company that carries the recovered products of the recycling works led by a former Ashgill researcher?", ["Lindgren"], [43, 9, 31, 39, 4],
     [(43, "post in the metallurgical laboratory at ashgill works"), (31, "recovered cobalt, nickel, and lithium products of the cinder lake"), (39, "network planner"), (4, "signals chief is pavel lindgren")]),
    ("multi_hop_4", "Who is the clinical chief of the telehealth network hosted by the trust that owns the facility which lost a production run in the brownout caused by the Ferrous Bay cable fault?", ["Tan"], [24, 22, 36, 14],
     [(22, "export cable of the ferrous bay offshore wind farm"), (36, "it owns the isotope works facility"), (36, "hosts the larkspur telehealth network"), (14, "clinical chief is dr. mireille tan")]),
    ("multi_hop_4", "Who is the chief financial officer of the group whose plant made the pivot blocks of the flood barrier that protects the port hosting the Ferrous Bay wind farm's operations base?", ["Castellanos"], [10, 11, 33],
     [(10, "operations base at dunmarrow port"), (11, "protects some 14,000 homes and the dunmarrow port"), (33, "pivot blocks for the six gates of the beacon hill"), (33, "chief financial officer is henriette castellanos")]),
]

# Aggregation: gold list = every item below must be in the answer. Evidence re-verified against the text.
AGG = [
    ("Which projects did the Calder Reach Development Board fund?", ["Gantry", "Beacon", "Quarrel", "Hollin", "Wetmoor", "Lodestar", "Larkspur"], [29],
     [(29, "gantry line light rail project, for which it has committed 140 million"), (29, "beacon hill flood barrier with 97 million"), (29, "quarrel hill pumped storage plant with 62 million"),
      (29, "hollin fibre backbone with 55 million"), (29, "wetmoor water reuse project with 38 million"), (29, "lodestar timing station with 21 million"), (29, "larkspur telehealth network with 18 million")]),
    ("Which projects did the Harbourgate Mutual Bank give its three largest infrastructure loans to?", ["Tidewrack", "Dunmarrow", "Ferrous"], [32],
     [(32, "44 million crowns to the tidewrack tidal array"), (32, "58 million crowns to the dunmarrow port automation"), (32, "120 million crowns to the ferrous bay offshore wind farm")]),
    ("Which projects received large castings or machined components from Ashgill Works?", ["Tidewrack", "Quarrel", "Beacon", "Gantry", "Lodestar"], [33],
     [(33, "rotor hubs of all 24 turbines of the tidewrack"), (33, "pump-turbine casings of the quarrel hill"), (33, "pivot blocks for the six gates of the beacon hill"),
      (33, "oscillators of the gantry line"), (33, "antennas of the lodestar timing station")]),
    ("Which two projects depend on products made at Corran Mill?", ["Wetmoor", "Ferrous"], [33],
     [(33, "membrane modules used by the wetmoor water reuse project"), (33, "144 blades of the ferrous bay offshore wind farm")]),
    ("Which projects and facilities were disrupted by the November 2023 brownout?", ["Isotope", "Merrow", "Wetmoor"], [22],
     [(22, "lost a full production run of medical isotopes"), (22, "disrupted the merrow cold chain's dunmarrow depot and reduced output at the wetmoor water reuse project")]),
    ("Which three projects suffered incidents in the week of 14 August 2022 that were linked to the damage at the Lodestar Timing Station?", ["Gantry", "Dunmarrow", "Quarrel"], [19],
     [(19, "fail-safe stops on the gantry line light rail network, a collision between two automated cranes at dunmarrow port, and the tripping of the quarrel hill pumped storage plant")]),
    ("Which project directors previously worked at Strand & Keel Shipping, the Harbourgate Mutual Bank, the Lumen Grid Company or Ashgill Works?", ["Kowalczyk", "Adebayo", "Valkanova", "Duarte"], [39, 41, 38, 43],
     [(39, "to join strand & keel shipping"), (41, "joined the harbourgate mutual bank"), (38, "recruited by the lumen grid company"), (43, "metallurgical laboratory at ashgill works")]),
    ("Which two project directors did their doctoral studies under Professor Ottoline Brandvik?", ["Halvorsen", "Raghunathan"], [37, 42],
     [(37, "his doctoral supervisor was professor ottoline brandvik"), (42, "her supervisor was professor ottoline brandvik")]),
    ("Which projects does the St. Idris Hospital Trust own, host or jointly sponsor?", ["Isotope", "Larkspur", "Merrow"], [36],
     [(36, "it owns the isotope works"), (36, "it hosts the larkspur telehealth network"), (36, "joint sponsor of the merrow cold chain")]),
    ("Which organizations and projects use the timing signal of the Lodestar Timing Station?", ["Gantry", "Dunmarrow", "Lumen", "Beacon"], [5],
     [(5, "the gantry line light rail project uses it"), (5, "the dunmarrow port automation project uses it"), (5, "the lumen grid company uses it"), (5, "beacon hill flood barrier became a fourth customer")]),
]


def check_evidence(label, ev):
    for doc, text in ev:
        if text.lower() not in DOCS[doc]:
            raise SystemExit(f"EVIDENCE MISSING in doc {doc:02d} for {label!r}: {text!r}")


def main():
    v5 = {q["id"]: q for q in json.loads((ROOT / "benchmark_v5" / "gold_qa.json").read_text(encoding="utf-8"))}
    out = []
    for q in v5.values():
        if q["category"] in ("real_single_hop", "single_hop"):
            out.append(q)
    for i in KEEP_V5:
        out.append(v5[i])

    next_id = 201
    for cat, question, kws, srcs, ev in NEW:
        check_evidence(question, ev)
        out.append({"id": next_id, "category": cat, "confidence": "high", "question": question, "expected_keywords": kws,
                    "forbidden_phrases": [], "source_docs": srcs, "evidence": [[d, t] for d, t in ev]})
        next_id += 1
    for question, kws, srcs, ev in AGG:
        check_evidence(question, ev)
        out.append({"id": next_id, "category": "aggregation", "confidence": "high", "question": question, "expected_keywords": kws,
                    "forbidden_phrases": [], "source_docs": srcs, "evidence": [[d, t] for d, t in ev]})
        next_id += 1

    # every expected keyword of a new/kept question must exist in at least one of its source documents
    for q in out:
        if q["category"] == "real_single_hop":
            continue
        for kw in q["expected_keywords"]:
            if not any(kw.lower() in DOCS[d] for d in q["source_docs"] if isinstance(d, int)):
                raise SystemExit(f"keyword {kw!r} not in source docs of Q{q['id']}")

    tails = Counter(q["expected_keywords"][0] for q in out if q["category"].startswith("multi_hop"))
    over = {k: v for k, v in tails.items() if v > MAX_TAIL}
    if over:
        raise SystemExit(f"answers used by more than {MAX_TAIL} multi-hop questions: {over}")
    ids = [q["id"] for q in out]
    assert len(ids) == len(set(ids))

    (HERE / "gold_qa.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    cats = Counter(q["category"] for q in out)
    print(f"wrote {len(out)} questions: {dict(cats)}")
    print("most common multi-hop answers:", tails.most_common(6))


if __name__ == "__main__":
    main()
