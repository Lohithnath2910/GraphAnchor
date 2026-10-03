#!/usr/bin/env python3
"""
Synthetic corpus generator for GraphAnchor evaluation.

Generates ~100 documents (1000+ words each) describing a fictional
company's projects, people and sectors, with deliberately designed
relational chains (1-hop, 2-hop, 3-hop) so retrieval quality can be
measured against KNOWN ground truth instead of guessed from a
qualitative transcript.

Usage:
    python generate_corpus.py --out ./corpus --gold ./gold_qa_v2.json --seed 42
"""
import argparse
import json
import random

PEOPLE = [
    "Meera Kapadia", "Daniel Osei", "Priya Ramaswamy", "Tomas Vargas",
    "Aiko Fujimori", "Bram Van Dijk", "Nadia Farouk", "Caleb Nwosu",
    "Ingrid Solberg", "Rohan Deshmukh", "Lucia Ferreira", "Omar Haddad",
    "Sienna Walsh", "Viktor Petrov", "Amara Johnson", "Hiro Tanaka",
    "Elena Vasquez", "Malik Thompson", "Freya Lindqvist", "Arjun Mehta",
    "Grace Okonkwo", "Felix Brandt", "Noor Siddiqui", "Hana Kobayashi",
]

PROJECTS = [
    "Aurora", "Basalt", "Cinderpath", "Dovetail", "Ember", "Fjord",
    "Glasswing", "Halcyon", "Indigo Run", "Juniper", "Kestrel",
    "Lumen", "Monarch", "Nightshade", "Opaline", "Pinnacle",
]

SECTORS = [
    "Propulsion Systems", "Cybersecurity", "Materials Science",
    "Logistics & Supply Chain", "Avionics", "Power Electronics",
    "Field Operations", "Data Platforms",
]

FACILITIES = [
    "Northfield Campus", "Bay Ridge Lab", "Meridian Works",
    "Crestline Annex", "Harborview Site", "Underhill Facility",
    "Vantage Park", "Redstone Yard",
]

random.seed(0)  # overridden in main()

# Combinatorial filler: slot-filled templates so sentences are not verbatim
# duplicates across the 100 documents (a fixed sentence bank repeated as-is
# produces near-identical chunks that hurt vector-similarity testing and
# read as obviously templated).
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
TOPICS = [
    ("budget", [
        "budget utilization held at {pct}% of the approved quarterly envelope",
        "spend tracked {pct}% against the baseline forecast for the period",
        "a variance of {pct}% against plan was flagged for finance review in {month}",
        "the {month} spend report showed utilization at {pct}% of plan",
    ]),
    ("schedule", [
        "the milestone originally targeted for {month} was re-confirmed with a {num}-day buffer",
        "a {num}-day slip against the baseline schedule was logged and accepted",
        "the next checkpoint review is scheduled for {month}, {num} weeks out",
        "{num} schedule dependencies were reconfirmed ahead of the {month} checkpoint",
    ]),
    ("quality", [
        "{num} items from the test plan were closed out with no deviations",
        "a quality review surfaced {num} minor findings, all since resolved",
        "defect density remained within {pct}% of the accepted threshold for the period",
        "{num} test cases were re-run in {month} following a minor spec update",
    ]),
    ("risk", [
        "{num} open risk items remain on the register, none rated severe",
        "a new risk was logged in {month} and assigned an owner for follow-up",
        "the risk register was reviewed in {month} with {num} items closed",
        "exposure was reassessed at {pct}% of the prior estimate after mitigation",
    ]),
    ("staffing", [
        "{num} additional contractor hours were allocated for the {month} period",
        "no staffing changes were proposed for the cycle beginning {month}",
        "cross-training for {num} team members was completed this period",
        "{num} onboarding sessions were held ahead of the {month} ramp-up",
    ]),
    ("stakeholder", [
        "stakeholders from adjacent sectors were briefed during the {month} sync",
        "a summary was circulated to leadership ahead of the {month} review",
        "{num} items of feedback from the prior cycle were incorporated into {month} planning",
        "a {num}-point summary was shared with sector leadership in {month}",
    ]),
    ("logistics", [
        "vendor deliveries for supporting components arrived within {num} days of the {month} schedule",
        "site logistics for the {month} visit were confirmed with {num} open items closed",
        "equipment calibration was completed ahead of the {month} checkpoint",
        "{num} shipments were received and reconciled against the {month} manifest",
    ]),
    ("compliance", [
        "an internal audit covering {num} controls was completed in {month} without material findings",
        "documentation was updated in {month} to reflect the latest approved revision",
        "a compliance checklist covering {num} controls was signed off in {month}",
        "{pct}% of required controls were re-validated during the {month} audit window",
    ]),
]

def _fill(template, rng):
    return template.format(
        pct=rng.choice([88, 91, 92, 94, 96, 97, 98, 101, 103, 105, 107]),
        num=rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 11, 12]),
        month=rng.choice(MONTHS),
    )

def pad_paragraph(rng, n_sentences=6):
    chosen_topics = rng.sample(TOPICS, k=min(n_sentences, len(TOPICS)))
    sentences = []
    for _, templates in chosen_topics:
        t = rng.choice(templates)
        filled = _fill(t, rng)
        sentences.append(filled[0].upper() + filled[1:] + ".")
    return " ".join(sentences)


def build_world(rng):
    """Assigns people, sectors, facilities and dependency edges to projects."""
    people = list(PEOPLE)
    rng.shuffle(people)
    projects = list(PROJECTS)

    world = {"projects": {}, "people": {}, "sector_managers": {}}

    # Assign a manager to each sector first (these are 1-hop facts)
    mgr_pool = people[:len(SECTORS)]
    for sector, mgr in zip(SECTORS, mgr_pool):
        world["sector_managers"][sector] = mgr

    remaining_people = people[len(SECTORS):]
    person_cycle = remaining_people + mgr_pool  # managers can also lead projects

    for i, proj in enumerate(projects):
        lead = person_cycle[i % len(person_cycle)]
        sector = SECTORS[i % len(SECTORS)]
        facility = FACILITIES[i % len(FACILITIES)]
        team = rng.sample([p for p in person_cycle if p != lead], k=3)
        world["projects"][proj] = {
            "lead": lead,
            "sector": sector,
            "facility": facility,
            "team": team,
            "depends_on": None,  # filled below
        }

    # Build a dependency chain: each project may depend on one earlier project
    for i, proj in enumerate(projects):
        if i == 0:
            continue
        if rng.random() < 0.7:
            dep = projects[rng.randrange(0, i)]
            world["projects"][proj]["depends_on"] = dep

    # Reverse index: person -> projects they lead/work on
    for proj, info in world["projects"].items():
        for person in [info["lead"]] + info["team"]:
            world["people"].setdefault(person, {"leads": [], "member_of": [], "sector": None})
        world["people"][info["lead"]]["leads"].append(proj)
        for member in info["team"]:
            world["people"][member]["member_of"].append(proj)

    for sector, mgr in world["sector_managers"].items():
        world["people"].setdefault(mgr, {"leads": [], "member_of": [], "sector": None})
        world["people"][mgr]["sector"] = sector

    return world


DOC_TYPES = [
    "status_report", "meeting_minutes", "risk_assessment",
    "incident_report", "personnel_memo", "technical_summary",
]

def doc_title(doc_type, proj):
    titles = {
        "status_report": f"Project {proj} - Monthly Status Report",
        "meeting_minutes": f"Project {proj} - Steering Committee Meeting Minutes",
        "risk_assessment": f"Project {proj} - Risk Assessment Memo",
        "incident_report": f"Project {proj} - Field Incident Report",
        "personnel_memo": f"Project {proj} - Personnel & Staffing Memo",
        "technical_summary": f"Project {proj} - Technical Summary",
    }
    return titles[doc_type]


def render_doc(rng, doc_type, proj, info):
    lead = info["lead"]
    sector = info["sector"]
    facility = info["facility"]
    team = info["team"]
    dep = info["depends_on"]

    lines = []
    lines.append(doc_title(doc_type, proj))
    lines.append("")

    intro = (
        f"Project {proj} is managed under the {sector} sector and is based out of "
        f"{facility}. The project is led by {lead}, who is responsible for coordinating "
        f"day-to-day execution and reporting progress to sector leadership."
    )
    lines.append(intro)
    lines.append("")

    team_para = (
        f"The core team supporting {lead} on Project {proj} includes "
        f"{', '.join(team)}. Each team member holds responsibility for a distinct "
        f"work stream, and {lead} convenes the group on a recurring basis to review "
        f"open items and dependencies."
    )
    lines.append(team_para)
    lines.append("")

    if dep:
        dep_para = (
            f"Project {proj} has a direct dependency on Project {dep}: deliverables from "
            f"Project {dep} feed into the {proj} workstream, and any schedule slip on "
            f"Project {dep} is tracked as a risk item for {proj}. {lead} maintains a "
            f"standing coordination channel with the Project {dep} team to monitor this "
            f"dependency."
        )
        lines.append(dep_para)
        lines.append("")

    # type-specific body
    if doc_type == "status_report":
        body = (
            f"This month, {lead} reported steady progress against the Project {proj} "
            f"milestone plan. {pad_paragraph(rng, 5)} The sector manager for "
            f"{sector} was briefed on overall trajectory and raised no immediate concerns."
        )
    elif doc_type == "meeting_minutes":
        body = (
            f"Attendees: {lead} (chair), {', '.join(team)}. {pad_paragraph(rng, 6)} "
            f"Action items were assigned and will be reviewed at the next session chaired "
            f"by {lead}."
        )
    elif doc_type == "risk_assessment":
        body = (
            f"{lead} and the Project {proj} team identified a small number of open risks "
            f"this period. {pad_paragraph(rng, 5)} The most significant dependency risk "
            + (f"remains tied to Project {dep}." if dep else "was assessed as low given the project's limited external dependencies.")
        )
    elif doc_type == "incident_report":
        body = (
            f"A minor operational incident was logged at {facility} during routine Project "
            f"{proj} activities. {pad_paragraph(rng, 5)} {lead} confirmed corrective actions "
            f"were completed and no further escalation was required."
        )
    elif doc_type == "personnel_memo":
        body = (
            f"This memo documents current staffing for Project {proj} under {lead}. "
            f"{pad_paragraph(rng, 4)} No staffing changes are proposed for the upcoming "
            f"period; {lead} will continue to lead resourcing decisions for the team."
        )
    else:  # technical_summary
        body = (
            f"The Project {proj} technical workstream, overseen by {lead}, made incremental "
            f"progress this period. {pad_paragraph(rng, 5)} Findings were documented and "
            f"shared with the {sector} sector for awareness."
        )
    lines.append(body)
    lines.append("")

    # pad to reach ~1000+ words with additional filler paragraphs
    while sum(len(l.split()) for l in lines) < 1010:
        lines.append(pad_paragraph(rng, 6))
        lines.append("")

    closing = (
        f"For questions regarding Project {proj}, contact {lead} directly or escalate "
        f"through the {sector} sector management chain."
    )
    lines.append(closing)

    return "\n\n".join(lines)


def render_sector_overview(rng, sector, world):
    mgr = world["sector_managers"][sector]
    projects_in_sector = [p for p, info in world["projects"].items() if info["sector"] == sector]
    lines = [f"{sector} - Sector Overview", ""]
    lines.append(
        f"The {sector} sector is managed by {mgr}, who oversees all active projects "
        f"within the sector and reports cross-project status to executive leadership."
    )
    if projects_in_sector:
        lines.append(
            f"Active projects currently within {sector} include: {', '.join(projects_in_sector)}."
        )
    while sum(len(l.split()) for l in lines) < 1010:
        lines.append(pad_paragraph(rng, 6))
    lines.append(
        f"{mgr} holds final sign-off authority on resourcing changes across all projects "
        f"listed above."
    )
    return "\n\n".join(lines)


def build_gold_questions(world):
    questions = []
    qid = 1

    # 1-hop: who leads a project
    for proj, info in list(world["projects"].items())[:10]:
        questions.append({
            "id": qid, "category": "single_hop", "hop_distance": 1, "confidence": "exact",
            "question": f"Who leads Project {proj}?",
            "expected_keywords": [info["lead"]],
            "forbidden_phrases": [],
        })
        qid += 1

    # 1-hop: which sector manages a project
    for proj, info in list(world["projects"].items())[10:16]:
        questions.append({
            "id": qid, "category": "single_hop", "hop_distance": 1, "confidence": "exact",
            "question": f"Which sector is Project {proj} managed under?",
            "expected_keywords": [info["sector"]],
            "forbidden_phrases": [],
        })
        qid += 1

    # 2-hop: project -> depends_on -> project (should be answerable, hops<=2)
    for proj, info in world["projects"].items():
        if info["depends_on"]:
            questions.append({
                "id": qid, "category": "multi_hop", "hop_distance": 2, "confidence": "exact",
                "question": f"What project does Project {proj} depend on?",
                "expected_keywords": [info["depends_on"]],
                "forbidden_phrases": [],
            })
            qid += 1

    # 2-hop: person -> leads -> project -> sector (who's sector is a person's project in)
    count = 0
    for proj, info in world["projects"].items():
        if count >= 8:
            break
        questions.append({
            "id": qid, "category": "multi_hop", "hop_distance": 2, "confidence": "exact",
            "question": f"{info['lead']} leads Project {proj}. Which sector does that project belong to?",
            "expected_keywords": [info["sector"]],
            "forbidden_phrases": [],
        })
        qid += 1
        count += 1

    # 3-hop (deliberately beyond the 2-hop cap): person -> project -> depends_on -> project's lead
    count = 0
    for proj, info in world["projects"].items():
        if count >= 6:
            break
        dep = info["depends_on"]
        if dep:
            dep_lead = world["projects"][dep]["lead"]
            questions.append({
                "id": qid, "category": "multi_hop_3plus", "hop_distance": 3, "confidence": "exact",
                "question": (
                    f"Project {proj} depends on another project. Who leads the project "
                    f"that Project {proj} depends on?"
                ),
                "expected_keywords": [dep_lead],
                "forbidden_phrases": [],
                "note": "3 relational hops from the asking project's lead; beyond the current 2-hop graph traversal cap, included to test/demonstrate that limitation.",
            })
            qid += 1
            count += 1

    return questions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./corpus")
    ap.add_argument("--gold", default="./gold_qa_v2.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    world = build_world(rng)

    doc_count = 0
    for proj, info in world["projects"].items():
        for doc_type in DOC_TYPES:
            text = render_doc(rng, doc_type, proj, info)
            fname = f"{doc_count+1:03d}_{proj.lower().replace(' ', '_')}_{doc_type}.txt"
            with open(os.path.join(args.out, fname), "w", encoding="utf-8") as f:
                f.write(text)
            doc_count += 1

    for sector in SECTORS[:100 - doc_count]:
        text = render_sector_overview(rng, sector, world)
        fname = f"{doc_count+1:03d}_sector_{sector.lower().replace(' ', '_').replace('&','and')}.txt"
        with open(os.path.join(args.out, fname), "w", encoding="utf-8") as f:
            f.write(text)
        doc_count += 1

    gold = build_gold_questions(world)
    with open(args.gold, "w", encoding="utf-8") as f:
        json.dump(gold, f, indent=2)

    print(f"Generated {doc_count} documents in {args.out}")
    print(f"Generated {len(gold)} gold QA pairs in {args.gold}")
    word_counts = []
    for fname in os.listdir(args.out):
        with open(os.path.join(args.out, fname), encoding="utf-8") as f:
            word_counts.append(len(f.read().split()))
    print(f"Word count range: {min(word_counts)}-{max(word_counts)}, avg {sum(word_counts)//len(word_counts)}")


if __name__ == "__main__":
    main()
