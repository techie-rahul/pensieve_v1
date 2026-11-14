"""DEVELOPMENT/DEMO-ONLY dataset seed mechanism for Pensieve.

Seeds a realistic historical dataset of ~22 published journal entries spanning the previous 30 days
for longitudinal pattern analysis and reflection demonstrations.

Usage:
    Seed demo user:
        python -m scripts.seed_demo
        or: python scripts/seed_demo.py

    Reset/remove demo user:
        python -m scripts.seed_demo --reset
        or: python scripts/seed_demo.py --reset

    Custom options:
        python -m scripts.seed_demo --email demo@pensieve.app --password DemoUser123!
"""

import argparse
from datetime import datetime, timedelta, timezone
import logging
import sys
from typing import List, Tuple

from app.auth.security import hash_password
from app.database import SessionLocal, init_db
from app.models.entry import EntryAnalysis, JournalEntry
from app.models.reflection import Reflection
from app.models.user import User
from app.services.ml_service import MLService

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pensieve.seed_demo")

DEFAULT_DEMO_EMAIL = "demo@pensieve.app"
DEFAULT_DEMO_PASSWORD = "DemoUser123!"
DEFAULT_DEMO_NAME = "Eleanor Vance"

# 22 Curated, realistic historical entries spanning 30 days with longitudinal progression
# (days_ago, hour_utc, minute_utc, title, mood, tags, content)
HISTORICAL_JOURNAL_ENTRIES: List[Tuple[int, int, int, str, str, List[str], str]] = [
    # Phase A: Days 29 to 20 ago — High pressure, cognitive load, anxiety, overthinking, sleep disruption
    (
        29, 8, 30,
        "Project Kickoff and Early Apprehension",
        "Anxious",
        ["work", "project", "deadlines"],
        "We had the big kickoff meeting this morning for the Q3 systems migration. The scope is noticeably broader than what we initially discussed, and the delivery milestones feel uncomfortably compressed. I noticed my chest tightening as the product lead walked through the roadmap. I spent the rest of the morning second-guessing whether our microservices architecture can actually support this load without failing. I really want to do a good job, but the sheer volume of open questions is making it hard to settle into a rhythm."
    ),
    (
        28, 22, 15,
        "Late Night Code Spiraling",
        "Overwhelmed",
        ["work", "overthinking", "code"],
        "Still sitting at my desk well past 10 PM. I got completely derailed by a subtle concurrency bug in the event ingestion worker. Instead of stepping back, I found myself going down rabbit holes, reading obscure thread-safety docs, and convincing myself that our whole storage approach is flawed. My mind is buzzing with worst-case scenarios about what happens if we slip on the sprint demo. Need to close the laptop and try to sleep, but turning my brain off feels impossible tonight."
    ),
    (
        25, 14, 0,
        "Frustrations with Shifting Requirements",
        "Frustrated",
        ["work", "communication"],
        "Right in the middle of our sprint review, the client requirements shifted again. Three days of refactoring our data models are essentially obsolete now. It took everything in me not to show my irritation during the video call. Why is it so difficult for stakeholders to commit to a direction before we start building? I feel like I am running on a treadmill that keeps speeding up while getting nowhere."
    ),
    (
        24, 9, 10,
        "Foggy Morning and Low Energy",
        "Exhausted",
        ["health", "sleep", "fatigue"],
        "Woke up feeling like I barely slept. Stared at my alarm for ten minutes before dragging myself out of bed. On my second cup of coffee and the screen glare is giving me a dull headache. Every single Jira ticket looks like a massive mountain right now. Just trying to put one foot in front of the other and get through the morning standup without zoning out."
    ),
    (
        22, 19, 45,
        "The Comparison Trap",
        "Insecure",
        ["self-doubt", "work", "career"],
        "Spent almost four hours today debugging an intermittent network timeout that David solved in twenty minutes. I hate how quickly that familiar feeling of impostor syndrome sneaks in. I look around at the senior engineers on the team and everyone seems so effortlessly capable, while I feel like I'm constantly treading water just to keep up. I know rationally that comparing my internal struggle to everyone else's highlight reel is unfair, but emotionally it still stings."
    ),
    (
        21, 11, 30,
        "Unable to Unwind on the Weekend",
        "Restless",
        ["personal", "boundaries", "weekend"],
        "It's Saturday morning, the sun is out, and yet my mind is stuck in Monday. I keep reflexively checking my work phone to see if any urgent alerts popped up on PagerDuty. I tried sitting down to read a book, but after three pages I realized I hadn't absorbed a single sentence because I was mentally composing an email to my engineering manager. I desperately need to figure out better boundaries between work and my personal life."
    ),
    (
        20, 21, 0,
        "Sunday Evening Unease",
        "Anxious",
        ["reflection", "work", "habits"],
        "That familiar Sunday evening knot in my stomach is back. Looking at my calendar for tomorrow, it is an absolute wall of back-to-back meetings from 9 AM to 4 PM. When am I actually supposed to write code and do deep work? I feel like I am constantly reacting to everyone else's emergencies instead of having any agency over my own day. Something in this routine has to change."
    ),
    # Phase B: Days 19 to 10 ago — Realization, initial attempts at structure, boundaries, mindfulness
    (
        18, 8, 15,
        "A Small Pause on the Walk to Work",
        "Reflective",
        ["awareness", "nature", "morning"],
        "While walking through the neighborhood park this morning, I forced myself to leave my headphones in my pocket. I noticed the cherry trees along the pathway are already beginning to bud with tiny pink blossoms. It struck me that I have walked past these exact trees every single morning for two weeks without looking up once. I've been living entirely in my head, disconnected from the physical world. It was a gentle reminder that life is happening right now, outside of my screen."
    ),
    (
        17, 13, 30,
        "Experimenting with Focus Blocks",
        "Determined",
        ["productivity", "habits", "work"],
        "Decided to try something different today: 25-minute uninterrupted focus blocks with Slack completely shut down. The first twenty minutes felt almost uncomfortable—my hands kept wanting to alt-tab and check for new notifications. But once I pushed through that initial friction, I got into a genuine flow state and wrote the entire validation layer without stopping. It felt remarkably good to finish something clean."
    ),
    (
        16, 20, 0,
        "Navigating Conflict with Marcus",
        "Relieved",
        ["relationships", "communication", "work"],
        "Had a pretty tense disagreement with Marcus during the API review. In the past, I would have either shut down or argued defensively. Instead, I took a deep breath and said, 'Hey, let's hop on a 10-minute huddle and draw this on a whiteboard together.' We realized we were actually agreeing on 90% of the architecture and just using different terminology. We ended up with a much cleaner schema and no lingering tension. A small victory in healthy communication."
    ),
    (
        14, 7, 30,
        "Morning Run and Clear Air",
        "Energized",
        ["fitness", "health", "morning"],
        "Laced up my running shoes and got out the door at 6:30 AM before the heat set in. Ran three miles around the reservoir. My legs were heavy for the first mile, but by the third mile my head felt clearer than it has in weeks. Physical fatigue is so much cleaner and healthier than mental exhaustion. Came into standup with real energy rather than relying on caffeine to wake up."
    ),
    (
        13, 22, 30,
        "Wobbles and Self-Compassion",
        "Discouraged",
        ["habits", "self-compassion", "reflection"],
        "Today wasn't great. Slipped right back into old habits: skipped lunch, drank four cups of coffee, and stared at my monitor until my eyes burned. Felt a wave of disappointment this evening, like all my efforts to build better routines were useless. But I am trying to practice some self-compassion. One disorganized day doesn't erase the progress of the last week. Tomorrow is simply a fresh canvas."
    ),
    (
        12, 16, 45,
        "The Power of a Polite No",
        "Empowered",
        ["boundaries", "work", "focus"],
        "Politely declined an optional committee meeting this afternoon to protect a two-hour block for database schema migrations. I felt a pang of guilt when hitting 'Decline', wondering if people would think I'm not a team player. But nobody questioned it, and I finished the entire migration script without errors before 5 PM. Setting boundaries feels uncomfortable at first, but the payoff in peace of mind is undeniable."
    ),
    (
        11, 12, 0,
        "Quiet Lunch on the Bench",
        "Peaceful",
        ["mindfulness", "rest", "nature"],
        "Left my phone on my desk and took a sandwich to the park bench across the street. Spent twenty minutes just eating slowly and watching pigeons and people walking their dogs. No podcasts, no work chats, no doomscrolling. Returning to the office, the afternoon problems didn't feel quite so apocalyptic. Silence is underrated."
    ),
    (
        10, 19, 15,
        "End of Sprint Retrospective",
        "Content",
        ["progress", "reflection", "work"],
        "Sprint ended today. We didn't hit 100% of our stretch goals, but we delivered the core search pipeline and test coverage is up to 88%. Looking back at where my head was two weeks ago compared to now, the panic has subsided into manageable focus. I'm learning that not every fire requires my immediate panic."
    ),
    # Phase C: Days 9 to 0 ago — Greater clarity, prioritization, emotional grounding, self-compassion, resilience
    (
        8, 9, 0,
        "Three Morning Priorities",
        "Focused",
        ["routine", "focus", "mindset"],
        "Started the day by writing down only three non-negotiable tasks on an index card before opening Slack or email. When the inevitable wave of 'urgent' side requests landed in my inbox at 10 AM, I used the card as an anchor. It's amazing how much clearer decision-making becomes when you've already decided what matters before the chaos begins."
    ),
    (
        7, 18, 30,
        "Staying Grounded During an Outage",
        "Calm",
        ["resilience", "work", "growth"],
        "Staging deployment threw a cascade of 500 errors right before end-of-day. Normally, my heart would be in my throat. Today, I took a deliberate slow breath, checked the structured logs, found the missing environment variable, and rolled back the release calmly within eight minutes. Panic never fixes a bug faster; calm systematic thinking does."
    ),
    (
        5, 10, 15,
        "Mentoring and Shared Knowledge",
        "Gratitude",
        ["mentoring", "connection", "team"],
        "Spent an hour pairing with Sarah, our new junior engineer, walking through our caching layer and distributed lock implementation. Explaining the reasoning and trade-offs out loud made me realize how much tacit knowledge I've accumulated over the past year. It felt wonderful to encourage her and demystify concepts that used to terrify me when I first started."
    ),
    (
        4, 21, 30,
        "Evening Stillness and Good Books",
        "Serene",
        ["philosophy", "gratitude", "evening"],
        "Cooked a warm pasta dinner at home and read two chapters of Marcus Aurelius's Meditations. His reminder about the dichotomy of control hit home tonight: 'You have power over your mind - not outside events. Realize this, and you will find strength.' Grateful for a quiet, peaceful evening."
    ),
    (
        2, 15, 0,
        "Architecture Review with Confidence",
        "Confident",
        ["growth", "communication", "work"],
        "Presented our proposed event-driven telemetry architecture to the staff engineers this afternoon. Received several tough, pointed questions about data consistency during network partitions. Instead of getting defensive, I genuinely appreciated the feedback and incorporated their suggestions into the design doc. I finally feel like a trusted peer in these technical discussions."
    ),
    (
        1, 11, 45,
        "Looking Back at Thirty Days",
        "Insightful",
        ["reflection", "perspective", "growth"],
        "Took some time during my coffee break to read through my journal entries from a month ago. The difference in tone is startling. Back then, every deadline felt like an existential crisis and every bug felt like personal failure. The workload is still demanding, but my internal posture has fundamentally shifted. I am learning to hold things lightly while still caring deeply about the quality of my craft."
    ),
    (
        0, 8, 30,
        "Ready for the Day Ahead",
        "Balanced",
        ["mindfulness", "mindset", "presence"],
        "Sitting with my morning tea before the week's final demos begin. Feeling centered, clear-headed, and prepared. Whatever happens in today's presentations, I know how to pace myself, lean on my team, and stay grounded in what is truly important. Stillness is not the absence of challenge; it is the presence of perspective."
    ),
]


def reset_demo_user(email: str = DEFAULT_DEMO_EMAIL) -> None:
    """Safely reset and remove the demo user and all associated entries and reflections."""
    init_db()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            print(f"[DEMO RESET] User '{email}' does not exist in local database. Nothing to clean up.")
            return

        db.delete(user)
        db.commit()
        print(f"[DEMO RESET] Successfully removed demo user '{email}' and all associated journal entries, ML analyses, and reflections.")
    finally:
        db.close()


def seed_demo_dataset(
    email: str = DEFAULT_DEMO_EMAIL,
    password: str = DEFAULT_DEMO_PASSWORD,
    name: str = DEFAULT_DEMO_NAME,
) -> None:
    """Seed the demo user with realistic longitudinal journal data and run live ML pipeline analysis."""
    print("=" * 70)
    print("PENSIEVE DEVELOPMENT/DEMO DATASET SEEDER")
    print("=" * 70)
    print(f"Target Demo Account: {email}")
    print(f"Password:           {password}")
    print(f"User Name:          {name}")
    print("-" * 70)

    init_db()
    db = SessionLocal()
    ml_service = MLService.get_instance()

    try:
        # 1. Find or create user
        user = db.query(User).filter(User.email == email).first()
        if user:
            print(f"[*] Found existing user '{email}'. Resetting user entries for clean demo state...")
            # Clean up old entries and reflections for fresh seed
            db.query(Reflection).filter(Reflection.user_id == user.id).delete()
            db.query(JournalEntry).filter(JournalEntry.user_id == user.id).delete()
            user.hashed_password = hash_password(password)
            user.name = name
            db.commit()
        else:
            print(f"[*] Creating new demo account '{email}'...")
            user = User(
                email=email,
                hashed_password=hash_password(password),
                name=name,
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        # 2. Seed entries with historical timestamps
        now = datetime.now(timezone.utc)
        print(f"[*] Seeding {len(HISTORICAL_JOURNAL_ENTRIES)} published journal entries across the previous 30 days...")
        created_entries = []

        for days_ago, hour, minute, title, mood, tags, content in HISTORICAL_JOURNAL_ENTRIES:
            entry_timestamp = now - timedelta(days=days_ago, hours=(now.hour - hour), minutes=(now.minute - minute))
            
            entry = JournalEntry(
                user_id=user.id,
                title=title,
                content=content,
                mood=mood,
                tags=tags,
                is_draft=False,
                created_at=entry_timestamp,
                updated_at=entry_timestamp,
            )
            db.add(entry)
            db.flush()  # assign ID

            # Run real ML analysis on entry (Phase 1 RoBERTa + Phase 2 Sentence-BERT + Phase 3 spaCy)
            analysis_data = ml_service.analyze_single_entry(content)
            entry_analysis = EntryAnalysis(
                entry_id=entry.id,
                user_id=user.id,
                emotions=analysis_data["emotions"],
                top_emotion=analysis_data["top_emotion"],
                theme=analysis_data["theme"],
                theme_cluster_id=analysis_data["theme_cluster_id"],
                linguistic_features=analysis_data["linguistic_features"],
                created_at=entry_timestamp,
            )
            db.add(entry_analysis)
            created_entries.append(entry)

        db.commit()
        print(f"[OK] Successfully inserted {len(created_entries)} entries and computed real Phase 1-3 ML analyses.")

        # 3. Compute real longitudinal patterns across seeded entries
        print("\n[*] Running Phase 3 longitudinal pattern synthesis across seeded history...")
        # Order entries chronologically for pattern analysis
        ordered_entries = sorted(created_entries, key=lambda e: e.created_at)
        patterns_report = ml_service.compute_longitudinal_patterns(ordered_entries)
        
        status_val = patterns_report.get("status")
        entry_count = patterns_report.get("entry_count", len(ordered_entries))
        time_span = patterns_report.get("time_span_days", 30)
        
        print(f"[OK] Longitudinal Patterns Status: {status_val}")
        print(f"    - Processed Entries: {entry_count}")
        print(f"    - Historical Time Span: {time_span} days")
        
        em_obs = patterns_report.get("emotion_trends", {}).get("observations", [])
        th_obs = patterns_report.get("theme_trends", {}).get("observations", [])
        ling_obs = patterns_report.get("linguistic_trends", {}).get("observations", [])
        
        print(f"    - Emotion Observations Discovered: {len(em_obs)}")
        print(f"    - Theme Observations Discovered:   {len(th_obs)}")
        print(f"    - Linguistic Observations:         {len(ling_obs)}")

        # 4. Generate real Phase 4 & Phase 5 Grounded Reflection
        print("\n[*] Generating Phase 4-5 Grounded Reflection...")
        gen_result = ml_service.generate_reflection(entries=ordered_entries)
        
        if gen_result.get("status") == "success":
            reflection = Reflection(
                user_id=user.id,
                reflection_text=gen_result["reflection"],
                grounded_concepts=gen_result["grounded_concepts"],
                confidence=gen_result["confidence"],
                disclaimer=gen_result["disclaimer"],
                input_summary=gen_result.get("audit"),
                created_at=now,
            )
            db.add(reflection)
            db.commit()
            print(f"[OK] Reflection successfully generated and persisted (Confidence: {gen_result['confidence']:.2f})")
            print(f"    - Grounded in {len(gen_result.get('grounded_concepts', []))} retrieved concepts:")
            for gc in gen_result.get("grounded_concepts", []):
                name = gc.get("name") if isinstance(gc, dict) else gc
                source = gc.get("source") if isinstance(gc, dict) else ""
                print(f"      * {name} ({source})")
        else:
            print(f"[!] Reflection Generation Status: {gen_result.get('status')}: {gen_result.get('message') or gen_result.get('reason')}")

        print("\n" + "=" * 70)
        print("DEMO DATASET SEEDING COMPLETE")
        print("=" * 70)
        print("Log into Pensieve using the Demo Credentials:")
        print(f"  URL:      http://localhost:5173/login")
        print(f"  Email:    {email}")
        print(f"  Password: {password}")
        print("=" * 70)

    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Seed or reset development/demo dataset for Pensieve presentation.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove demo user and all associated entries, analyses, and reflections.",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=DEFAULT_DEMO_EMAIL,
        help=f"Demo account email (default: {DEFAULT_DEMO_EMAIL})",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=DEFAULT_DEMO_PASSWORD,
        help=f"Demo account password (default: {DEFAULT_DEMO_PASSWORD})",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_DEMO_NAME,
        help=f"Demo user name (default: {DEFAULT_DEMO_NAME})",
    )

    args = parser.parse_args()

    if args.reset:
        reset_demo_user(email=args.email)
    else:
        seed_demo_dataset(email=args.email, password=args.password, name=args.name)


if __name__ == "__main__":
    main()
