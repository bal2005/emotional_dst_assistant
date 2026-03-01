import os
from typing import Dict, List
from neo4j import GraphDatabase


# ---------------------------
# Neo4j connection settings
# ---------------------------
# Example (local):
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=your_password
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "dbpwd@123")


# ---------------------------
# Mock dataset (~30 situations)
# Each situation is a record:
# Emotion, Event, Activity, Place, Remedy
# ---------------------------
DATA: List[Dict] = [
    # stressed / exams cluster
    {"emotion": "Stressed", "event": "Exam", "activity": "Walk", "activity_category": "outdoor",
     "place": "Children's Park", "area": "Anna Nagar", "city": "Chennai", "remedy": "Deep Breathing", "remedy_type": "breathing"},
    {"emotion": "Stressed", "event": "Exam", "activity": "Cycling", "activity_category": "outdoor",
     "place": "Anna Nagar Tower Park", "area": "Anna Nagar", "city": "Chennai", "remedy": "Mindful Break", "remedy_type": "mindfulness"},
    {"emotion": "Stressed", "event": "Project Deadline", "activity": "Quiet Reading", "activity_category": "indoor",
     "place": "Anna Centenary Library", "area": "Kotturpuram", "city": "Chennai", "remedy": "Time Boxing", "remedy_type": "planning"},
    {"emotion": "Stressed", "event": "Work Pressure", "activity": "Nature Walk", "activity_category": "outdoor",
     "place": "Semmozhi Poonga", "area": "Teynampet", "city": "Chennai", "remedy": "Deep Breathing", "remedy_type": "breathing"},
    {"emotion": "Stressed", "event": "Interview", "activity": "Meditation", "activity_category": "indoor",
     "place": "Home", "area": "—", "city": "Chennai", "remedy": "Guided Meditation", "remedy_type": "meditation"},

    # anxious cluster
    {"emotion": "Anxious", "event": "Exam", "activity": "Slow Walk", "activity_category": "outdoor",
     "place": "Nageswara Rao Park", "area": "Mylapore", "city": "Chennai", "remedy": "Deep Breathing", "remedy_type": "breathing"},
    {"emotion": "Anxious", "event": "Public Speaking", "activity": "Breathing Exercise", "activity_category": "indoor",
     "place": "Home", "area": "—", "city": "Chennai", "remedy": "Box Breathing", "remedy_type": "breathing"},
    {"emotion": "Anxious", "event": "Uncertain Future", "activity": "Journaling", "activity_category": "indoor",
     "place": "Quiet Cafe", "area": "Adyar", "city": "Chennai", "remedy": "Journaling Prompt", "remedy_type": "journaling"},
    {"emotion": "Anxious", "event": "Family Pressure", "activity": "Temple Visit", "activity_category": "spiritual",
     "place": "Kapaleeshwarar Temple", "area": "Mylapore", "city": "Chennai", "remedy": "Grounding Practice", "remedy_type": "mindfulness"},
    {"emotion": "Anxious", "event": "Health Worry", "activity": "Light Stretching", "activity_category": "indoor",
     "place": "Home", "area": "—", "city": "Chennai", "remedy": "Progressive Muscle Relaxation", "remedy_type": "relaxation"},

    # sad cluster
    {"emotion": "Sad", "event": "Breakup", "activity": "Beach Walk", "activity_category": "outdoor",
     "place": "Elliot's Beach", "area": "Besant Nagar", "city": "Chennai", "remedy": "Talk to a Friend", "remedy_type": "social"},
    {"emotion": "Sad", "event": "Loneliness", "activity": "Music Listening", "activity_category": "indoor",
     "place": "Home", "area": "—", "city": "Chennai", "remedy": "Comfort Playlist", "remedy_type": "music"},
    {"emotion": "Sad", "event": "Failure", "activity": "Park Sit & Reflect", "activity_category": "outdoor",
     "place": "Chetpet Eco Park", "area": "Chetpet", "city": "Chennai", "remedy": "Self-Compassion Note", "remedy_type": "journaling"},
    {"emotion": "Sad", "event": "Grief", "activity": "Quiet Prayer", "activity_category": "spiritual",
     "place": "San Thome Basilica", "area": "Santhome", "city": "Chennai", "remedy": "Breathing + Prayer", "remedy_type": "spiritual"},
    {"emotion": "Sad", "event": "Homesick", "activity": "Call Family", "activity_category": "social",
     "place": "Home", "area": "—", "city": "Chennai", "remedy": "Talk to a Friend", "remedy_type": "social"},

    # lonely cluster
    {"emotion": "Lonely", "event": "Weekend Alone", "activity": "Walk", "activity_category": "outdoor",
     "place": "Marina Beach", "area": "Triplicane", "city": "Chennai", "remedy": "Join a Group Activity", "remedy_type": "social"},
    {"emotion": "Lonely", "event": "New City", "activity": "Community Meetup", "activity_category": "social",
     "place": "Phoenix Marketcity", "area": "Velachery", "city": "Chennai", "remedy": "Join a Group Activity", "remedy_type": "social"},
    {"emotion": "Lonely", "event": "No Close Friends", "activity": "Coffee + Reading", "activity_category": "indoor",
     "place": "Quiet Cafe", "area": "Alwarpet", "city": "Chennai", "remedy": "Message Someone", "remedy_type": "social"},
    {"emotion": "Lonely", "event": "Missing Friends", "activity": "Visit Bookstore", "activity_category": "indoor",
     "place": "Express Avenue Mall", "area": "Royapettah", "city": "Chennai", "remedy": "Message Someone", "remedy_type": "social"},
    {"emotion": "Lonely", "event": "Festival Season", "activity": "Temple Visit", "activity_category": "spiritual",
     "place": "Parthasarathy Temple", "area": "Triplicane", "city": "Chennai", "remedy": "Join a Group Activity", "remedy_type": "social"},

    # bored cluster
    {"emotion": "Bored", "event": "Free Evening", "activity": "Movie", "activity_category": "leisure",
     "place": "Sathyam Cinemas", "area": "Royapettah", "city": "Chennai", "remedy": "Try Something New", "remedy_type": "activation"},
    {"emotion": "Bored", "event": "No Plans", "activity": "Museum Visit", "activity_category": "leisure",
     "place": "Government Museum", "area": "Egmore", "city": "Chennai", "remedy": "Try Something New", "remedy_type": "activation"},
    {"emotion": "Bored", "event": "Weekend", "activity": "Boating", "activity_category": "leisure",
     "place": "Muttukadu Backwaters", "area": "ECR", "city": "Chennai", "remedy": "Try Something New", "remedy_type": "activation"},
    {"emotion": "Bored", "event": "Routine", "activity": "Planetarium Visit", "activity_category": "leisure",
     "place": "Birla Planetarium", "area": "Kotturpuram", "city": "Chennai", "remedy": "Try Something New", "remedy_type": "activation"},
    {"emotion": "Bored", "event": "Free Day", "activity": "Cultural Visit", "activity_category": "leisure",
     "place": "DakshinaChitra", "area": "ECR", "city": "Chennai", "remedy": "Try Something New", "remedy_type": "activation"},

    # angry cluster
    {"emotion": "Angry", "event": "Argument", "activity": "Brisk Walk", "activity_category": "outdoor",
     "place": "Guindy National Park", "area": "Guindy", "city": "Chennai", "remedy": "Cooldown Walk", "remedy_type": "regulation"},
    {"emotion": "Angry", "event": "Traffic", "activity": "Breathing Exercise", "activity_category": "indoor",
     "place": "Home", "area": "—", "city": "Chennai", "remedy": "Box Breathing", "remedy_type": "breathing"},
    {"emotion": "Angry", "event": "Frustration", "activity": "Workout", "activity_category": "fitness",
     "place": "Gym", "area": "Guindy", "city": "Chennai", "remedy": "Physical Release", "remedy_type": "fitness"},
    {"emotion": "Angry", "event": "Criticism", "activity": "Journaling", "activity_category": "indoor",
     "place": "Home", "area": "—", "city": "Chennai", "remedy": "Write it Out", "remedy_type": "journaling"},
    {"emotion": "Angry", "event": "Office Conflict", "activity": "Walk", "activity_category": "outdoor",
     "place": "Semmozhi Poonga", "area": "Teynampet", "city": "Chennai", "remedy": "Cooldown Walk", "remedy_type": "regulation"},

    # happy cluster
    {"emotion": "Happy", "event": "Celebration", "activity": "Dinner Out", "activity_category": "social",
     "place": "T Nagar", "area": "T Nagar", "city": "Chennai", "remedy": "Savor the Moment", "remedy_type": "mindfulness"},
    {"emotion": "Happy", "event": "Good News", "activity": "Beach Visit", "activity_category": "outdoor",
     "place": "Thiruvanmiyur Beach", "area": "Thiruvanmiyur", "city": "Chennai", "remedy": "Share with Someone", "remedy_type": "social"},
    {"emotion": "Happy", "event": "Weekend Fun", "activity": "Picnic", "activity_category": "outdoor",
     "place": "Theosophical Society", "area": "Adyar", "city": "Chennai", "remedy": "Savor the Moment", "remedy_type": "mindfulness"},
    {"emotion": "Happy", "event": "Achievement", "activity": "Photo Walk", "activity_category": "leisure",
     "place": "Valluvar Kottam", "area": "Nungambakkam", "city": "Chennai", "remedy": "Celebrate Small Wins", "remedy_type": "activation"},
    {"emotion": "Happy", "event": "Relax Day", "activity": "Coffee", "activity_category": "social",
     "place": "Quiet Cafe", "area": "Besant Nagar", "city": "Chennai", "remedy": "Share with Someone", "remedy_type": "social"},
]


# ---------------------------
# Cypher: constraints
# ---------------------------
CONSTRAINTS = [
    "CREATE CONSTRAINT emotion_name IF NOT EXISTS FOR (n:Emotion) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT event_name IF NOT EXISTS FOR (n:Event) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT activity_name IF NOT EXISTS FOR (n:Activity) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT remedy_name IF NOT EXISTS FOR (n:Remedy) REQUIRE n.name IS UNIQUE",
    # Place uniqueness: name+area+city (Neo4j doesn't support composite UNIQUE in all versions)
    # We'll approximate with a synthetic key:
    "CREATE CONSTRAINT place_key IF NOT EXISTS FOR (n:Place) REQUIRE n.key IS UNIQUE",
]


# ---------------------------
# Cypher: upsert record
# ---------------------------
UPSERT_CYPHER = """
MERGE (em:Emotion {name: $emotion})

MERGE (ev:Event {name: $event})
MERGE (ev)-[:EVOKES]->(em)

MERGE (a:Activity {name: $activity})
ON CREATE SET a.category = $activity_category
ON MATCH SET a.category = coalesce(a.category, $activity_category)

MERGE (em)-[:SUGGESTS]->(a)

MERGE (p:Place {key: $place_key})
ON CREATE SET p.name = $place, p.area = $area, p.city = $city
ON MATCH SET p.name = coalesce(p.name, $place),
              p.area = coalesce(p.area, $area),
              p.city = coalesce(p.city, $city)

MERGE (a)-[:AT]->(p)

MERGE (r:Remedy {name: $remedy})
ON CREATE SET r.type = $remedy_type
ON MATCH SET r.type = coalesce(r.type, $remedy_type)

MERGE (em)-[:HELPED_BY]->(r)
MERGE (a)-[:SUPPORTED_BY]->(r)
"""


def make_place_key(place: str, area: str, city: str) -> str:
    # stable unique key for Place
    return f"{place.strip().lower()}|{area.strip().lower()}|{city.strip().lower()}"


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # Create constraints
        for c in CONSTRAINTS:
            session.run(c)

        # Upsert records
        for row in DATA:
            params = dict(row)
            params["place_key"] = make_place_key(row["place"], row["area"], row["city"])
            session.run(UPSERT_CYPHER, params)

        # Quick sanity counts
        counts = session.run("""
        MATCH (em:Emotion)
        WITH count(em) AS emotions
        MATCH (ev:Event)
        WITH emotions, count(ev) AS events
        MATCH (a:Activity)
        WITH emotions, events, count(a) AS activities
        MATCH (p:Place)
        WITH emotions, events, activities, count(p) AS places
        MATCH (r:Remedy)
        RETURN emotions, events, activities, places, count(r) AS remedies
        """).single()

        print("✅ Done! Node counts:", dict(counts))

        rel_counts = session.run("""
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(*) AS c
        ORDER BY c DESC
        """).data()

        print("✅ Relationship counts:")
        for r in rel_counts:
            print(f"  {r['rel_type']}: {r['c']}")

    driver.close()


if __name__ == "__main__":
    main()