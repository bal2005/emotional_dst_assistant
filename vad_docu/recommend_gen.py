import os
import json
from neo4j import GraphDatabase
from emotional_dst import process_utterance  # your DST code

# ===== Neo4j Config =====
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "testpassword"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ===== Optional: LLM Config (Gemini) =====
import google.generativeai as genai
GEMINI_KEY = "*****************"
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# ===== Query Neo4j for recommendations =====
def query_recommendations(emotion_name):
    with driver.session() as session:
        query = """
        MATCH (e:Emotion {name: $emotion})-[:MANAGED_BY]->(a:Activity)
        OPTIONAL MATCH (a)-[:HELD_AT]->(p:Place)
        OPTIONAL MATCH (a)-[:SCHEDULED_AS]->(ev:Event)
        OPTIONAL MATCH (a)-[:HAS_REMEDY]->(r:Remedy)
        RETURN e.name AS emotion,
               a.name AS activity,
               p.name AS place, p.city AS city, p.address AS address,
               ev.name AS event, ev.date AS date, ev.time AS time,
               r.name AS remedy, r.type AS remedy_type
        """
        results = session.run(query, emotion=emotion_name)
        return [dict(record) for record in results]

# ===== Use LLM to make a friendly recommendation =====
def generate_recommendation_text(emotion_name, recs):
    if not GEMINI_KEY or not recs:
        return f"For your current emotion '{emotion_name}', here are some activities: " + \
               ", ".join(r['activity'] for r in recs)
    
    # Build a compact summary for the LLM
    structured_summary = json.dumps(recs, indent=2)
    prompt = f"""
    The user is currently feeling {emotion_name}.
    Based on the following structured recommendations from a wellness graph database:
    {structured_summary}
    Suggest a friendly, empathetic recommendation in 2-3 sentences.
    """
    try:
        resp = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"(LLM error) For '{emotion_name}', suggested activities: " + \
               ", ".join(r['activity'] for r in recs)

# ===== Main pipeline =====
def recommend_from_text(user_text):
    # Step 1: Detect emotion
    dst_result = process_utterance(user_text)
    emotion_name = dst_result["mapped_emotion"]

    # Step 2: Query Neo4j
    recs = query_recommendations(emotion_name)

    # Step 3: Generate friendly text
    friendly_text = generate_recommendation_text(emotion_name, recs)

    return {
        "detected_emotion": emotion_name,
        "dst_result": dst_result,
        "recommendations": recs,
        "friendly_text": friendly_text
    }

# ===== Example usage =====
if __name__ == "__main__":
    text = "I am happy"
    output = recommend_from_text(text)
    print(json.dumps(output, indent=2))
