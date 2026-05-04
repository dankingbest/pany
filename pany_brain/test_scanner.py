import asyncio
import os
import sys

# Füge das aktuelle Verzeichnis zum Python-Pfad hinzu
sys.path.append(os.getcwd())

from pany_brain.agent import root_agent
from google.adk.runners import InMemoryRunner
from google.genai import types

async def interactive_test():
    # Pfade zu deinen Bildern
    img1_path = "bill1.jpeg"
    img2_path = "bill2.jpeg"

    # Prüfen ob Dateien existieren
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Fehler: Bilder nicht gefunden!")
        return

    # Bilder laden
    with open(img1_path, "rb") as f:
        img1_data = f.read()
    with open(img2_path, "rb") as f:
        img2_data = f.read()

    # Initialer Content
    user_parts = [
        types.Part.from_bytes(data=img1_data, mime_type="image/jpeg"),
        types.Part.from_bytes(data=img2_data, mime_type="image/jpeg"),
        types.Part.from_text(text="Hier sind die Fotos meines Belegs. Bitte verarbeite sie für meinen Vorrat.")
    ]
    new_message = types.Content(parts=user_parts, role="user")

    runner = InMemoryRunner(agent=root_agent)
    user_id = "test_user"
    session_id = "test_session_interactive"

    print("\n--- Interaktiver Agenten-Test (Scanner) ---\n")
    print("Tipp: Wenn der Agent Fragen stellt, kannst du antworten. Gib 'exit' zum Beenden ein.\n")

    async with runner:
        # Session anlegen
        session = await runner.session_service.get_session(app_name=runner.app_name, user_id=user_id, session_id=session_id)
        if not session:
            await runner.session_service.create_session(app_name=runner.app_name, user_id=user_id, session_id=session_id)

        while True:
            async for event in runner.run_async(
                user_id=user_id, 
                session_id=session_id, 
                new_message=new_message
            ):
                # Text-Ausgabe
                if hasattr(event, 'content') and event.content:
                    for part in event.content.parts:
                        if part.text:
                            print(f"\n[{event.author}]: {part.text}")
                
                # Tool-Visualisierung
                for fc in event.get_function_calls():
                    if fc.name != 'transfer_to_agent':
                        print(f"  -> Tool-Aufruf: {fc.name}({fc.args})")

            # Benutzereingabe für den nächsten Turn
            user_input = input("\nDeine Antwort (oder 'exit'): ")
            if user_input.lower() == 'exit':
                break
            
            new_message = types.Content(parts=[types.Part.from_text(text=user_input)], role="user")

    print("\n--- Test beendet ---")

if __name__ == "__main__":
    asyncio.run(interactive_test())
