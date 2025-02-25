import argparse
from config import *
from autonomous import autonomous_conversation_loop
from brain import Brain

def run_autonomous_mode():
    """
    Runs the chatbot in fully autonomous mode, where Ouro and Brain talk
    indefinitely until you press Ctrl+C.
    """
    print("🤖 Running in Autonomous Mode...")
    autonomous_conversation_loop()

def run_interactive_mode():
    """
    Runs the chatbot in a simpler, user-driven interactive mode.
    You type messages, and Brain generates replies (without Ouro).
    """
    print("💬 Running in Interactive Mode...")
    brain = Brain()
    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("👋 Exiting interactive mode. Goodbye!")
                break
            # In interactive mode, we give an empty context for simplicity
            response = brain.generate_reply(user_input, "")
            print(f"Brain: {response}\n")
        except KeyboardInterrupt:
            print("\n👋 Exiting interactive mode. Goodbye!")
            break

def main():
    """
    Main function to select the operating mode:
     1) autonomous   - Ouro & Brain talk infinitely
     2) interactive  - You talk to Brain directly
    """
    parser = argparse.ArgumentParser(description="Run Ouro LLM Chatbot in different modes.")
    parser.add_argument("--mode", choices=["autonomous", "interactive"], default="interactive",
                        help="Choose 'autonomous' or 'interactive' mode.")
    args = parser.parse_args()

    if args.mode == "autonomous":
        run_autonomous_mode()
    else:
        run_interactive_mode()

if __name__ == "__main__":
    main()