"""Create an agent, run it, and stream its output.

Usage:
    python examples/python/agents.py
"""

from kothagpt import KothaGPT


def main() -> None:
    with KothaGPT() as client:
        agent = client.agents.create(
            {
                "name": "bangla-helper",
                "description": "A Bangla-speaking assistant.",
                "instructions": "সবসময় বাংলায় উত্তর দাও এবং সংক্ষিপ্ত রাখো।",
                "tools": ["calculator"],
            }
        )
        print(f"Created agent {agent.id}\n")

        run = client.agents.run(agent.id, "১২ ও ৮ এর যোগফল কত?")
        print(f"Run {run.id}: {run.status}\n{run.output}")

        print("\nStreaming run:")
        for event in client.agents.stream(agent.id, "৩ এবং ৪ গুণ করো"):
            if event.get("event") == "run.delta":
                print(event.get("delta", ""), end="", flush=True)
            elif event.get("event") == "run.completed":
                print()


if __name__ == "__main__":
    main()
