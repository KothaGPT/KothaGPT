from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

from kothagpt import KothaGPT, types
from kothagpt import __version__ as sdk_version

CLI_VERSION = "0.1.0"


def _client() -> KothaGPT:
    return KothaGPT(
        base_url=os.getenv("KOTHAGPT_API_URL", "http://localhost:8000"),
        api_key=os.getenv("KOTHAGPT_API_KEY"),
    )


def cmd_models(args: argparse.Namespace) -> int:
    with _client() as client:
        models = client.models.list()
        for model in models:
            print(f"{model.id}\t{model.context_window}\t{model.description}")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    with _client() as client:
        if args.stream:
            for chunk in client.chat.stream(
                [{"role": "user", "content": args.message}],
                model=args.model,
                temperature=args.temperature,
            ):
                sys.stdout.write(chunk.delta)
                sys.stdout.flush()
            print()
        else:
            response = client.chat.create(
                [{"role": "user", "content": args.message}],
                model=args.model,
                temperature=args.temperature,
            )
            print(response.text)
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    with _client() as client:
        response = client.embeddings.create(args.text)
        if args.json:
            print(json.dumps(response.model_dump()))
        else:
            for item in response.data:
                print(json.dumps(item.embedding))
    return 0


def cmd_rerank(args: argparse.Namespace) -> int:
    with _client() as client:
        response = client.rerank.create(args.query, args.document, top_n=args.top_n)
        if args.json:
            print(json.dumps(response.model_dump()))
        else:
            for result in response.results:
                print(f"{result.index}\t{result.relevance_score}\t{result.document}")
    return 0


def cmd_tools_list(args: argparse.Namespace) -> int:
    with _client() as client:
        for tool in client.tools.list():
            fn = tool.function
            print(f"{fn.name}\t{fn.description}")
    return 0


def cmd_tools_invoke(args: argparse.Namespace) -> int:
    arguments: dict[str, object] = {}
    for raw in args.arg:
        key, _, value = raw.partition("=")
        arguments[key] = value
    with _client() as client:
        result = client.tools.invoke(args.name, arguments)
        print(json.dumps(result, ensure_ascii=False))
    return 0


def cmd_agents_create(args: argparse.Namespace) -> int:
    spec = types.AgentSpec(
        name=args.name,
        description=args.description,
        instructions=args.instructions,
        model=args.model,
        tools=args.tool or [],
    )
    with _client() as client:
        agent = client.agents.create(spec)
        print(json.dumps(agent.model_dump(), ensure_ascii=False))
    return 0


def cmd_agents_list(args: argparse.Namespace) -> int:
    with _client() as client:
        for agent in client.agents.list():
            print(f"{agent.id}\t{agent.name}\t{agent.model}\ttools={agent.tools}")
    return 0


def cmd_agents_run(args: argparse.Namespace) -> int:
    with _client() as client:
        if args.stream:
            for event in client.agents.stream(args.agent_id, args.message):
                if event.get("event") == "run.delta":
                    sys.stdout.write(event.get("delta", ""))
                    sys.stdout.flush()
                elif event.get("event") == "run.completed":
                    print()
            return 0
        run = client.agents.run(args.agent_id, args.message)
        if args.json:
            print(json.dumps(run.model_dump(), ensure_ascii=False))
        else:
            print(run.output or "")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kothagpt", description="Kotha GPT CLI")
    parser.add_argument(
        "--version", action="version", version=f"kothagpt {CLI_VERSION} (sdk {sdk_version})"
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="API base URL (default: $KOTHAGPT_API_URL or http://localhost:8000)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("models", help="List available models")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("chat", help="Send a chat message")
    p.add_argument("message")
    p.add_argument("--model", default="kothagpt")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--stream", action="store_true")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("embed", help="Create text embeddings")
    p.add_argument("text", nargs="+")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("rerank", help="Rerank documents against a query")
    p.add_argument("--query", required=True)
    p.add_argument("--document", action="append", required=True, dest="document")
    p.add_argument("--top-n", type=int, default=None, dest="top_n")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_rerank)

    tools = sub.add_parser("tools", help="Tool discovery and invocation")
    tools_sub = tools.add_subparsers(dest="tools_command", required=True)
    t = tools_sub.add_parser("list", help="List tools")
    t.set_defaults(func=cmd_tools_list)
    t = tools_sub.add_parser("invoke", help="Invoke a tool")
    t.add_argument("name")
    t.add_argument("--arg", action="append", default=[], metavar="KEY=VALUE")
    t.set_defaults(func=cmd_tools_invoke)

    agents = sub.add_parser("agents", help="Agent management and execution")
    agents_sub = agents.add_subparsers(dest="agents_command", required=True)
    a = agents_sub.add_parser("create", help="Create an agent")
    a.add_argument("--name", required=True)
    a.add_argument("--description", default=None)
    a.add_argument("--instructions", default=None)
    a.add_argument("--model", default="kothagpt")
    a.add_argument("--tool", action="append", default=[], dest="tool")
    a.set_defaults(func=cmd_agents_create)
    a = agents_sub.add_parser("list", help="List agents")
    a.set_defaults(func=cmd_agents_list)
    a = agents_sub.add_parser("run", help="Run an agent")
    a.add_argument("agent_id")
    a.add_argument("message")
    a.add_argument("--stream", action="store_true")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_agents_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "api_url", None):
        os.environ["KOTHAGPT_API_URL"] = args.api_url
    try:
        return args.func(args)
    except BrokenPipeError:
        with contextlib.suppress(Exception):
            sys.stdout.close()
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
