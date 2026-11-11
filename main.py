import argparse
import sys
from pathlib import Path

from returns_agent.contracts import AgentError
from returns_agent.data import DEFAULT_DATA


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Обработать обращения о возврате и запустить eval.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="JSON с обращениями")
    parser.add_argument("--out", type=Path, default=Path("out"), help="Каталог постоянного состояния")
    args = parser.parse_args()
    from returns_agent.runner import run
    try:
        run(args.data, args.out)
        from eval import evaluate_files
        return evaluate_files(args.data, args.out)
    except AgentError as exc:
        print(f"Ошибка: {exc.code}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
