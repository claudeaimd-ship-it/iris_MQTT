import sys
from pathlib import Path

from app.src.AppFactory import AppFactory
from app.src.core.utils.TimestampedFileLogger import TimestampedFileLogger

DEFAULT_VALUES_PATH = Path(__file__).parent / "config" / "default_values.json"
SEQUENCE_PATH       = Path(__file__).parent / "config" / "sequence_001.json"
LOGS_PATH           = Path(__file__).parent / "logs"


def main() -> None:
    with TimestampedFileLogger(str(LOGS_PATH), suffix="inspection"):
        factory  = AppFactory(str(DEFAULT_VALUES_PATH), str(SEQUENCE_PATH))
        executor = factory.create_sequence_executor()
        factory.initialize_hardware()

        try:
            while True:
                part_id = input("Enter part ID (or 'q' to quit): ").strip()
                if part_id.lower() == "q":
                    break
                part = executor.run(part_id)
                print(f"Result: {'OK' if part.overall_status else 'NOK'}")
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            factory.shutdown()
            print("[INFO] Application shut down cleanly.")


if __name__ == "__main__":
    sys.exit(main())
