import os
import sys
import argparse
from src.utils.logger import get_logger

logger = get_logger("DownloadKiTS")

def main():
    parser = argparse.ArgumentParser(description="Download KiTS21 Imaging Data")
    parser.add_argument("--cases", type=int, default=300, help="Number of cases to download")
    args = parser.parse_args()

    kits_repo_dir = "kits21"
    if not os.path.exists(kits_repo_dir):
        logger.error(f"KiTS21 repository not found at {kits_repo_dir}. Clone it first.")
        sys.exit(1)

    logger.info(f"Checking official KiTS21 data directory...")
    # Use official get_imaging script if available
    get_imaging_script = os.path.join(kits_repo_dir, "kits21", "command_line", "get_imaging.py")
    if os.path.exists(get_imaging_script):
        cmd = f"python {get_imaging_script}"
        logger.info(f"Running official download command: {cmd}")
        os.system(cmd)
    else:
        logger.info("Official get_imaging.py not directly found under command_line. Checking kits21 scripts...")
        os.system(f"python -m kits21.command_line.get_imaging")

if __name__ == "__main__":
    main()
