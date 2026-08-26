import argparse
import json
import os
from pathlib import Path
import subprocess


def resolve_latest(output_root):
    latest = output_root / 'latest'
    if latest.exists():
        return latest.resolve()

    latest_text = output_root / 'LATEST.txt'
    if latest_text.is_file():
        return Path(
            latest_text.read_text(encoding='utf-8').strip()
        ).resolve()

    raise FileNotFoundError(
        f'No completed dataset run found below {output_root}'
    )


def validate_dataset(run_dir, summary):
    dataset_yaml = Path(summary['dataset_yaml'])
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f'Missing {dataset_yaml}')

    checked_labels = 0
    for split in ('train', 'val', 'test'):
        label_dir = run_dir / 'dataset' / 'labels' / split
        image_dir = run_dir / 'dataset' / 'images' / split
        if not label_dir.is_dir() or not image_dir.is_dir():
            raise FileNotFoundError(
                f'Missing YOLO split directories for {split}'
            )

        for label_path in label_dir.glob('*.txt'):
            image_path = image_dir / f'{label_path.stem}.jpg'
            if not image_path.is_file():
                raise FileNotFoundError(
                    f'No image for label {label_path}'
                )
            for line in label_path.read_text(
                encoding='utf-8'
            ).splitlines():
                fields = line.split()
                if len(fields) != 5 or fields[0] != '0':
                    raise ValueError(
                        f'Invalid YOLO label in {label_path}: {line}'
                    )
                values = [float(value) for value in fields[1:]]
                if any(value < 0.0 or value > 1.0 for value in values):
                    raise ValueError(
                        f'Out-of-range YOLO label in {label_path}: {line}'
                    )
                checked_labels += 1

    return checked_labels


def open_preview(preview_path):
    if not preview_path.is_file():
        raise FileNotFoundError(f'Missing preview image: {preview_path}')

    try:
        windows_path = subprocess.check_output(
            ['wslpath', '-w', str(preview_path)],
            text=True,
        ).strip()
        subprocess.Popen(['explorer.exe', windows_path])
    except (FileNotFoundError, subprocess.SubprocessError):
        print(f'Open this image manually: {preview_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Validate and display the latest synthetic YOLO dataset.'
    )
    parser.add_argument(
        '--output-root',
        default='~/.ros/autonomy_lab/datasets/inspection_panel',
        help='Dataset output root containing the latest link.',
    )
    parser.add_argument(
        '--no-open',
        action='store_true',
        help='Validate and print the report without opening preview.jpg.',
    )
    arguments = parser.parse_args()

    output_root = Path(
        os.path.expanduser(arguments.output_root)
    ).resolve()
    run_dir = resolve_latest(output_root)
    summary_path = run_dir / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    label_count = validate_dataset(run_dir, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Validated {label_count} YOLO bounding boxes.')

    if not arguments.no_open:
        open_preview(Path(summary['preview']))


if __name__ == '__main__':
    main()
