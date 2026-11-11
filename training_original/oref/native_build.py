"""Build the original routing kernels in a source-keyed local cache."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile


def routing_library():
    source = Path(__file__).with_name('native24_original82_kernels.cpp')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    root = Path(os.environ.get('OREF_BUILD_DIR', Path(tempfile.gettempdir()) / 'oref_cepr_build')) / digest
    root.mkdir(parents=True, exist_ok=True)
    output = root / 'routing.so'
    if not output.exists():
        temporary = root / f'routing.{os.getpid()}.so'
        try:
            subprocess.run(['g++', '-O3', '-std=c++17', '-fPIC', '-shared', '-fopenmp',
                            str(source), '-o', str(temporary)], check=True)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    return output
