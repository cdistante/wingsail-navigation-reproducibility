"""Record actual simulator/media runtime without importing numerical packages."""
from __future__ import annotations
import hashlib,importlib.metadata,json,os,platform,subprocess,sys
import imageio_ffmpeg
from pathlib import Path
STUDY=Path(__file__).resolve().parents[1]
def call(*args):
    return subprocess.check_output(args,text=True).strip()
def main():
    packages={}
    for name in ("numpy","scipy","casadi","torch","matplotlib","xarray","pandas","imageio-ffmpeg","acados-template"):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]="source checkout / not registered as a distribution"
    source=os.environ.get("ACADOS_SOURCE_DIR","")
    record={"python":sys.version,"platform":platform.platform(),"machine":platform.machine(),
            "processor":platform.processor(),"logical_cpus":os.cpu_count(),"packages":packages,
            "acados":{"source":source,"commit":call("git","-C",source,"rev-parse","HEAD"),
                      "describe":call("git","-C",source,"describe","--tags","--always")},
            "ffmpeg":call(imageio_ffmpeg.get_ffmpeg_exe(),"-version").splitlines()[0],
            "ffmpeg_executable":imageio_ffmpeg.get_ffmpeg_exe(),
            "execution":"Eight native full-loop attempts in two strict four-worker waves, all retained. Each has an isolated acados build and numerical thread limits1. Complete choose and choose-through-accept/stop attempts are separately timed; virtual vessel/AIS time pauses during computation.",
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (STUDY/"software_versions.json").write_text(json.dumps(record,indent=2)+"\n")
    print(STUDY/"software_versions.json")
if __name__=="__main__":main()
