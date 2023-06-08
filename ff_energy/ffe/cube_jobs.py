from ff_energy.ffe.templates import g_template, esp_cubes_template
from pathlib import Path

def XYZ_to_job(path,
               method="PBE1PBE",
               basis="aug-cc-pVDZ",
               outdir=None):
    xyzstr = "".join(open(path).readlines()[2:])
    #  com file
    g = g_template.render(XYZ_STRING=xyzstr,
                          KEY=path.name,
                          METHOD=method,
                          BASIS=basis)
    #  slurm file
    e = esp_cubes_template.render(KEY=path.name)

    if outdir is None:
        outdir = path.parent
    else:
        outdir = Path(outdir)

    with open(outdir / f"{path.name}.com", "w") as f:
        f.write(g)
    with open(outdir / f"{path.name}.sh", "w") as f:
        f.write(e)




if __name__ == "__main__":
    for _ in Path("/home/boittier/pcbach/dcm/nms/").glob("*.xyz"):
        XYZ_to_job(_)

