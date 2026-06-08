import os
from datetime import datetime
from gurobipy import GRB

def write_iis_with_context(
    model,
    *,
    base_dir,
    topo_name,
    run_index,
    stage,              # e.g. "controller_selection", "mcf_arc_init", "baseline1"
    extra_tag="",       # optional (e.g. "load_scale_1.2")
):
    """
    Writes IIS (.ilp) with rich context so no file is overwritten.
    """

    os.makedirs(base_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    fname = (
        f"{topo_name}"
        f"_run{run_index:03d}"
        f"_{stage}"
    )

    if extra_tag:
        fname += f"_{extra_tag}"

    fname += f"_{ts}.ilp"

    path = os.path.join(base_dir, fname)

    try:
        model.setParam(GRB.Param.Presolve, 0)
        model.setParam(GRB.Param.IISMethod, 1)
        model.computeIIS()
        model.write(path)

        print(f"[IIS] written → {path}")
        return path

    except Exception as e:
        print(f"[IIS ERROR] {topo_name} run {run_index} stage {stage}: {e}")
        return None