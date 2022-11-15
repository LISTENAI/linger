import torch
import prettytable as pt


def clamp_with_dynamic(input: torch.Tensor, dynamic_percent: float = 0.9, layer_name: str = "layer", tb_all=None):
    clamp_value = 0
    if dynamic_percent < 1.0:
        x = input.data.clone().cpu()
        x = x.abs().reshape(-1).sort().values
        clamp_index = int(len(x) * dynamic_percent) - 1
        clamp_value = x[clamp_index]
        max_abs = x[-1]
        if len(x) == 1:
            mean_abs = x[-1]
        else:
            mean_abs = x.mean()
        length = max_abs / mean_abs
        versu = max_abs / clamp_value
        if length >= 1:
            index_list = [layer_name, mean_abs,
                          max_abs, length, clamp_value, versu]
            tb_all.add_row(index_list)
        if versu > 10:
            index_list = ["!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!",
                          "!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!"]
            tb_all.add_row(index_list)


def wb_analyse(path: str , save_log_path: str ="wb_analyse.log"):
    if isinstance(path, str):
        model = torch.load(path)
    else:
        model = path

    wb_flile_path = save_log_path
    wb_flile = open(wb_flile_path, 'w+')
    tb_all = pt.PrettyTable()

    tb_all.field_names = ["Layer_name", "Mean", "Max",
                          "Multiple(Max/Mean)", "Dynamic 0.99", "Versu(Max/Dynamic)"]

    for k in model.keys():
        if "running" not in k:
            v = model[k]
            clamp_with_dynamic(v, 0.99, k, tb_all)

    wb_flile.write(str(tb_all))
    wb_flile.close()
