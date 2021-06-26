#!/usr/bin/env python


def configure_mpl(plt, figscale=10, fontsize=20, legendfontsize="xx-small"):
    updater_dict = {
        "figure.figsize": (round(1.62 * figscale, 1), round(1 * figscale, 1)),
        "font.size": fontsize,
        "legend.fontsize": legendfontsize,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    for k, v in updater_dict.items():
        plt.rcParams[k] = v
    print("\nMatplotilb has been configured as follows:\n{}.\n".format(updater_dict))


## EOF
